import torch
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
from Utils import evaluate_map, evaluate_pr

def _now():
    return time.strftime("%Y%m%d-%H%M%S")

def _save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, config, class_names, metrics):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "config": asdict(config),
        "class_names": class_names,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, scaler, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return (checkpoint.get("epoch", -1) + 1, checkpoint.get("class_names"),
            checkpoint.get("config"), checkpoint.get("metrics"))

@dataclass
class EvalCtx:
    loader: object
    class_names: list

def train(model, train_loader, config, device="cuda", val_fn=None,
          eval_ctx: EvalCtx | None = None, resume_path=None):
    run_root = Path(getattr(config.train, "checkpoint_dir", "runs"))
    run_dir = run_root / _now()
    run_dir.mkdir(parents=True, exist_ok=True)
    (best_score, best_path) = (-float("inf"), None)
    kept = deque(maxlen=getattr(config.train, "keep_last_k", 0))

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.train.lr,
                      weight_decay=config.train.weight_decay)

    use_amp = (torch.device(device).type == "cuda")

    scaler = amp.GradScaler('cuda') if use_amp else None

    ac = amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp \
        else nullcontext()

    scheduler = CosineAnnealingLR(optimizer, T_max=config.train.epochs)

    history = {
        "epoch": [], "train_loss": [], "train_cls": [], "train_reg": [],
        "val_loss": [], "val_cls": [], "val_reg": [], "map75": [],
        "map50": [], "map30": [], "precision": [], "recall": []
    }

    start_epoch = 0
    if resume_path:
        start_epoch, _, _, _ = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            path=resume_path,
            device=device
        )

    metric_name = getattr(config.train, "early_metric", getattr(config.train, "best_metric", "map50"))
    mode_max = metric_name != "val_loss"  # mAP → max, val_loss → min
    min_delta = float(getattr(config.train, "early_min_delta", 0.01))
    patience = int(getattr(config.train, "early_patience", 0))
    use_es = bool(getattr(config.train, "early_stop", False)) and patience > 0

    early_best = (-float("inf")) if mode_max else (float("inf"))
    wait = 0

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        num_batches = 0
        # Train Loss
        loss_sum = 0.0
        classification_sum = 0.0
        regression_sum = 0.0

        acc = getattr(config.train, "accum_steps", 1)

        for i, (imgs, targets) in enumerate(train_loader, start=1):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with ac:
                loss_dict = model(imgs, targets)
                raw_classification_loss = loss_dict["classification"]
                raw_regression_loss = loss_dict["bbox_regression"]
                raw_loss = raw_classification_loss + raw_regression_loss

                loss = raw_loss / acc  # scale ONLY for backwards

            if scaler:
                scaler.scale(loss).backward()
                if i % acc == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if i % acc == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            num_batches += 1
            loss_sum += float(raw_loss.item())
            classification_sum += float(raw_classification_loss.item())
            regression_sum += float(raw_regression_loss.item())

        if (num_batches % acc) != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = loss_sum / max(num_batches, 1)
        train_classification = classification_sum / max(num_batches, 1)
        train_regression = regression_sum / max(num_batches, 1)

        """ EVALUATION """
        val_loss = None
        val_classification = None
        val_regression = None

        if val_fn:
            val_loss, val_classification, val_regression = val_fn(model, device)
            print(f"{epoch=}: \n"
                  f"train_loss={train_loss:.4f}     cls={train_classification:.4f}      reg={train_regression:.4f}  \n"
                  f"val_loss={val_loss:.4f}         cls={val_classification:.4f}        reg={val_regression:.4f}")
        else:
            print(f"{epoch=}: "
                  f"train_loss={train_loss:.4f}    cls={train_classification:.4f}     reg={train_regression:.4f}")

        # Decide if running with max batches or no max
        full_epoch = bool(getattr(config.eval, "full_every", 0)) and (epoch % config.eval.full_every == 0)
        full_last = bool(getattr(config.eval, "full_on_last", True)) and (epoch == config.train.epochs - 1)
        maxb = None if (full_epoch or full_last) else getattr(config.eval, "map_max_batches", None)
        is_full = (maxb is None)
        # Print which run it is (max batches or none)
        if getattr(config.eval, "do_map", False) and eval_ctx and (epoch % config.eval.map_every == 0):
            print(f"[eval] mode={'full' if maxb is None else f'sampled({maxb} batches)'}")

        # mAP block
        map75 = None
        map50 = None
        map30 = None
        if (getattr(config.eval, "do_map", False) and eval_ctx and
                (epoch % config.eval.map_every == 0)):

            # mAP@0.75
            r75 = evaluate_map(model=model, loader=eval_ctx.loader,
                               class_names=eval_ctx.class_names, device=device,
                               iou_thr=0.75, max_batches=maxb)
            map75 = r75['mAP@0.5']
            print(f"mAP@0.75={map75:.3f} ")

            # mAP@0.5
            r50 = evaluate_map(model=model, loader=eval_ctx.loader,
                               class_names=eval_ctx.class_names, device=device,
                               iou_thr=config.eval.map_iou, max_batches=maxb
                               )
            map50 = r50['mAP@0.5']
            print(f"mAP@0.5={map50:.3f} ")

            # mAP@0.3
            r30 = evaluate_map(model=model, loader=eval_ctx.loader,
                               class_names=eval_ctx.class_names, device=device,
                               iou_thr=0.3, max_batches=maxb)
            map30 = r30['mAP@0.5']
            print(f"mAP@0.3={map30:.3f} ")

        # Precision Recall
        pr = {"precision": None, "recall": None}
        if eval_ctx:
            pr = evaluate_pr(model=model, loader=eval_ctx.loader,
                             device=device, iou_thr=config.eval.map_iou,
                             score_thresh=getattr(model, "score_thresh", None),
                             max_batches=maxb)
            print(f"Precision@{config.eval.map_iou}={pr['precision']:.3f}     "
                  f"Recall@{config.eval.map_iou}={pr['recall']:.3f} \n")

        # Append to Lists in set
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_cls"].append(train_classification)
        history["train_reg"].append(train_regression)
        history["val_loss"].append(val_loss)
        history["val_cls"].append(val_classification)
        history["val_reg"].append(val_regression)
        history["map75"].append(map75)
        history["map50"].append(map50)
        history["map30"].append(map30)
        history["precision"].append(pr['precision'])
        history["recall"].append(pr['recall'])

        # Saving checkpoint
        metrics = {
            "train_loss": train_loss,
            "train_classification": train_classification,
            "train_regression": train_regression,
            "val_loss": val_loss,
            "val_classification": val_classification,
            "val_regression": val_regression,
            "map75": map75,
            "map50": map50,
            "map30": map30,
            "precision": pr['precision'],
            "recall": pr['recall']
        }



        # Always save last
        _save_checkpoint(path=run_dir / "last.pth", model=model,
                         optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                         epoch=epoch, config=config,
                         class_names=eval_ctx.class_names if eval_ctx else [],
                         metrics=metrics)

        # Periodic save
        if getattr(config.train, "save_every", 0) and (epoch % config.train.save_every == 0):
            ep_name = f"epoch{epoch:04d}_m{(map50 if map50 is not None else 0):.3f}.pth"
            ep_path = run_dir / ep_name
            _save_checkpoint(path=ep_path, model=model,
                             optimizer=optimizer, scheduler=scheduler,
                             scaler=scaler, epoch=epoch, config=config,
                             class_names=eval_ctx.class_names if eval_ctx else [],
                             metrics=metrics)

            # Rotate
            k = getattr(config.train, "keep_last_k", 0)
            if k:
                if len(kept) == k:
                    old = kept.popleft()
                    try:
                        os.remove(old)
                    except OSError:
                        pass
                kept.append(ep_path)

        # Best by metric
        metric_name = getattr(config.train, "best_metric", "map50")
        score = metrics.get(metric_name)
        if getattr(config.train, "save_best", False) and (score is not None) and (score > best_score):
            best_score = score
            best_path = run_dir / f"best_{metric_name}.pth"
            _save_checkpoint(path=best_path, model=model,
                             optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                             epoch=epoch, config=config,
                             class_names=eval_ctx.class_names if eval_ctx else [],
                             metrics=metrics)

        if use_es and is_full:
            curr = metrics.get(metric_name)
            if curr is not None:
                improved = (curr - early_best > min_delta) if mode_max else \
                    (early_best - curr > min_delta)
                if improved:
                    early_best = curr
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"[early stop] no {metric_name} improvement > {min_delta} for {patience} full evals")
                        break


        scheduler.step()

    return history
