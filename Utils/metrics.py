import torch
import numpy as np
import pandas as pd
from pathlib import Path

def box_iou(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    area_a = (a[:,2]-a[:,0]).clamp(min=0) * (a[:,3]-a[:,1]).clamp(min=0)
    area_b = (b[:,2]-b[:,0]).clamp(min=0) * (b[:,3]-b[:,1]).clamp(min=0)
    lt = torch.max(a[:,None,:2], b[:,:2])
    rb = torch.min(a[:,None,2:], b[:,2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0]*wh[:,:,1]
    return inter / (area_a[:,None] + area_b - inter + 1e-6)

def precision_recall(outputs, targets, iou_thr=0.5):
    """

    :param outputs:
    :param targets:
    :param iou_thr:
    :return:
    """
    tp, fp, fn = 0, 0, 0
    for out, tgt in zip(outputs, targets):
        if len(out["boxes"])==0 and len(tgt["boxes"])==0: continue
        if len(out["boxes"])==0: fn += len(tgt["boxes"]); continue
        if len(tgt["boxes"])==0: fp += len(out["boxes"]); continue
        ious = box_iou(out["boxes"], tgt["boxes"])
        assigned = set()
        for j in range(len(out["boxes"])):
            i = torch.argmax(ious[j])
            if (ious[j, i] >= iou_thr and i.item() not in assigned and
                    out["labels"][j]==tgt["labels"][i]):
                tp += 1; assigned.add(i.item())
            else:
                fp += 1
        fn += (len(tgt["boxes"]) - len(assigned))
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    return {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec}

def _voc_ap(rec, prec):
    mrec  = np.concatenate(([0.0], rec,  [1.0]))
    mprec = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mprec.size-1, 0, -1):
        mprec[i-1] = max(mprec[i-1], mprec[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i+1]-mrec[i]) * mprec[i+1]))

@torch.no_grad()
def evaluate_map(model, loader, class_names, device="cuda", iou_thr=0.5,
                 max_batches=None):
    model.to(device).eval()
    K = len(class_names)
    gt_count = [0]*K
    preds = [[] for _ in range(K)]  # per-class list of (score, tp)

    n_seen = 0
    for imgs, targets in loader:
        if max_batches is not None and n_seen >= max_batches:
            break
        n_seen += 1
        imgs = [i.to(device) for i in imgs]
        outs = model(imgs)

        for out, tgt in zip(outs, targets):
            g_boxes = tgt["boxes"].cpu()
            g_labels = tgt["labels"].cpu().int()
            for c in g_labels.tolist():
                gt_count[int(c)] += 1

            p_boxes = out["boxes"].cpu()
            p_scores = out["scores"].cpu()
            p_labels = out["labels"].cpu().int()

            # per-class greedy matching
            for c in p_labels.unique().tolist():
                c = int(c)
                p_idx = (p_labels == c).nonzero(as_tuple=True)[0]
                if p_idx.numel() == 0:
                    continue
                order = torch.argsort(p_scores[p_idx], descending=True)
                p_idx = p_idx[order]
                pb, ps = p_boxes[p_idx], p_scores[p_idx]

                g_idx = (g_labels == c).nonzero(as_tuple=True)[0]
                gb = g_boxes[g_idx]
                matched = set()
                for j in range(pb.size(0)):
                    tp = 0
                    if gb.numel()>0:
                        lt = torch.max(pb[j:j+1,:2], gb[:,:2])
                        rb = torch.min(pb[j:j+1,2:], gb[:,2:])
                        wh = (rb - lt).clamp(min=0)
                        inter = (wh[:, 0]*wh[:, 1])
                        area_p = (pb[j, 2]-pb[j, 0]).clamp(min=0)*(pb[j,3]-pb[j,1]).clamp(min=0)
                        area_g = (gb[:, 2]-gb[:, 0]).clamp(min=0)*(gb[:,3]-gb[:,1]).clamp(min=0)
                        ious = inter / (area_p + area_g - inter + 1e-6)
                        mi, midx = torch.max(ious, dim=0)
                        if mi.item() >= iou_thr and int(midx) not in matched:
                            tp = 1; matched.add(int(midx))
                    preds[c].append((float(ps[j]), tp))

    # AP per class
    ap = []
    per_class = {}
    for c in range(K):
        pcs = preds[c]
        if len(pcs) == 0 or gt_count[c] == 0:
            per_class[str(class_names[c])] = 0.0
            ap.append(0.0); continue
        scores, tps = zip(*pcs)
        order = np.argsort(-np.array(scores))
        tps = np.array(tps)[order]
        fps = 1 - tps
        tp_c = np.cumsum(tps)
        fp_c = np.cumsum(fps)
        rec = tp_c / max(gt_count[c], 1)
        prec = tp_c / np.maximum(tp_c + fp_c, 1)
        A = _voc_ap(rec, prec)
        per_class[str(class_names[c])] = A
        ap.append(A)
    return {"mAP@0.5": float(np.mean(ap)), "per_class_AP": per_class,
            "gt_counts": gt_count}

@torch.no_grad()
def evaluate_pr(model, loader, device="cuda", iou_thr=0.5, score_thresh=None, class_agnostic=False, max_batches=None):
    model.to(device).eval()
    outs_all, targs_all = [], []
    seen = 0
    for imgs, targets in loader:
        if max_batches is not None and seen >= max_batches: break
        seen += 1
        imgs = [i.to(device) for i in imgs]
        outs = model(imgs)

        for o in outs:
            keep = (o["scores"] >= float(score_thresh)) if score_thresh is not None else torch.ones_like(o["scores"], dtype=torch.bool)
            boxes = o["boxes"][keep].cpu()
            labels = o["labels"][keep].cpu()
            if class_agnostic:
                labels = torch.zeros_like(labels)
            outs_all.append({"boxes": boxes, "labels": labels})

        for t in targets:
            boxes = t["boxes"].cpu()
            labels = t["labels"].cpu()
            if class_agnostic:
                labels = torch.zeros_like(labels)
            targs_all.append({"boxes": boxes, "labels": labels})

    return precision_recall(outs_all, targs_all, iou_thr=iou_thr)

def sweep_score_thresh(model, loader, device="cuda", iou=0.5,
                       thresholds=np.linspace(0.01, 0.90, 30),
                       max_batches=None):
    rows = []
    model.eval()
    for th in thresholds:
        r = evaluate_pr(model=model, loader=loader, device=device,
                        iou_thr=iou, score_thresh=float(th), max_batches=max_batches)
        p, q = r["precision"], r["recall"]
        f1 = (2 * p * q) / (p + q + 1e-9)
        rows.append({"th": float(th), "precision": p, "recall": q, "f1": f1})
    best = max(rows, key=lambda z: z["f1"])

    return rows, best

@torch.no_grad()
def build_gt_index(loader):
    gt = {}
    for _, targets in loader:
        for t in targets:
            name = Path(str(t["image_path"])).name
            gt[name] = {
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu().to(torch.int64)
            }
    return gt

def match_predictions(preds_df: pd.DataFrame, gt_index, iou_thr=0.5):
    """Return (preds_eval_df, gt_counts). Adds columns: tp (bool), match_iou (float)."""
    rows = []
    gt_counts = {}
    for name, g in gt_index.items():
        for c in g["labels"].tolist():
            gt_counts[int(c)] = gt_counts.get(int(c), 0) + 1

    for name, df in preds_df.groupby("image_path"):
        g = gt_index.get(name)
        if g is None:
            for _, r in df.iterrows():
                rows.append({**r, "tp": False, "match_iou": float("nan")})
            continue
        gb, gl = g["boxes"], g["labels"]
        used = set()
        df = df.sort_values("score", ascending=False)
        for _, r in df.iterrows():
            pb = torch.tensor([[r.xmin, r.ymin, r.xmax, r.ymax]], dtype=torch.float32)
            pl = int(r.label_id)
            if gb.numel() == 0:
                rows.append({**r, "tp": False, "match_iou": float("nan")}); continue
            ious = box_iou(pb, gb)[0].numpy()
            j = int(np.argmax(ious))
            ok = (ious[j] >= iou_thr) and (j not in used) and (int(gl[j]) == pl)
            if ok:
                used.add(j)
                rows.append({**r, "tp": True, "match_iou": float(ious[j]),
                             "gt_idx": j, "gt_label_id": int(gl[j])})
            else:
                rows.append({**r, "tp": False, "match_iou": float(ious[j]),
                             "gt_idx": None, "gt_label_id": None})

    return pd.DataFrame(rows), gt_counts

def pr_curve_for_class(preds_eval_df: pd.DataFrame, gt_counts: dict, class_id: int):
    dfc = preds_eval_df[preds_eval_df["label_id"] == int(class_id)].sort_values("score", ascending=False)
    if dfc.empty:
        return np.array([0.0]), np.array([0.0]), 0.0
    tp = dfc["tp"].astype(int).to_numpy()
    fp = 1 - tp
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / max(gt_counts.get(int(class_id), 0), 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    ap = _voc_ap(rec, prec)
    return rec, prec, ap

def detection_only_counts(preds, gt, iou=0.5):
    TPd = FPd = FNd = 0
    for img, pdf in preds.groupby("image_path"):
        g = gt.get(img)
        if g is None:
            FPd += len(pdf); continue
        p = torch.tensor(pdf[["xmin","ymin","xmax","ymax"]].values, dtype=torch.float32)
        gb = g["boxes"]
        if len(p)==0: FNd += len(gb); continue
        if len(gb)==0: FPd += len(p); continue
        I = box_iou(p, gb)
        used_p, used_g = set(), set()
        while True:
            k = torch.argmax(I).item()
            i, j = divmod(k, I.shape[1])
            if I[i, j] < iou: break
            if i in used_p or j in used_g: I[i, j] = -1; continue
            used_p.add(i); used_g.add(j)
            I[i, :] = -1; I[:, j] = -1
        TPd += len(used_p)
        FPd += len(p) - len(used_p)
        FNd += len(gb) - len(used_g)
    return {"TPd": TPd, "FPd": FPd, "FNd": FNd}

def classification_on_matched(preds, gt, class_names, iou=0.5):
    pairs = []  # (gt_label, pred_label)
    for img, pdf in preds.groupby("image_path"):
        g = gt.get(img)
        if g is None: continue
        p = torch.tensor(pdf[["xmin","ymin","xmax","ymax"]].values, dtype=torch.float32)
        pl = torch.tensor(pdf["label_id"].values, dtype=torch.int64)
        gb, gl = g["boxes"], g["labels"].to(torch.int64)
        if len(p)==0 or len(gb)==0: continue
        I = box_iou(p, gb)
        used_p, used_g = set(), set()
        while True:
            k = torch.argmax(I).item()
            i, j = divmod(k, I.shape[1])
            if I[i, j] < iou: break
            if i in used_p or j in used_g: I[i, j] = -1; continue
            pairs.append((int(gl[j].item()), int(pl[i].item())))
            used_p.add(i); used_g.add(j)
            I[i, :] = -1; I[:, j] = -1

    if not pairs:
        return None, None, None

    dfp = pd.DataFrame(pairs, columns=["gt","pred"])
    cm = pd.crosstab(dfp["gt"], dfp["pred"], rownames=["GT"], colnames=["Pred"], dropna=False)
    # label with names
    cm.index = [class_names[i] for i in cm.index]
    cm.columns = [class_names[i] for i in cm.columns]
    cls_acc = np.trace(cm.values) / cm.values.sum()
    per_class_recall = (cm.values.diagonal() / cm.sum(axis=1).to_numpy()).tolist()
    return cm, cls_acc, per_class_recall














