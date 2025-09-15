# one-image eval + overlay
from pathlib import Path

import cv2, torch, numpy as np, pandas as pd
from Utils.metrics import build_gt_index, match_predictions, precision_recall
from dataloaders.dataloader import SignDataset  # path per your project

def collate(batch):
    ims, tgts = list(zip(*batch))
    return list(ims), list(tgts)

def load_model_from_ckpt(path, model, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()

@torch.no_grad()
def eval_one_image(model, image_path, gt_df_for_image, class_names, images_root,
                   device, score_thresh=0.6, iou_thr=0.5, out_path="overlay.jpg"):
    # 1) Mini dataset/loader (reuses your preprocessing exactly)
    base = Path(image_path).name
    df1 = gt_df_for_image.copy()
    df1["image_path"] = base  # ensure consistent name

    # map mzl_label -> model indices (handles 'other' if present)
    name2idx = {str(n): i for i, n in enumerate(class_names)}

    def map_lbl(x):
        s = str(int(x)) if pd.notna(x) else None
        if s in name2idx:
            return name2idx[s]
        return name2idx["other"] if "other" in name2idx else np.nan

    df1["_mapped_label"] = df1["mzl_label"].apply(map_lbl).astype("int64")

    ds = SignDataset(
        data=df1,
        images_root=images_root,
        class_names=class_names,
        transforms=None
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    # 2) Inference
    model.eval().to(device)
    imgs, tgts = next(iter(dl))
    preds = model([imgs[0].to(device)])[0]
    # filter by score_thresh
    keep = preds["scores"] >= score_thresh
    boxes = preds["boxes"][keep].cpu().numpy()
    labels = preds["labels"][keep].cpu().numpy()
    scores = preds["scores"][keep].cpu().numpy()

    # 3) Build tiny preds DataFrame to reuse your metrics code
    pred_rows = []
    for b, c, s in zip(boxes, labels, scores):
        x1,y1,x2,y2 = b.tolist()
        pred_rows.append({"image_path": base,
                          "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                          "label_id": int(c), "label_name": class_names[int(c)],
                          "score": float(s)})
    preds_df = pd.DataFrame(pred_rows)

    # 4) GT index + match, then per-image precision/recall
    gt = build_gt_index(dl)  # loader already has GT for this image
    preds_eval, _ = match_predictions(preds_df, gt, iou_thr=iou_thr)
    pr = precision_recall(
        outputs=[{"boxes": torch.tensor(boxes), "labels": torch.tensor(labels)}],
        targets=[{"boxes": tgts[0]["boxes"], "labels": tgts[0]["labels"]}],
        iou_thr=iou_thr,
    )
    metrics = {
        "precision@{:.2f}".format(iou_thr): pr["precision"],
        "recall@{:.2f}".format(iou_thr): pr["recall"],
        "tp": pr["tp"], "fp": pr["fp"], "fn": pr["fn"],
    }

    # 5) Overlay: GT = yellow, TP = green, FP = red
    full_img_path = Path(images_root) / base
    im_bgr = cv2.imread(str(full_img_path), cv2.IMREAD_COLOR)
    for b in tgts[0]["boxes"].cpu().numpy():
        x1, y1, x2, y2 = map(int, b);
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for _, r in preds_eval.iterrows():
        x1, y1, x2, y2 = int(r.xmin), int(r.ymin), int(r.xmax), int(r.ymax)
        color = (0, 200, 0) if bool(r.tp) else (255, 0, 0)
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, 2)

    y = 10
    legend = [("GT", (0, 255, 255)), ("TP", (0, 200, 0)), ("FP", (255, 0, 0))]
    for txt, col in legend:
        cv2.rectangle(im_bgr, (10, y), (40, y + 20), col, -1)
        cv2.putText(
            im_bgr,
            txt,
            (50, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255), 1,
            cv2.LINE_AA
        )
        y += 28

    out_path = Path(out_path);
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im_bgr)
    return metrics, preds_eval, str(out_path), im_bgr
