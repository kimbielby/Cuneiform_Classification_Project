import torch
import pandas as pd
from pathlib import Path

@torch.no_grad()
def run_inference(model, loader, class_names, out_csv="predictions.csv", device="cuda", score_thresh=None):
    model.to(device).eval()

    if score_thresh is not None and hasattr(model, "score_thresh"):
        model.score_thresh = float(score_thresh)

    rows = []
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        outs = model(imgs)
        for o, t in zip(outs, targets):
            img_id = Path(str(t["image_path"])).name
            boxes = o["boxes"].cpu().numpy()
            scores = o["scores"].cpu().numpy()
            labels = o["labels"].cpu().numpy().astype(int)
            if score_thresh is not None:
                keep = scores >= float(score_thresh)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            for box, score, lid in zip(boxes, scores, labels):
                rows.append({
                    "image_path": img_id,
                    "label_id": int(lid),
                    "label_name": class_names[int(lid)],
                    "xmin": float(box[0]), "ymin": float(box[1]),
                    "xmax": float(box[2]), "ymax": float(box[3]),
                    "score": float(score),
                })
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df




