import os
import numpy as np
import pandas as pd
import cv2
import albumentations as A

IMAGE_SIZE = 512

def _aug_with_boxes(to_gray_p=0.0, strength="strong"):

    if strength == "strong":
        rrc = A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE),
                                  scale=(0.75, 1.0), ratio=(0.85, 1.15), p=1.0)
        geo = A.Affine(scale=(0.9, 1.1), rotate=(-12, 12),
                       translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                       interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                       fill=0, fill_mask=0, p=0.8)
        photo = [
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(std_range=(5 / 255.0, 20 / 255.0), p=0.2),
        ]
        flip = A.HorizontalFlip(p=0.5)
        sharp = A.UnsharpMask(blur_limit=3, alpha=(0.1, 0.3), threshold=0, p=0.3)
        min_vis = 0.20
    else:  # "medium" / fallback
        rrc = A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE),
                                  scale=(0.9, 1.0), ratio=(0.95, 1.05), p=1.0)
        geo, photo, flip = A.NoOp(), [], A.HorizontalFlip(p=0.5)
        sharp, min_vis = A.UnsharpMask(blur_limit=3, alpha=(0.1, 0.3), threshold=0, p=0.2), 0.30

    return A.Compose(
        [rrc, geo, flip, A.ToGray(p=to_gray_p), *photo, sharp],
        bbox_params=A.BboxParams(format="pascal_voc",
                                 label_fields=["labels", "mapped_labels"],
                                 min_area=16, min_visibility=min_vis)
    )

def augment_train_df(train_df: pd.DataFrame, images_root):
    aug = _aug_with_boxes(to_gray_p=1.0, strength="strong")
    aug_no_gray = _aug_with_boxes(to_gray_p=0.0, strength="strong")
    out_rows = []
    have_mapped = "_mapped_label" in train_df.columns
    nan = np.nan

    # Read each image once, write N times
    for i, (img_name, g) in enumerate(train_df.groupby("image_path", sort=False)):
        g_valid = g.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
        if g_valid.empty:
            continue

        path = images_root / img_name
        if "VAT" in img_name:
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            im = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if im is None:
            continue

        boxes = g_valid[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
        labels = g_valid["mzl_label"].tolist()
        mapped = g_valid["_mapped_label"].tolist() if have_mapped else labels

        stem, ext = os.path.splitext(img_name)

        top4 = {0, 1, 2, 3}
        has_top4 = any(pd.notna(m) and int(m) in top4 for m in mapped)
        times_local = 4 if has_top4 else 1
        # per-image loop
        for k in range(times_local):
            new_name = f"{stem}_aug{k}{ext}"
            out_path = images_root / new_name

            if "VAT" in img_name or i > 50:
                data = aug_no_gray(image=im, bboxes=boxes, labels=labels, mapped_labels=(mapped or labels))
            else:
                data = aug(image=im, bboxes=boxes, labels=labels, mapped_labels=(mapped or labels))

            aug_im, aug_boxes, aug_labels, aug_mapped = (
                data["image"], data["bboxes"], data["labels"], data["mapped_labels"]
            )

            if not aug_boxes:
                continue

            cv2.imwrite(str(out_path), aug_im)

            for (xmin, ymin, xmax, ymax), lbl, m_lbl in zip(
                    aug_boxes, aug_labels, (aug_mapped or aug_labels)
            ):
                row = {
                    "image_path": new_name,
                    "xmin": float(xmin), "ymin": float(ymin),
                    "xmax": float(xmax), "ymax": float(ymax),
                    "mzl_label": lbl,
                }

                if have_mapped:
                    row["_mapped_label"] = int(m_lbl)
                out_rows.append(row)

    if not out_rows:
        return train_df

    aug_df = pd.DataFrame(out_rows)
    # align columns to train_df
    for c in train_df.columns:
        if c not in aug_df.columns:
            aug_df[c] = nan
    aug_df = aug_df[train_df.columns]

    return pd.concat([train_df, aug_df], ignore_index=True)
