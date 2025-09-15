import cv2
import numpy as np
import torch
from pathlib import Path

class SignDataset(torch.utils.data.Dataset):
    def __init__(self, data, images_root, class_names=None,
                 transforms=None):
        self.df = data
        self.images_root = Path(images_root)
        self.transforms = transforms

        if "_mapped_label" not in self.df:
            src = "mzl_label"

            classes = sorted(
                self.df[src].unique().tolist() if class_names is None
                else list(class_names)
            )

            idmap = {c: i for i, c in enumerate(classes)}
            self.df["_mapped_label"] = self.df[src].map(idmap).astype("int64")
            self.class_names = [str(c) for c in classes]
        else:
            self.class_names = [
                str(c) for c in (class_names if class_names is not None else [])
            ]

        self.groups = list(self.df.groupby("image_path"))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        img_name, g = self.groups[idx]
        img_path = self.images_root / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0

        g_valid = g.dropna(subset=["xmin", "ymin", "xmax", "ymax"]).copy()
        if not g_valid.empty:
            w = (g_valid["xmax"] - g_valid["xmin"]).astype("float32")
            h = (g_valid["ymax"] - g_valid["ymin"]).astype("float32")
            g_valid = g_valid[(w > 0) & (h > 0)]

        boxes = g_valid[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32") \
            if not g_valid.empty else np.zeros((0, 4), dtype="float32")
        labels = g_valid["_mapped_label"].values.astype("int64") if not g_valid.empty \
            else np.zeros((0,), dtype="int64")

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_path": Path(img_name).name,
            "image_id": torch.tensor(idx, dtype=torch.int64),
        }

        img = torch.from_numpy(img).permute(2, 0, 1)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

def get_dataloader(data, images_root, class_names=None, batch_size=4,
                   shuffle=True, num_workers=4, transforms=None,
                   collate_fn=None):

    if collate_fn is None:
        from Utils.collate import collate as collate_fn

    dataset = SignDataset(data=data, images_root=images_root, class_names=class_names,
                          transforms=transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return loader, dataset.class_names

