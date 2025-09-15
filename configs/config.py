from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml

@dataclass
class PathsConfig:
    images_root: Path
    csv_annot: Path

@dataclass
class DatasetConfig:
    class_allowlist_path: Optional[Path] = None
    class_allowlist: Optional[List[int]] = None
    include_other_class: bool = False
    other_class_name: str = "other"

@dataclass
class ModelConfig:
    backbone: str
    use_imagenet_weights: bool
    anchor_sizes: List[List[int]]
    aspect_ratios: List[List[float]]
    score_thresh: float
    nms_thresh: float
    detections_per_img: int

@dataclass
class TrainConfig:
    batch_size: int
    accum_steps: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    freeze_backbone_epochs: int
    checkpoint_dir: Path
    save_every: int
    keep_last_k: int
    save_best: bool
    best_metric: str

@dataclass
class EvalConfig:
    do_map: bool
    map_iou: float
    map_every: int
    map_max_batches: int | None
    full_every: int
    full_on_last: bool

@dataclass
class Config:
    paths: PathsConfig
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig

def _as_int_list(xs) -> List[int]:
    out = []
    for x in xs:
        try:
            out.append(int(round(float(x))))
        except Exception:
            pass
    return sorted(list(dict.fromkeys(out)))

def _load_allowlist_file(p: Path) -> List[int]:
    d = yaml.safe_load(p.read_text())
    xs = d.get("class_allowlist") or d.get("class_whitelist")
    if xs is None:
        raise ValueError(f"No class_allowlist/whitelist in {p}")
    return _as_int_list(xs)

def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    base = config_path.parent
    d = yaml.safe_load(config_path.read_text())

    # paths
    p = d["paths"]
    paths = PathsConfig(
        images_root=(base / p["images_root"]).resolve(),
        csv_annot=(base / p["csv_annot"]).resolve(),
    )

    # dataset
    ds = d.get("dataset", {})
    allow_inline = ds.get("class_allowlist")  # optional inline list
    allow_path = ds.get("class_allowlist_path")  # optional external YAML
    allow: Optional[List[int]] = None
    allow_file: Optional[Path] = None

    if allow_inline is not None:
        allow = _as_int_list(allow_inline)
    elif allow_path is not None:
        allow_file = (base / allow_path).resolve()
        allow = _load_allowlist_file(allow_file)

    dataset = DatasetConfig(
        class_allowlist_path=allow_file,
        class_allowlist=allow,
        include_other_class=bool(ds.get("include_other_class", False)),
        other_class_name=str(ds.get("other_class_name", "other"))
    )

    # model
    m = d["model"]
    model = ModelConfig(
        backbone=m["backbone"],
        use_imagenet_weights=bool(m["use_imagenet_weights"]),
        anchor_sizes=[_as_int_list(g) for g in m["anchor_sizes"]],
        aspect_ratios=[[float(a) for a in g] for g in m["aspect_ratios"]],
        score_thresh=float(m["score_thresh"]),
        nms_thresh=float(m["nms_thresh"]),
        detections_per_img=int(m["detections_per_img"]),
    )

    # train
    t = d["train"]
    train = TrainConfig(
        batch_size=int(t["batch_size"]),
        accum_steps=int(t["accum_steps"]),
        epochs=int(t["epochs"]),
        lr=float(t["lr"]),
        weight_decay=float(t["weight_decay"]),
        num_workers=int(t["num_workers"]),
        freeze_backbone_epochs=int(t["freeze_backbone_epochs"]),
        checkpoint_dir=Path(t["ckpt_dir"]),
        save_every=int(t["save_every"]),
        keep_last_k=int(t["keep_last_k"]),
        save_best=bool(t["save_best"]),
        best_metric=str(t["best_metric"]),
    )

    # eval
    evaluate = EvalConfig(**d.get("eval", {
        "do_map": False,
        "map_iou": 0.5,
        "map_every": 1,
        "map_max_batches": None,
        "full_every": 0,
        "full_on_last": True
    }))

    return Config(paths=paths, dataset=dataset, model=model, train=train,
                  eval=evaluate)










