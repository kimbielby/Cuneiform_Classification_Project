import pandas as pd
import yaml
import numpy as np
from pathlib import Path

def _resolve_allowlist(config):
    dataset = config.dataset

    if dataset.class_allowlist:
        return sorted(int(x) for x in dataset.class_allowlist)

    p = dataset.class_allowlist_path
    if p:
        d = yaml.safe_load(Path(p).read_text())
        xs = d.get("class_allowlist")
        return sorted(int(x) for x in xs)

    return None

def prepare_and_split(og_df, config):
    df = og_df.copy()

    # Normalize labels to ints (adjust if yours are strings)
    df["mzl_norm"] = pd.to_numeric(df.mzl_label, errors="coerce")

    # Apply allowlist from config (loaded first)
    allow = _resolve_allowlist(config)
    include_other = bool(config.dataset.include_other_class)
    other_name = config.dataset.other_class_name

    df["_mapped_label"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if allow is None:
        valid = df["mzl_norm"].notna()
        class_names = sorted(df.loc[valid, "mzl_norm"].astype(int).unique().tolist())
        idmap = {int(c): i for i, c in enumerate(class_names)}
        df.loc[valid, "_mapped_label"] = df.loc[valid, "mzl_norm"].astype(int).map(idmap)
    else:
        allow = [int(x) for x in allow]
        allow_set = set(allow)
        class_names = allow.copy()
        valid = df["mzl_norm"].notna()
        if include_other:
            other_id = len(class_names)
            def _map(v):
                v = int(v)
                return class_names.index(v) if v in allow_set else other_id
            df.loc[valid, "_mapped_label"] = df.loc[valid, "mzl_norm"].apply(_map).astype("Int64")
            class_names = class_names + [other_name]
        else:
            idmap = {int(c): i for i, c in enumerate(class_names)}
            keep = valid & df["mzl_norm"].astype(int).isin(allow_set)
            df.loc[keep, "_mapped_label"] = df.loc[keep, "mzl_norm"].astype("Int64").map(idmap)

    # SPLIT BY IMAGE
    rng = np.random.default_rng()
    # Compute per-image eligibility
    grp = df.groupby("image_path", sort=False)

    def has_box(g):
        return g[["xmin", "ymin", "xmax", "ymax"]].notna().all(axis=1).any()

    if allow is None:
        def has_allow(g):
            return g["mzl_norm"].notna().any()
    else:
        allow_set = set(int(x) for x in allow)
        def has_allow(g):
            return g["mzl_norm"].isin(allow_set).any()

    eligible = [k for k, g in grp if has_box(g) and has_allow(g)]
    others = [k for k, g in grp if k not in eligible]

    eligible = rng.permutation(np.array(eligible, dtype=object))
    others = rng.permutation(np.array(others, dtype=object))

    # Target counts
    n_total = len(eligible) + len(others)
    t_frac, v_frac, s_frac = 0.60, 0.20, 0.20
    n_test_target = int(round(s_frac * n_total))

    # Pick test from eligible only
    n_test = min(n_test_target, len(eligible))
    test_images = eligible[:n_test]

    # Split the rest into train/val
    remain = np.concatenate([eligible[n_test:], others])
    n_rem = len(remain)
    # Keep train:val ratio on the remaining pool
    tv = t_frac + v_frac
    n_train = int(round((t_frac / tv) * n_rem))
    train_images = remain[:n_train]
    val_images = remain[n_train:]

    # Build dfs
    train_df = df[df.image_path.isin(train_images)].sample(frac=1).reset_index(drop=True)
    val_df = df[df.image_path.isin(val_images)].sample(frac=1).reset_index(drop=True)
    test_df = df[df.image_path.isin(test_images)].sample(frac=1).reset_index(drop=True)

    return train_df, val_df, test_df, class_names












