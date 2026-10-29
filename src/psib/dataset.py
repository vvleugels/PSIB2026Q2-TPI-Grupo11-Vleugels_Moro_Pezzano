"""Dataset inventory and reproducible reference-mask-based sampling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .data_io import list_cases, load_case


def build_inventory(data_dir: str | Path, candidate_pool_size: int = 15, seed: int = 2026) -> pd.DataFrame:
    """Inventory a seeded subsample of labeled cases, loading one volume at a time.

    The candidate pool is sampled from ``list_cases(data_dir)`` and is a
    reproducible subsample of the available labeled cases (131 in the supplied
    LiTS/Task03 training split). Unlabeled test images are never inventoried.
    """
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive")
    available = list_cases(data_dir)
    if not available:
        raise ValueError(f"No paired CT volumes and masks found in {data_dir}")
    rng = np.random.default_rng(seed)
    pool = rng.choice(np.asarray(available, dtype=int), size=min(candidate_pool_size, len(available)), replace=False)
    rows: list[dict[str, object]] = []
    for pool_rank, case_id in enumerate(pool, start=1):
        ct, seg, spacing = load_case(case_id, data_dir)
        tumor = seg == 2
        tumor_voxels = int(np.count_nonzero(tumor))
        sx, sy, sz = spacing
        rows.append({
            "case_id": int(case_id), "candidate_pool_rank": pool_rank, "candidate_pool_seed": seed,
            "shape": "x".join(map(str, ct.shape)),
            "shape_x": ct.shape[0], "shape_y": ct.shape[1], "shape_z": ct.shape[2],
            "spacing_x_mm": sx, "spacing_y_mm": sy, "spacing_z_mm": sz,
            "n_slices": ct.shape[2], "hu_min": float(np.min(ct)),
            "hu_max": float(np.max(ct)), "hu_mean": float(np.mean(ct)),
            "liver_voxels": int(np.count_nonzero(seg == 1)),
            "tumor_voxels": tumor_voxels,
            "tumor_volume_cm3": tumor_voxels * sx * sy * sz / 1000.0,
            "tumor_slices": int(np.count_nonzero(np.any(tumor, axis=(0, 1)))),
            "tumor_lesions_3d": int(ndimage.label(tumor)[1]),
        })
        del ct, seg
    return pd.DataFrame(rows).sort_values("candidate_pool_rank").reset_index(drop=True)


def tumor_burden_terciles(inventory: pd.DataFrame) -> pd.Series:
    """Assign tumor-volume terciles or report when the pool cannot support them."""
    volume = inventory["tumor_volume_cm3"].astype(float)
    bins = pd.qcut(volume, 3, labels=False, duplicates="drop")
    present = sorted(pd.Series(bins).dropna().unique().astype(int).tolist())
    if present != [0, 1, 2]:
        raise ValueError(
            "Candidate pool does not cover all three tumor-volume terciles "
            f"(small/medium/large); observed {len(present)} distinct size group(s). "
            "Increase candidate_pool_size or review tumor-volume ties before selecting cases."
        )
    return pd.Series(np.asarray(bins, dtype=int), index=inventory.index).map({0: "small", 1: "medium", 2: "large"})


def select_cases(inventory: pd.DataFrame, n_cases: int, seed: int) -> list[int]:
    """Select reproducible, tercile-balanced cases, favoring multi-lesion cases.

    Terciles and selected slices use the supplied reference masks. The input
    inventory is a seeded candidate-pool subsample of the 131 labeled LiTS
    cases, not an unbiased cohort or a learned prediction. Missing terciles
    and the absence of multi-lesion cases are reported as errors.
    """
    if n_cases < 1:
        raise ValueError("n_cases must be positive")
    required = {"case_id", "tumor_volume_cm3", "tumor_lesions_3d"}
    if not required.issubset(inventory.columns):
        raise ValueError(f"Inventory must include {sorted(required)}")
    frame = inventory.copy()
    frame["burden_tercile"] = tumor_burden_terciles(frame)
    if n_cases > len(frame):
        raise ValueError(f"Requested {n_cases} final cases from a pool of only {len(frame)}")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    bins = ["small", "medium", "large"]
    base, remainder = divmod(min(n_cases, len(frame)), 3)
    quotas = {name: base + (i < remainder) for i, name in enumerate(bins)}
    for name in bins:
        group = frame[frame.burden_tercile == name]
        ids = group.case_id.to_numpy(dtype=int, copy=True)
        rng.shuffle(ids)
        selected.extend(ids[:quotas[name]].tolist())
    # Fill any short tercile quota (possible with tiny/tied datasets).
    remaining = frame[~frame.case_id.isin(selected)].copy()
    if len(selected) < min(n_cases, len(frame)):
        ids = remaining.case_id.to_numpy(dtype=int, copy=True)
        rng.shuffle(ids)
        selected.extend(ids[:min(n_cases, len(frame)) - len(selected)].tolist())
    # Include two multi-lesion cases when available and the requested sample allows it.
    target_multi = min(2, n_cases, int((frame.tumor_lesions_3d > 1).sum()))
    if target_multi == 0:
        raise ValueError("Candidate pool contains no multi-lesion case; cannot meet the requested selection criteria")
    selected_multi = int(frame[frame.case_id.isin(selected)].tumor_lesions_3d.gt(1).sum())
    candidates = frame[frame.tumor_lesions_3d > 1].copy()
    candidates = candidates[~candidates.case_id.isin(selected)]
    if len(candidates):
        candidates = candidates.sample(frac=1, random_state=seed)
        for candidate in candidates.itertuples():
            if selected_multi >= target_multi:
                break
            # Swap within the same tercile where possible, preserving balance.
            removable = frame[frame.case_id.isin(selected) & frame.tumor_lesions_3d.le(1)]
            same_bin = removable[removable.burden_tercile == candidate.burden_tercile]
            pool = same_bin if not same_bin.empty else removable
            if pool.empty:
                break
            drop_id = int(pool.iloc[-1].case_id)
            selected.remove(drop_id)
            selected.append(int(candidate.case_id))
            selected_multi += 1
    return [int(v) for v in frame[frame.case_id.isin(selected)].case_id]


def select_slices(seg: np.ndarray, n_slices: int, minimum_gap: int = 5) -> list[int]:
    """Choose tumor-rich ``(X,Y,Z)`` slices, keeping indices at least gap apart."""
    if seg.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation, got shape {seg.shape}")
    if n_slices < 1 or minimum_gap < 0:
        raise ValueError("n_slices must be positive and minimum_gap nonnegative")
    areas = np.count_nonzero(seg == 2, axis=(0, 1))
    # Zero-tumor slices remain valid tied candidates for selected low-burden
    # cases; prefer liver-containing slices to make their preprocessing usable.
    liver_areas = np.count_nonzero(seg > 0, axis=(0, 1))
    ranked = sorted(range(len(areas)), key=lambda k: (-int(areas[k]), -int(liver_areas[k]), k))
    chosen: list[int] = []
    for k in ranked:
        if all(abs(k - other) >= minimum_gap for other in chosen):
            chosen.append(k)
            if len(chosen) == n_slices:
                break
    return sorted(chosen)
