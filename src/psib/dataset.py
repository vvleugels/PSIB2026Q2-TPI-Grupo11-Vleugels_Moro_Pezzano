"""Dataset inventory and reproducible reference-mask-based sampling."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage

from .data_io import list_cases, load_case


def build_inventory(data_dir: str | Path, candidate_pool_size: int = 15, seed: int = 2026,
                    density_tolerance_hu: float = 5.0) -> pd.DataFrame:
    """Inventory a seeded subsample of labeled cases, loading one volume at a time.

    The candidate pool is sampled from ``list_cases(data_dir)`` and is a
    reproducible subsample of the available labeled cases. The original
    Task03 training split has 131 pairs; the bundled shareable subset has 10.
    Unlabeled test images are never inventoried.
    """
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive")
    if not np.isfinite(density_tolerance_hu) or density_tolerance_hu < 0:
        raise ValueError("density_tolerance_hu must be finite and nonnegative")
    available = list_cases(data_dir)
    if not available:
        raise ValueError(f"No paired CT volumes and masks found in {data_dir}")
    rng = np.random.default_rng(seed)
    pool = rng.choice(np.asarray(available, dtype=int), size=min(candidate_pool_size, len(available)), replace=False)
    rows: list[dict[str, object]] = []
    for pool_rank, case_id in enumerate(pool, start=1):
        ct, seg, spacing = load_case(case_id, data_dir)
        tumor = seg == 2
        healthy_liver = seg == 1
        liver = seg > 0
        tumor_voxels = int(np.count_nonzero(tumor))
        liver_voxels = int(np.count_nonzero(healthy_liver))
        liver_total_voxels = int(np.count_nonzero(liver))
        sx, sy, sz = spacing
        healthy_median = float(np.median(ct[healthy_liver])) if liver_voxels else float("nan")
        tumor_median = float(np.median(ct[tumor])) if tumor_voxels else float("nan")
        delta_hu = tumor_median - healthy_median
        abs_delta = abs(delta_hu)
        contrast_class = ("<20" if abs_delta < 20 else "20-40" if abs_delta < 40
                          else "40-60" if abs_delta < 60 else ">=60") if np.isfinite(abs_delta) else "unknown"
        if not np.isfinite(delta_hu):
            density = "unknown"
        elif abs(delta_hu) <= density_tolerance_hu:
            density = "isodense"
        else:
            density = "hypodense" if delta_hu < 0 else "hyperdense"
        labels_6, n_components_6 = ndimage.label(tumor, structure=ndimage.generate_binary_structure(3, 1))
        if n_components_6:
            component_voxels = np.bincount(labels_6.ravel())[1:]
            component_diameters = (6.0 * component_voxels * sx * sy * sz / np.pi) ** (1 / 3)
            component_sizes = (int(np.count_nonzero(component_diameters < 10)),
                               int(np.count_nonzero((component_diameters >= 10) & (component_diameters <= 20))),
                               int(np.count_nonzero(component_diameters > 20)))
        else:
            component_sizes = (0, 0, 0)
        rows.append({
            "case_id": int(case_id), "candidate_pool_rank": pool_rank, "candidate_pool_seed": seed,
            "shape": "x".join(map(str, ct.shape)),
            "shape_x": ct.shape[0], "shape_y": ct.shape[1], "shape_z": ct.shape[2],
            "spacing_x_mm": sx, "spacing_y_mm": sy, "spacing_z_mm": sz,
            "n_slices": ct.shape[2], "hu_min": float(np.min(ct)),
            "hu_max": float(np.max(ct)), "hu_mean": float(np.mean(ct)),
            "liver_voxels": liver_voxels,
            "liver_total_voxels": liver_total_voxels,
            "liver_volume_cm3": liver_total_voxels * sx * sy * sz / 1000.0,
            "tumor_burden": tumor_voxels / liver_total_voxels if liver_total_voxels else float("nan"),
            "tumor_voxels": tumor_voxels,
            "tumor_volume_cm3": tumor_voxels * sx * sy * sz / 1000.0,
            "tumor_slices": int(np.count_nonzero(np.any(tumor, axis=(0, 1)))),
            "tumor_lesions_3d": int(ndimage.label(tumor)[1]),
            "median_hu_healthy_liver": healthy_median,
            "median_hu_tumor": tumor_median,
            "delta_hu": delta_hu,
            "contrast_class": contrast_class,
            "density": density,
            "density_tolerance_hu": float(density_tolerance_hu),
            "lesions_lt10mm": component_sizes[0],
            "lesions_10_20mm": component_sizes[1],
            "lesions_gt20mm": component_sizes[2],
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


def dataset_context_table(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return the compact case-level report table from a context inventory."""
    columns = ["case_id", "shape", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
               "tumor_volume_cm3", "tumor_lesions_3d", "tumor_burden", "delta_hu", "density",
               "lesions_lt10mm", "lesions_10_20mm", "lesions_gt20mm"]
    missing = sorted(set(columns) - set(inventory.columns))
    if missing:
        raise ValueError(f"Inventory lacks dataset-context columns: {missing}")
    return inventory.loc[:, columns].rename(columns={"tumor_lesions_3d": "lesions"}).copy()


def select_cases(inventory: pd.DataFrame, n_cases: int, seed: int) -> list[int]:
    """Select reproducible, tercile-balanced cases, favoring multi-lesion cases.

    Terciles and selected slices use the supplied reference masks. The input
    inventory is a seeded candidate-pool subsample of the available labeled
    LiTS cases, not an unbiased cohort or a learned prediction. Missing terciles
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


def select_slice_records(slice_stats, n_slices: int, spacing_z_mm: float, *,
                         strategy: str = "stratified", minimum_gap_mm: float = 10.0,
                         min_tumor_pixels: int = 20,
                         legacy_minimum_gap: int = 2) -> tuple[list[dict], list[str]]:
    """Select slices from ``(index, tumor area, liver area)`` statistics.

    ``stratified`` assigns largest, median, small, p75 and tumor-free roles,
    choosing spaced slices in millimetres. For 3 slices it uses largest, small,
    tumor-free; 4 adds median; 5 uses all roles in the stated priority order.
    ``area_top`` preserves the historical area/liver/index ranking and index gap.
    Short roles are skipped and returned for CSV criteria and warning messages.
    """
    if n_slices not in (3, 4, 5):
        raise ValueError("n_slices must be 3, 4 or 5")
    if not np.isfinite(spacing_z_mm) or spacing_z_mm <= 0:
        raise ValueError("spacing_z_mm must be positive and finite")
    if not np.isfinite(minimum_gap_mm) or minimum_gap_mm < 0 or min_tumor_pixels < 0:
        raise ValueError("minimum_slice_gap_mm and min_tumor_pixels must be nonnegative")
    if strategy not in ("stratified", "area_top"):
        raise ValueError("slice_strategy must be 'stratified' or 'area_top'")
    rows = [{"slice_index": int(k), "tumor_area_px": int(t), "liver_area_px": int(l)}
            for k, t, l in slice_stats]
    rows.sort(key=lambda row: row["slice_index"])
    if not rows:
        return [], ["no slices available"]

    if strategy == "area_top":
        chosen = []
        for row in sorted(rows, key=lambda r: (-r["tumor_area_px"], -r["liver_area_px"], r["slice_index"])):
            if all(abs(row["slice_index"] - other["slice_index"]) >= legacy_minimum_gap
                   for other in chosen):
                chosen.append({**row, "slice_role": "area_top"})
                if len(chosen) == n_slices:
                    break
        return sorted(chosen, key=lambda r: r["slice_index"]), ([] if len(chosen) == n_slices else
                [f"requested {n_slices} area_top slices, selected {len(chosen)}"])

    eligible = [r for r in rows if r["tumor_area_px"] >= min_tumor_pixels]
    tumor = [r for r in rows if r["tumor_area_px"] > 0]
    free = [r for r in rows if r["tumor_area_px"] == 0 and r["liver_area_px"] > 0]
    areas = np.asarray([r["tumor_area_px"] for r in eligible], dtype=float)
    median = float(np.median(areas)) if len(areas) else None
    p75 = float(np.quantile(areas, .75)) if len(areas) else None
    role_candidates = {
        "largest": sorted(tumor, key=lambda r: (-r["tumor_area_px"], r["slice_index"])),
        "median": sorted(eligible, key=lambda r: (abs(r["tumor_area_px"] - median), r["slice_index"])) if median is not None else [],
        "small": sorted(eligible, key=lambda r: (r["tumor_area_px"], r["slice_index"])),
        "p75": sorted(eligible, key=lambda r: (abs(r["tumor_area_px"] - p75), r["slice_index"])) if p75 is not None else [],
        "tumor_free": sorted(free, key=lambda r: (-r["liver_area_px"], r["slice_index"])),
    }
    roles = ({3: ["largest", "small", "tumor_free"],
              4: ["largest", "median", "small", "tumor_free"],
              5: ["largest", "median", "small", "p75", "tumor_free"]})[n_slices]
    chosen: list[dict] = []
    shortages: list[str] = []
    for role in roles:
        picked = next((row for row in role_candidates[role]
                       if all(row["slice_index"] != other["slice_index"]
                              and abs(row["slice_index"] - other["slice_index"]) * spacing_z_mm
                              >= minimum_gap_mm - 1e-9 for other in chosen)), None)
        if picked is None:
            shortages.append(f"{role} role unavailable under tumor-area and {minimum_gap_mm:g} mm gap criteria")
            continue
        chosen.append({**picked, "slice_role": role})
    if len(chosen) < n_slices:
        shortages.append(f"requested {n_slices} slices, selected {len(chosen)}")
    return sorted(chosen, key=lambda r: r["slice_index"]), shortages
