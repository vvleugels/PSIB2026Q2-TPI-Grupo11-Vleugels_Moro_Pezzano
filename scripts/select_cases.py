"""Select balanced cases and tumor-rich slices from reference masks."""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from psib.config import load_config
from psib.data_io import load_segmentation
from psib.dataset import select_cases, select_slice_records, tumor_burden_terciles


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    selection = cfg.get("selection", {})
    n_cases = int(selection.get("n_cases", 10))
    seed = int(selection.get("seed", 2026))
    n_slices = int(selection.get("slices_per_case", 5))
    strategy = str(selection.get("slice_strategy", "stratified"))
    gap_mm = float(selection.get("minimum_slice_gap_mm", 10.0))
    min_tumor = int(selection.get("min_tumor_pixels", 20))
    legacy_gap = int(selection.get("minimum_slice_gap", 2))
    if not 3 <= n_slices <= 5:
        raise ValueError("selection.slices_per_case must be between 3 and 5")
    out = Path(cfg["_output_dir"])
    inventory_path = out / "dataset_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError("Run scripts/explore_dataset.py first to create the candidate-pool inventory")
    inventory = pd.read_csv(inventory_path)
    try:
        chosen = select_cases(inventory, n_cases, seed)
    except ValueError as exc:
        raise SystemExit(f"Cannot select final cases from candidate pool: {exc}") from exc
    inventory["burden_tercile"] = tumor_burden_terciles(inventory)
    multi_target = min(2, n_cases, int((inventory.tumor_lesions_3d > 1).sum()))
    gap_description = f"minimum index gap={legacy_gap}" if strategy == "area_top" else f"minimum gap={gap_mm:g} mm"
    rows = []
    for rank, case_id in enumerate(chosen, start=1):
        seg = load_segmentation(case_id, cfg["data_dir"])
        case = inventory.loc[inventory.case_id == case_id].iloc[0]
        spacing_z = float(case.spacing_z_mm)
        stats = [(k, int((seg[:, :, k] == 2).sum()), int((seg[:, :, k] > 0).sum()))
                 for k in range(seg.shape[2])]
        slices, shortages = select_slice_records(stats, n_slices, spacing_z,
            strategy=strategy, minimum_gap_mm=gap_mm, min_tumor_pixels=min_tumor,
            legacy_minimum_gap=legacy_gap)
        if shortages:
            warnings.warn(f"Case {case_id}: " + "; ".join(shortages), RuntimeWarning)
        for slice_row in slices:
            slice_index = slice_row["slice_index"]
            rows.append({"selection_rank": rank, "case_id": case_id, "slice_index": slice_index,
                         "tumor_volume_cm3": case.tumor_volume_cm3,
                         "tumor_lesions_3d": case.tumor_lesions_3d,
                         "burden_tercile": case.burden_tercile,
                         "candidate_pool_size": len(inventory), "seed": seed,
                         "selection_criteria": f"seed={seed}; balanced across candidate-pool tumor-volume terciles; "
                                               f"prioritize at least {multi_target} multi-lesion cases when available; "
                                               f"strategy={strategy}, {gap_description}; selected using reference masks; "
                                               + ('shortage: ' + '; '.join(shortages) if shortages else 'all requested roles filled'),
                         "slice_role": slice_row["slice_role"],
                         "tumor_area_px": slice_row["tumor_area_px"],
                         "liver_area_px": slice_row["liver_area_px"],
                         "slice_position_mm": slice_index * spacing_z})
    target = out / "selected_cases_slices.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    actual_multi = int(inventory[inventory.case_id.isin(chosen)].tumor_lesions_3d.gt(1).sum())
    print(f"Selected {len(chosen)} cases across small/medium/large terciles; "
          f"multi-lesion cases={actual_multi}; case/slice pairs={len(rows)}; saved {target}")


if __name__ == "__main__":
    main()
