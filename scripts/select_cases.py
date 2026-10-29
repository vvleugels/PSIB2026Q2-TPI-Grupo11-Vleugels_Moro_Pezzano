"""Select balanced cases and tumor-rich slices from reference masks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from psib.config import load_config
from psib.data_io import load_segmentation
from psib.dataset import select_cases, select_slices, tumor_burden_terciles


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    selection = cfg.get("selection", {})
    n_cases = int(selection.get("n_cases", 10))
    seed = int(selection.get("seed", 2026))
    n_slices = int(selection.get("slices_per_case", 5))
    gap = int(selection.get("minimum_slice_gap", 5))
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
    rows = []
    for rank, case_id in enumerate(chosen, start=1):
        seg = load_segmentation(case_id, cfg["data_dir"])
        slices = select_slices(seg, n_slices, gap)
        case = inventory.loc[inventory.case_id == case_id].iloc[0]
        for slice_index in slices:
            rows.append({"selection_rank": rank, "case_id": case_id, "slice_index": slice_index,
                         "tumor_volume_cm3": case.tumor_volume_cm3,
                         "tumor_lesions_3d": case.tumor_lesions_3d,
                         "burden_tercile": case.burden_tercile,
                         "candidate_pool_size": len(inventory), "seed": seed,
                         "selection_criteria": f"seed={seed}; balanced across candidate-pool tumor-volume terciles; "
                                               f"prioritize at least {multi_target} multi-lesion cases when available; "
                                               f"top tumor-area slices, minimum gap={gap}; selected using reference masks"})
    target = out / "selected_cases_slices.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    actual_multi = int(inventory[inventory.case_id.isin(chosen)].tumor_lesions_3d.gt(1).sum())
    print(f"Selected {len(chosen)} cases across small/medium/large terciles; "
          f"multi-lesion cases={actual_multi}; case/slice pairs={len(rows)}; saved {target}")


if __name__ == "__main__":
    main()
