"""Create the reference-mask-based dataset inventory and context figures."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psib.config import load_config
from psib.data_io import list_cases, load_case
from psib.dataset import build_inventory


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    out = Path(cfg["_output_dir"])
    selection = cfg.get("selection", {})
    pool_size = int(selection.get("candidate_pool_size", 15))
    seed = int(selection.get("seed", 2026))
    inventory = build_inventory(cfg["data_dir"], pool_size, seed)
    inventory.to_csv(out / "dataset_inventory.csv", index=False)
    inventory[["candidate_pool_rank", "case_id", "candidate_pool_seed"]].assign(
        pool_criteria=f"seeded random subsample from {len(list_cases(cfg['data_dir']))} labeled pairs; excludes unlabeled test images"
    ).to_csv(out / "candidate_pool.csv", index=False)
    figures = out / "figures"
    # Estimate the HU distribution from the candidate pool, sampling each case
    # independently so memory remains bounded to one volume at a time.
    hu_values = []
    import numpy as np
    rng = np.random.default_rng(seed)
    for case_id in inventory.case_id:
        ct, _, _ = load_case(int(case_id), cfg["data_dir"])
        flat = ct.ravel()
        take = min(100_000, flat.size)
        hu_values.append(flat[rng.choice(flat.size, size=take, replace=False)])
        del ct
    plots = (("hu_histogram.png", np.concatenate(hu_values), "CT intensity (HU; sampled voxels)", "Voxels"),
             ("tumor_volume_distribution.png", inventory.tumor_volume_cm3, "Tumor volume (cm³)", "Cases"),
             ("spacing_distribution.png", inventory.spacing_z_mm, "Slice spacing (mm)", "Cases"))
    for filename, values, xlabel, ylabel in plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=40 if filename == "hu_histogram.png" else 20, color="#367a9a", edgecolor="white")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Candidate pool (n={len(inventory)} of 131 labeled cases)")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
    print(f"Saved candidate pool ({len(inventory)} of 131 labeled cases), inventory, and context figures under {out}")


if __name__ == "__main__":
    main()
