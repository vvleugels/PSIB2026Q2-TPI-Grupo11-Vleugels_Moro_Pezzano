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
from psib.dataset import build_inventory, dataset_context_table


def save_context_outputs(inventory, output_dir: str | Path):
    """Write the report-ready case table and labeled HU/volume scatter figure."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "dataset_context_table.csv"
    dataset_context_table(inventory).to_csv(table_path, index=False)
    figure_path = output / "figures" / "delta_hu_vs_tumor_volume.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    valid = inventory["delta_hu"].notna() & inventory["tumor_volume_cm3"].notna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(inventory.loc[valid, "tumor_volume_cm3"], inventory.loc[valid, "delta_hu"],
               color="#367a9a", edgecolor="white", zorder=2)
    for row in inventory.loc[valid].itertuples():
        ax.annotate(str(row.case_id), (row.tumor_volume_cm3, row.delta_hu),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#777777", linewidth=.8)
    ax.set(xlabel="Tumor volume (cm³)", ylabel="Median tumor HU − healthy-liver HU",
           title="HU contrast versus tumor volume by case")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return table_path, figure_path


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    out = Path(cfg["_output_dir"])
    selection = cfg.get("selection", {})
    pool_size = int(selection.get("candidate_pool_size", 15))
    seed = int(selection.get("seed", 2026))
    density_tolerance = float(cfg.get("dataset_context", {}).get("density_tolerance_hu", 5.0))
    available_cases = list_cases(cfg["data_dir"])
    inventory = build_inventory(cfg["data_dir"], pool_size, seed, density_tolerance)
    inventory.to_csv(out / "dataset_inventory.csv", index=False)
    save_context_outputs(inventory, out)
    inventory[["candidate_pool_rank", "case_id", "candidate_pool_seed"]].assign(
        pool_criteria=f"seeded random subsample from {len(available_cases)} labeled pairs; excludes unlabeled test images"
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
        ax.set(xlabel=xlabel, ylabel=ylabel,
               title=f"Candidate pool (n={len(inventory)} of {len(available_cases)} labeled cases)")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
    print(f"Saved candidate pool ({len(inventory)} of {len(available_cases)} labeled cases), "
          f"inventory, and context figures under {out}")


if __name__ == "__main__":
    main()
