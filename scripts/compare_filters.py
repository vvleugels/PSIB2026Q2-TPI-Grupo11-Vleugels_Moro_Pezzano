"""Compare denoising filters on the first selected CT slice and save a figure."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from psib.config import load_config
from psib.data_io import iter_case_slices
from psib.preprocessing import apply_hu_window, denoise


def compare_filters(config_path: str | Path = ROOT / "config.yaml") -> Path:
    """Save windowed, median, Gaussian, and bilateral views of one selected slice."""
    cfg = load_config(config_path)
    selection_path = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Selection CSV not found: {selection_path}. "
            "Run scripts/select_slices_2d.py or scripts/select_cases.py first."
        )
    try:
        selected = pd.read_csv(selection_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Selection CSV is empty: {selection_path}") from exc
    if selected.empty:
        raise ValueError(f"Selection CSV is empty: {selection_path}")

    row = selected.iloc[0]
    case_id, slice_index = int(row.case_id), int(row.slice_index)
    _, slice_hu, _, _ = next(iter_case_slices(case_id, [slice_index], cfg["data_dir"]))
    settings = cfg.get("preprocessing", {})
    window = settings.get("hu_window", {})
    center, width = float(window.get("center", 60)), float(window.get("width", 400))
    windowed = apply_hu_window(slice_hu, center, width)
    filters = settings.get("denoising", {})
    images = [windowed] + [
        denoise(windowed, {**filters, "method": method})
        for method in ("median", "gaussian", "bilateral")
    ]
    titles = (
        f"HU window\ncenter={center:g}, width={width:g}",
        f"Median\nsize={int(filters.get('median_size', 3))}",
        f"Gaussian\nsigma={float(filters.get('gaussian_sigma', 1.0)):g}",
        "Bilateral\n"
        f"color={float(filters.get('bilateral_sigma_color', 0.08)):g}, "
        f"spatial={float(filters.get('bilateral_sigma_spatial', 3.0)):g}",
    )
    output = Path(cfg["_output_dir"]) / "figures" / (
        f"filter_comparison_case_{case_id:03d}_slice_{slice_index:03d}.png"
    )
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    try:
        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_axis_off()
        figure.suptitle(f"Case {case_id} — slice {slice_index}")
        figure.tight_layout()
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)
    print(f"Saved filter comparison to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    try:
        compare_filters(args.config)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
