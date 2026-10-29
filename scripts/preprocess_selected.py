"""Preprocess selected slices and save local before/after figures."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from psib.config import load_config
from psib.data_io import get_slice, load_case
from psib.preprocessing import preprocess_slice


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    out = Path(cfg["_output_dir"]) / "figures"
    selected = pd.read_csv(Path(cfg["_output_dir"]) / "selected_cases_slices.csv")
    for case_id, rows in selected.groupby("case_id", sort=False):
        ct, seg, _ = load_case(int(case_id), cfg["data_dir"])
        for row in rows.itertuples(index=False):
            hu = get_slice(ct, int(row.slice_index))
            liver = get_slice(seg, int(row.slice_index)) > 0
            processed = preprocess_slice(hu, liver, cfg)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, image, title in zip(axes, (processed["windowed"], processed["denoised"], processed["cropped"]),
                                        ("HU window", "Denoised", "Liver crop")):
                ax.imshow(image.T, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"Case {row.case_id} — slice {row.slice_index}")
            fig.tight_layout()
            fig.savefig(out / f"preprocess_case_{row.case_id:03d}_slice_{row.slice_index:03d}.png", dpi=150)
            plt.close(fig)
        del ct, seg
    print(f"Saved {len(selected)} preprocessing figures to {out}")


if __name__ == "__main__":
    main()
