"""Plot descriptive GLCM comparisons between reference tumor and healthy liver."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psib.config import load_config


REGION_ORDER = ("tumor_reference", "healthy_liver_reference")
REGION_LABELS = ("Tumor (referencia)", "Hígado sano (referencia)")
FEATURES = (
    ("contrast", "Contraste"),
    ("homogeneity", "Homogeneidad"),
    ("energy", "Energía"),
    ("correlation", "Correlación"),
)
COLORS = ("#2878a5", "#55a868")


def plot_texture_comparison(features_dir: str | Path) -> Path:
    """Save four boxplots from the per-region texture table and return the path."""
    features_dir = Path(features_dir)
    source = features_dir / "texture_per_region.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Falta {source}; ejecute scripts/extract_features.py primero.")
    frame = pd.read_csv(source)
    required = {"region", *(name for name, _ in FEATURES)}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{source} debe contener filas y columnas {sorted(required)}.")

    values_by_region = {}
    for region, label in zip(REGION_ORDER, REGION_LABELS):
        selected = frame.loc[frame.region == region]
        if selected.empty:
            raise ValueError(f"No hay observaciones para la región {label} en {source}.")
        values_by_region[region] = {}
        for feature, _ in FEATURES:
            values = pd.to_numeric(selected[feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"No hay valores finitos de {feature} para la región {label}.")
            values_by_region[region][feature] = values

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    try:
        for ax, (feature, title) in zip(axes.flat, FEATURES):
            data = [values_by_region[region][feature] for region in REGION_ORDER]
            labels = [f"{label}\n(n={len(values)})" for label, values in zip(REGION_LABELS, data)]
            artists = ax.boxplot(data, patch_artist=True, showmeans=True)
            ax.set_xticks((1, 2), labels=labels)
            for patch, color in zip(artists["boxes"], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
            ax.set_title(title)
            ax.set_ylabel("Valor GLCM")
            ax.grid(axis="y", alpha=0.25)
        figure.suptitle(
            "Textura GLCM por región de referencia\n"
            "Comparación descriptiva por corte (línea: mediana; triángulo: media)"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        output = features_dir / "figures" / "glcm_tissue_comparison.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output


def plot_case_texture_comparison(features_dir: str | Path) -> Path:
    """Plot case-averaged GLCM values with lines connecting paired regions."""
    features_dir = Path(features_dir)
    source = features_dir / "texture_per_case.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Falta {source}; ejecute scripts/extract_features.py primero.")
    frame = pd.read_csv(source)
    required = {"case_id", "region", *(name for name, _ in FEATURES)}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{source} debe contener filas y columnas {sorted(required)}.")

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    try:
        for ax, (feature, title) in zip(axes.flat, FEATURES):
            pivot = frame.pivot(index="case_id", columns="region", values=feature)
            tumor = pivot[REGION_ORDER[0]].dropna() if REGION_ORDER[0] in pivot else pd.Series(dtype=float)
            healthy = pivot[REGION_ORDER[1]].dropna() if REGION_ORDER[1] in pivot else pd.Series(dtype=float)
            if tumor.empty or healthy.empty:
                raise ValueError(f"Se requieren valores de {feature} para ambas regiones en {source}.")

            paired = pivot.dropna(subset=list(REGION_ORDER))
            if not paired.empty:
                jitter = np.linspace(-0.07, 0.07, len(paired)) if len(paired) > 1 else np.array([0.0])
                for offset, row in zip(jitter, paired.itertuples()):
                    ax.plot((1 + offset, 2 + offset),
                            (getattr(row, REGION_ORDER[0]), getattr(row, REGION_ORDER[1])),
                            color="#777777", alpha=0.4, linewidth=0.9, zorder=1)

            data = (tumor.to_numpy(dtype=float), healthy.to_numpy(dtype=float))
            artists = ax.boxplot(data, patch_artist=True, showmeans=True)
            for patch, color in zip(artists["boxes"], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
            if not paired.empty:
                ax.scatter(1 + jitter, paired[REGION_ORDER[0]], color=COLORS[0], s=20, zorder=3)
                ax.scatter(2 + jitter, paired[REGION_ORDER[1]], color=COLORS[1], s=20, zorder=3)
            ax.set_xticks((1, 2), labels=(
                f"Tumor\n(n={len(tumor)} casos)",
                f"Hígado sano\n(n={len(healthy)} casos)",
            ))
            ax.set_title(title)
            ax.set_ylabel("Media de cortes por caso")
            ax.grid(axis="y", alpha=0.25)
        figure.suptitle(
            "Textura GLCM por caso (cortes seleccionados)\n"
            "Las líneas conectan ambas regiones del mismo caso"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        output = features_dir / "figures" / "glcm_tissue_comparison_by_case.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    features_dir = Path(cfg["_output_dir"]) / "features"
    slice_plot = plot_texture_comparison(features_dir)
    case_plot = plot_case_texture_comparison(features_dir)
    print(f"Saved texture comparison figures to {slice_plot} and {case_plot}", flush=True)


if __name__ == "__main__":
    main()
