"""Bland-Altman and correlation analysis for matched 2D lesion groups."""
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
from scipy import stats


SIZE_CLASSES = ("<10", "10-20", ">=20", "none")
SIZE_COLORS = {"<10": "#4C9F70", "10-20": "#E3A63B", ">=20": "#5878B9", "none": "#888888"}
PROPERTIES = (
    ("area", "area_ref_mm2", "area_pred_mm2"),
    ("max_feret", "max_feret_ref_mm", "max_feret_pred_mm"),
)


def agreement_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate absolute/relative Bland-Altman summaries and correlations."""
    required = {"status", *(column for _, ref, pred in PROPERTIES for column in (ref, pred)), "size_class"}
    if not required.issubset(frame.columns):
        raise ValueError(f"La tabla debe tener columnas {sorted(required)}.")
    matched = frame.loc[frame.status == "TP"]
    if matched.empty:
        raise ValueError("No hay grupos TP para comparar medidas automáticas y de referencia.")
    rows = []
    for name, ref_col, pred_col in PROPERTIES:
        data = matched[[ref_col, pred_col]].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(data[ref_col]) & np.isfinite(data[pred_col]) & (data[ref_col] > 0)
        data = matched.loc[valid, [ref_col, pred_col, "size_class"]].copy()
        reference = data[ref_col].to_numpy(dtype=float)
        prediction = data[pred_col].to_numpy(dtype=float)
        difference = prediction - reference
        mean = float(((prediction + reference) / 2).mean()) if len(data) else np.nan
        mean_difference = float(difference.mean()) if len(data) else np.nan
        sd_difference = float(difference.std(ddof=1)) if len(data) > 1 else np.nan
        percent_difference = 100.0 * difference / reference if len(data) else np.asarray([])
        mean_percent = float(percent_difference.mean()) if len(data) else np.nan
        sd_percent = float(percent_difference.std(ddof=1)) if len(data) > 1 else np.nan
        pearson = _correlation(stats.pearsonr, reference, prediction)
        spearman = _correlation(stats.spearmanr, reference, prediction)
        rows.append({
            "measure": name, "n_matched_groups": len(data), "mean_reference": float(reference.mean()) if len(data) else np.nan,
            "mean_automatic": float(prediction.mean()) if len(data) else np.nan,
            "mean_of_pair_means": mean, "mean_difference": mean_difference,
            "sd_difference": sd_difference,
            "loa_lower": mean_difference - 1.96 * sd_difference if np.isfinite(sd_difference) else np.nan,
            "loa_upper": mean_difference + 1.96 * sd_difference if np.isfinite(sd_difference) else np.nan,
            "mean_percent_difference": mean_percent, "sd_percent_difference": sd_percent,
            "loa_percent_lower": mean_percent - 1.96 * sd_percent if np.isfinite(sd_percent) else np.nan,
            "loa_percent_upper": mean_percent + 1.96 * sd_percent if np.isfinite(sd_percent) else np.nan,
            "pearson_r": pearson[0], "pearson_p": pearson[1],
            "spearman_rho": spearman[0], "spearman_p": spearman[1],
        })
    return pd.DataFrame(rows)


def _correlation(function, reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    if len(reference) < 2 or np.ptp(reference) == 0 or np.ptp(prediction) == 0:
        return np.nan, np.nan
    result = function(reference, prediction)
    return float(result.statistic), float(result.pvalue)


def plot_bland_altman(matches: pd.DataFrame, measure: str, summary: pd.DataFrame, path: Path) -> Path:
    columns = {"area": ("area_ref_mm2", "area_pred_mm2", "Área (mm²)"),
               "max_feret": ("max_feret_ref_mm", "max_feret_pred_mm", "Feret máximo (mm)")}
    if measure not in columns:
        raise ValueError("measure debe ser 'area' o 'max_feret'.")
    ref_col, pred_col, label = columns[measure]
    matched = matches.loc[matches.status == "TP"].copy()
    matched = matched.loc[np.isfinite(matched[ref_col]) & np.isfinite(matched[pred_col]) & (matched[ref_col] > 0)]
    x = (matched[ref_col].to_numpy(float) + matched[pred_col].to_numpy(float)) / 2
    y = matched[pred_col].to_numpy(float) - matched[ref_col].to_numpy(float)
    row = summary.loc[summary.measure == measure].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        for size_class in SIZE_CLASSES:
            selected = matched.size_class.astype(str).to_numpy() == size_class
            if selected.any():
                ax.scatter(x[selected], y[selected], color=SIZE_COLORS[size_class], label=size_class, alpha=0.8)
        ax.axhline(row.mean_difference, color="#222222", linewidth=1.2, label="Sesgo medio")
        if np.isfinite(row.loa_lower):
            ax.axhline(row.loa_lower, color="#A44848", linestyle="--", label="Límites 95 %")
            ax.axhline(row.loa_upper, color="#A44848", linestyle="--")
        ax.set(xlabel=f"Media de referencia y automática · {label}", ylabel=f"Automática − referencia · {label}",
               title=f"Bland–Altman · {measure}")
        ax.grid(alpha=0.25)
        ax.legend(title="Tamaño ref.")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path


def analyze(matches_path: Path, output_dir: Path | None = None) -> Path:
    matches = pd.read_csv(matches_path)
    summary = agreement_statistics(matches)
    output = output_dir or matches_path.parent / "agreement"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "agreement_summary.csv", index=False)
    for measure, _, _ in PROPERTIES:
        plot_bland_altman(matches, measure, summary, output / f"bland_altman_{measure}.png")
    print(f"Agreement analysis saved to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesion-matches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        analyze(args.lesion_matches.resolve(), args.output_dir.resolve() if args.output_dir else None)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error en análisis de concordancia: {exc}\n")


if __name__ == "__main__":
    main()
