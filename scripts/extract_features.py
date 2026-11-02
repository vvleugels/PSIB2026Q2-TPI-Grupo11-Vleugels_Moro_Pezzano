"""Extract 2D geometry and masked GLCM features from selected axial slices."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from psib.config import load_config
from psib.data_io import iter_case_slices
from psib.features import (
    GEOMETRY_COLUMNS,
    TEXTURE_COLUMNS,
    extract_geometric_features,
    extract_texture_features,
)
from psib.preprocessing import prepare_denoised_slice


TEXTURE_PROPERTIES = TEXTURE_COLUMNS[:-2]
TEXTURE_TEST_PROPERTIES = ("contrast", "homogeneity", "energy", "correlation", "entropy")
REGIONS = (("tumor_reference", 2), ("healthy_liver_reference", 1))


def read_selection(path: Path) -> pd.DataFrame:
    """Read and validate the case/slice pairs used by the 2D workflows."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe la selección {path}; ejecute scripts/select_slices_2d.py primero.")
    frame = pd.read_csv(path)
    if frame.empty or not {"case_id", "slice_index"}.issubset(frame.columns):
        raise ValueError("La selección debe tener filas y columnas case_id y slice_index.")
    for column in ("case_id", "slice_index"):
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"{column} debe contener enteros no negativos.")
        frame[column] = values.astype(np.int64)
    if frame.duplicated(["case_id", "slice_index"]).any():
        raise ValueError("La selección contiene pares case_id/slice_index duplicados.")
    return frame


def summarize_texture(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each GLCM property by reference-defined tissue region."""
    rows = []
    for region in (name for name, _ in REGIONS):
        selected = frame[frame.region == region]
        for property_name in TEXTURE_PROPERTIES:
            values = selected[property_name].dropna().to_numpy(dtype=float)
            rows.append({
                "region": region,
                "feature": property_name,
                "n_slices": int(values.size),
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std(ddof=0)) if values.size else float("nan"),
            })
    return pd.DataFrame(rows, columns=("region", "feature", "n_slices", "mean", "std"))


def summarize_texture_by_case(frame: pd.DataFrame) -> pd.DataFrame:
    """Average valid selected slices within each case and reference region."""
    rows = []
    for case_id in sorted(frame.case_id.unique()):
        for region_name, _ in REGIONS:
            selected = frame[(frame.case_id == case_id) & (frame.region == region_name)]
            valid = selected[TEXTURE_PROPERTIES[0]].notna()
            row: dict[str, object] = {
                "case_id": int(case_id),
                "region": region_name,
                "n_slices": int(valid.sum()),
            }
            for property_name in TEXTURE_PROPERTIES:
                values = pd.to_numeric(selected[property_name], errors="coerce").dropna()
                row[property_name] = float(values.mean()) if not values.empty else float("nan")
            rows.append(row)
    columns = ("case_id", "region", "n_slices", *TEXTURE_PROPERTIES)
    return pd.DataFrame(rows, columns=columns)


def summarize_paired_texture_differences(case_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired differences and exact sign tests by GLCM property."""
    details = []
    summaries = []
    for property_name in TEXTURE_TEST_PROPERTIES:
        pivot = case_frame.pivot(index="case_id", columns="region", values=property_name)
        if not all(region in pivot.columns for region, _ in REGIONS):
            paired = pd.DataFrame(columns=[region for region, _ in REGIONS])
        else:
            paired = pivot.dropna(subset=[region for region, _ in REGIONS])
        differences = paired[REGIONS[0][0]] - paired[REGIONS[1][0]]
        for case_id, row in paired.iterrows():
            details.append({
                "case_id": int(case_id),
                "feature": property_name,
                "tumor_reference": float(row[REGIONS[0][0]]),
                "healthy_liver_reference": float(row[REGIONS[1][0]]),
                "difference_tumor_minus_liver": float(row[REGIONS[0][0]] - row[REGIONS[1][0]]),
            })
        summaries.append({
            "feature": property_name,
            "n_pairs": int(len(differences)),
            "mean_difference": float(differences.mean()) if len(differences) else float("nan"),
            "median_difference": float(differences.median()) if len(differences) else float("nan"),
            "std_difference": float(differences.std(ddof=0)) if len(differences) else float("nan"),
            "n_tumor_higher": int((differences > 0).sum()),
            "n_liver_higher": int((differences < 0).sum()),
            "n_equal": int((differences == 0).sum()),
            "n_nonzero": int((differences != 0).sum()),
            "p_sign_test": _exact_sign_test_pvalue(
                int((differences > 0).sum()), int((differences < 0).sum())
            ),
        })
    adjusted = _holm_adjust([row["p_sign_test"] for row in summaries])
    for row, p_value in zip(summaries, adjusted):
        row["p_holm"] = p_value
    detail_columns = (
        "case_id", "feature", "tumor_reference", "healthy_liver_reference",
        "difference_tumor_minus_liver",
    )
    summary_columns = (
        "feature", "n_pairs", "mean_difference", "median_difference", "std_difference",
        "n_tumor_higher", "n_liver_higher", "n_equal", "n_nonzero", "p_sign_test", "p_holm",
    )
    return (pd.DataFrame(details, columns=detail_columns),
            pd.DataFrame(summaries, columns=summary_columns))


def _exact_sign_test_pvalue(n_positive: int, n_negative: int) -> float:
    """Two-sided exact sign-test p-value, excluding tied pairs."""
    n = n_positive + n_negative
    if n == 0:
        return float("nan")
    lower_tail = min(n_positive, n_negative)
    probability = sum(math.comb(n, k) for k in range(lower_tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values, preserving NaNs and original order."""
    adjusted = [float("nan")] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running_max = max(running_max, min(1.0, (count - rank) * value))
        adjusted[index] = running_max
    return adjusted


def _extract_one_level(cfg, output_dir: Path, levels: int) -> tuple[Path, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = read_selection(Path(cfg["_output_dir"]) / "selected_cases_slices.csv")
    texture_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    reasons_by_case: dict[int, list[str]] = {}
    for case_id, rows in selection.groupby("case_id", sort=True):
        indices = sorted(int(value) for value in rows.slice_index)
        tumor_reasons = []
        for slice_index, ct, seg, spacing in iter_case_slices(int(case_id), indices, cfg["data_dir"]):
            image = prepare_denoised_slice(ct, cfg)
            for region_name, label_value in REGIONS:
                roi = seg == label_value
                texture = extract_texture_features(image, roi, levels=levels)
                nan_reason = ("empty_roi" if not np.any(roi) else
                              "no_valid_pairs" if texture["pair_count"] == 0 else "")
                if region_name == "tumor_reference" and nan_reason:
                    tumor_reasons.append(f"corte {slice_index}: {nan_reason}")
                texture_rows.append({
                    "case_id": int(case_id), "slice_index": int(slice_index), "region": region_name,
                    "pixel_count": int(np.count_nonzero(roi)),
                    "area_mm2": float(np.count_nonzero(roi) * spacing[0] * spacing[1]),
                    **texture, "nan_reason": nan_reason,
                })
            component_table = extract_geometric_features(seg == 2, spacing[:2])
            geometry_rows.extend({"case_id": int(case_id), "slice_index": int(slice_index), **component}
                                 for component in component_table.to_dict("records"))
        if tumor_reasons and len(tumor_reasons) == len(indices):
            reasons_by_case[int(case_id)] = tumor_reasons

    texture_columns = ("case_id", "slice_index", "region", "pixel_count", "area_mm2",
                       *TEXTURE_PROPERTIES, "pair_count", "n_offsets", "nan_reason")
    texture_frame = pd.DataFrame(texture_rows, columns=texture_columns)
    geometry_frame = pd.DataFrame(geometry_rows, columns=("case_id", "slice_index", *GEOMETRY_COLUMNS))
    summary = summarize_texture(texture_frame)
    case_summary = summarize_texture_by_case(texture_frame)
    paired_differences, difference_summary = summarize_paired_texture_differences(case_summary)
    texture_frame.to_csv(output_dir / "texture_per_region.csv", index=False)
    geometry_frame.to_csv(output_dir / "tumor_geometry.csv", index=False)
    summary.to_csv(output_dir / "texture_summary.csv", index=False)
    case_summary.to_csv(output_dir / "texture_per_case.csv", index=False)
    paired_differences.to_csv(output_dir / "texture_case_differences.csv", index=False)
    difference_summary.to_csv(output_dir / "texture_difference_summary.csv", index=False)
    diagnostic_lines = [f"Casos sin textura tumoral computable en todos sus cortes (GLCM levels={levels}):"]
    diagnostic_lines.extend(f"Caso {case_id:03d}: " + "; ".join(reasons)
                            for case_id, reasons in sorted(reasons_by_case.items()))
    if not reasons_by_case:
        diagnostic_lines.append("Ningún caso quedó sin textura computable en todos sus cortes.")
    (output_dir / "diagnostics.txt").write_text("\n".join(diagnostic_lines) + "\n", encoding="utf-8")
    print(f"Saved feature tables at {levels} GLCM levels under {output_dir}", flush=True)
    return output_dir, texture_frame


def extract_features(config_path: str | Path | None = None,
                     levels: list[int] | tuple[int, ...] | None = None) -> Path:
    """Write feature tables at the configured GLCM level or requested sensitivity levels."""
    cfg = load_config(config_path)
    root = Path(cfg["_output_dir"]) / "features"
    if levels is None:
        configured = cfg.get("features", {}).get("glcm_levels", 16)
        if isinstance(configured, bool) or not isinstance(configured, int) or not 2 <= configured <= 256:
            raise ValueError("features.glcm_levels debe ser un entero entre 2 y 256.")
        return _extract_one_level(cfg, root, configured)[0]
    levels = tuple(levels)
    if not levels or any(isinstance(value, bool) or value not in (8, 16, 32) for value in levels):
        raise ValueError("levels debe incluir uno o más valores de 8, 16 o 32.")
    if len(set(levels)) != len(levels):
        raise ValueError("levels no debe repetir valores.")
    sensitivity = []
    for level in levels:
        level_dir = root / f"levels_{level}"
        _, frame = _extract_one_level(cfg, level_dir, level)
        for region, _ in REGIONS:
            selected = frame[frame.region == region]
            for property_name in TEXTURE_PROPERTIES:
                values = pd.to_numeric(selected[property_name], errors="coerce").dropna()
                sensitivity.append({"levels": level, "region": region, "feature": property_name,
                                    "n_values": len(values),
                                    "median": float(values.median()) if len(values) else np.nan})
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensitivity).to_csv(root / "texture_levels_sensitivity.csv", index=False)
    print(f"Saved GLCM level sensitivity under {root}", flush=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--levels", type=int, nargs="+", choices=(8, 16, 32),
                        help="Run sensitivity analysis, e.g. --levels 8 16 32.")
    args = parser.parse_args()
    extract_features(args.config, args.levels)


if __name__ == "__main__":
    main()
