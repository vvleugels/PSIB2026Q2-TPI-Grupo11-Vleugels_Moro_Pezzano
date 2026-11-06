"""Per-case run summaries focused on medium (10–20 mm) and large (>=20 mm) tumors.

Works for 2D segmentation runs (``results/segmentation/<run>``) and for 3D
stacked-prediction outputs (``results/segmentation/<run>/morphology3d_prediction*``).
Lesion size is the maximum axial diameter between voxel/pixel centers, in mm.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

SIZE_GROUPS = (("medium", "Medianos 10–20 mm", 10.0, 20.0), ("large", "Grandes ≥20 mm", 20.0, np.inf))
COVERED_FRACTION = 0.5
MEAN_LABEL = "Media"


def _completed(state_path: Path) -> bool:
    try:
        return json.loads(state_path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, ValueError):
        return False


def configurations(run_dir: Path) -> list[str]:
    """Configuration IDs evaluated in a 2D run (one for normal runs, many for sweeps)."""
    path = Path(run_dir) / "metrics_per_case.csv"
    if not path.is_file():
        return []
    ids = pd.read_csv(path, usecols=["configuration_id"]).configuration_id.astype(str)
    return sorted(ids.unique())


def list_runs(results_dir: Path) -> list[dict]:
    """Completed 2D runs and 3D outputs, newest first, with a human-readable label."""
    runs = []
    for state in (Path(results_dir) / "segmentation").glob("*/run.json"):
        run_dir = state.parent
        if not _completed(state) or not (run_dir / "metrics_per_case.csv").is_file():
            continue
        n_cases = pd.read_csv(run_dir / "metrics_per_case.csv", usecols=["case_id"]).case_id.nunique()
        runs.append({"kind": "2D", "path": run_dir, "mtime": state.stat().st_mtime,
                     "label": f"2D · {run_dir.name} · {n_cases} casos"})
        for output in run_dir.glob("morphology3d_prediction*/run.json"):
            if not _completed(output) or not (output.parent / "morphology_3d_prediction_per_case.csv").is_file():
                continue
            info = json.loads(output.read_text(encoding="utf-8"))
            filters = []
            if info.get("min_component_mm3"):
                filters.append(f"≥{info['min_component_mm3']:g} mm³")
            if (info.get("min_slices") or 1) > 1:
                filters.append(f"≥{info['min_slices']} cortes")
            runs.append({"kind": "3D", "path": output.parent, "mtime": output.stat().st_mtime,
                         "label": f"3D · {run_dir.name}/{output.parent.name} · {info.get('n_cases', '?')} casos"
                                  + (f" · filtro {', '.join(filters)}" if filters else " · sin filtro")})
    return sorted(runs, key=lambda item: item["mtime"], reverse=True)


def lesion_file_2d(run_dir: Path, configuration_id: str) -> Path:
    name = "lesion_size_metrics.csv" if configuration_id == "cfg_000" else f"lesion_size_metrics_{configuration_id}.csv"
    return Path(run_dir) / name


def compute_lesion_sizes_2d(run_dir: Path, configuration_id: str = "cfg_000", data_dir=None,
                            progress=print) -> Path:
    """Per-lesion metrics for every reference lesion on the evaluated slices of a 2D run.

    Lesions are 8-connected reference components per slice; a lesion's matched
    prediction is the union of predicted components touching it.
    """
    from .config import load_config
    from .data_io import iter_case_slices
    from .morphology3d import _maximum_pair_distance

    run_dir = Path(run_dir)
    slices = pd.read_csv(run_dir / "metrics_per_slice.csv")
    slices = slices[slices.configuration_id.astype(str) == configuration_id]
    if slices.empty:
        raise ValueError(f"La corrida {run_dir.name} no tiene cortes para {configuration_id}.")
    data_dir = data_dir or load_config(run_dir / "effective_config.yaml")["data_dir"]
    prediction_dir = run_dir / "predictions" / configuration_id
    structure = np.ones((3, 3), dtype=bool)
    rows = []
    cases = sorted(slices.case_id.unique())
    for position, (case_id, group) in enumerate(slices.groupby("case_id"), start=1):
        progress(f"Lesiones por tamaño · caso {int(case_id):03d} ({position}/{len(cases)})")
        wanted = sorted(int(k) for k in group.slice_index)
        for k, _, labels, spacing in iter_case_slices(int(case_id), wanted, data_dir, mask_only=True):
            reference = labels == 2
            prediction = np.load(prediction_dir / f"case_{int(case_id):03d}_slice_{int(k):03d}.npy").astype(bool)
            ref_labels, n_ref = ndimage.label(reference, structure)
            pred_labels, _ = ndimage.label(prediction, structure)
            pred_sizes = np.bincount(pred_labels.ravel())
            for lesion_id in range(1, n_ref + 1):
                lesion = ref_labels == lesion_id
                touching = np.unique(pred_labels[lesion & prediction])
                touching = touching[touching > 0]
                intersection = int((lesion & prediction).sum())
                area = int(lesion.sum())
                matched = int(pred_sizes[touching].sum()) if touching.size else 0
                rows.append({"configuration_id": configuration_id, "case_id": int(case_id), "slice_index": int(k),
                             "area_px": area,
                             "max_axial_feret_mm": _maximum_pair_distance(np.argwhere(lesion) * np.asarray(spacing[:2])),
                             "coverage": intersection / area, "lesion_dice": 2 * intersection / (area + matched),
                             "detected": intersection > 0})
    output = lesion_file_2d(run_dir, configuration_id)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _load(path: Path, configuration_id: str | None):
    """Return (per-case frame with case_dice/precision, lesion frame or None, kind)."""
    path = Path(path)
    if (path / "morphology_3d_prediction_per_case.csv").is_file():
        cases = pd.read_csv(path / "morphology_3d_prediction_per_case.csv")
        ref, pred = cases.reference_volume_cm3, cases.prediction_clean_volume_cm3
        intersection = cases.dice_3d_clean * (ref + pred) / 2
        cases = pd.DataFrame({"case_id": cases.case_id, "case_dice": cases.dice_3d_clean,
                              "precision": np.where(pred > 0, intersection / pred.where(pred > 0), np.nan)})
        lesion_path = path / "morphology_3d_prediction_lesions.csv"
        lesions = pd.read_csv(lesion_path) if lesion_path.is_file() else None
        return cases, lesions, "3D"
    metrics = pd.read_csv(path / "metrics_per_case.csv")
    configuration_id = configuration_id or str(metrics.configuration_id.astype(str).iloc[0])
    metrics = metrics[metrics.configuration_id.astype(str) == configuration_id]
    cases = pd.DataFrame({"case_id": metrics.case_id, "case_dice": metrics.dice,
                          "precision": np.where(metrics.pred_area > 0,
                                                metrics.intersection / metrics.pred_area.where(metrics.pred_area > 0),
                                                np.nan)})
    lesion_path = lesion_file_2d(path, configuration_id)
    lesions = None
    if lesion_path.is_file():
        lesions = pd.read_csv(lesion_path)
        # Older files from the exploratory script used "feret_mm".
        if "max_axial_feret_mm" not in lesions and "feret_mm" in lesions:
            lesions = lesions.rename(columns={"feret_mm": "max_axial_feret_mm"})
    return cases, lesions, "2D"


def case_table(path: Path, configuration_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per case plus a final mean row.

    Columns per size group: ``<g>_total`` lesions, ``<g>_covered`` (>=50 % of the
    lesion predicted), ``<g>_dice`` (mean lesion Dice). In the mean row, counts
    are totals over cases and ``<g>_dice`` is the mean over all lesions of the
    group; ``case_dice`` and ``precision`` are means over cases.
    """
    cases, lesions, kind = _load(path, configuration_id)
    table = cases.sort_values("case_id").reset_index(drop=True)
    info = {"kind": kind, "has_sizes": lesions is not None, "n_cases": len(table)}
    groups = {}
    for key, _, low, high in SIZE_GROUPS:
        if lesions is None:
            for suffix in ("total", "covered", "dice"):
                table[f"{key}_{suffix}"] = np.nan
            continue
        group = lesions[(lesions.max_axial_feret_mm >= low) & (lesions.max_axial_feret_mm < high)].copy()
        group["covered"] = group.coverage >= COVERED_FRACTION
        groups[key] = group
        per_case = group.groupby("case_id").agg(total=("coverage", "size"), covered=("covered", "sum"),
                                                dice=("lesion_dice", "mean"))
        joined = table[["case_id"]].join(per_case, on="case_id")
        table[f"{key}_total"] = joined.total.fillna(0).astype(int)
        table[f"{key}_covered"] = joined.covered.fillna(0).astype(int)
        table[f"{key}_dice"] = joined.dice
    mean = {"case_id": MEAN_LABEL, "case_dice": table.case_dice.mean(), "precision": table.precision.mean()}
    for key, *_ in SIZE_GROUPS:
        if lesions is None:
            mean.update({f"{key}_total": np.nan, f"{key}_covered": np.nan, f"{key}_dice": np.nan})
        else:
            group = groups[key]
            mean.update({f"{key}_total": len(group), f"{key}_covered": int(group.covered.sum()),
                         f"{key}_dice": group.lesion_dice.mean() if len(group) else np.nan})
    table = pd.concat([table.astype({"case_id": object}), pd.DataFrame([mean])], ignore_index=True)
    return table, info
