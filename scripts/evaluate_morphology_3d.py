"""Estimate 3D tumors by stacking the 2D segmenter over selected full cases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from psib.config import load_config
from psib.data_io import iter_case_slices
from psib.metrics import dice_score, iou_score
from psib.morphology3d import remove_small_components_3d, tumor_components_3d
from psib.preprocessing import prepare_denoised_slice
from psib.segmentation import _validate_settings, segment_slice


SEGMENTATION_COLUMNS = ("method", "threshold", "k", "direction", "opening", "closing",
                       "fill_holes", "footprint", "radius", "min_area")
MEASURE_COLUMNS = (
    "case_id", "configuration_id", "n_slices", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
    "reference_tumor_voxels", "prediction_raw_voxels", "prediction_clean_voxels",
    "reference_volume_cm3", "prediction_raw_volume_cm3", "prediction_clean_volume_cm3",
    "clean_volume_error_cm3", "clean_volume_relative_error", "dice_3d_raw", "iou_3d_raw",
    "dice_3d_clean", "iou_3d_clean", "reference_components_6", "prediction_components_raw_6",
    "prediction_components_clean_6", "reference_max_axial_feret_mm",
    "prediction_raw_max_axial_feret_mm", "prediction_clean_max_axial_feret_mm",
    "clean_axial_feret_error_mm", "clean_axial_feret_relative_error",
)


def resolve_configuration(run_dir: Path, configuration_id: str) -> tuple[str, dict]:
    configs = pd.read_csv(run_dir / "configurations.csv")
    if configs.empty or "configuration_id" not in configs:
        raise ValueError("La corrida no contiene configuraciones de segmentación.")
    if configuration_id == "best":
        metrics = pd.read_csv(run_dir / "metrics_per_case.csv")
        if metrics.empty or not {"configuration_id", "dice"}.issubset(metrics.columns):
            raise ValueError("No se puede elegir 'best': faltan métricas por caso.")
        summary = metrics.groupby("configuration_id", as_index=False).agg(
            dice_mean=("dice", "mean"), iou_mean=("iou", "mean") if "iou" in metrics else ("dice", "mean"))
        candidates = summary.merge(configs, on="configuration_id", validate="one_to_one")
        winner = candidates.sort_values(["dice_mean", "iou_mean", "configuration_id"],
                                        ascending=[False, False, True]).iloc[0]
        configuration_id = str(winner.configuration_id)
    matched = configs.loc[configs.configuration_id.astype(str) == configuration_id]
    if matched.empty:
        raise ValueError(f"La configuración {configuration_id!r} no está en {run_dir / 'configurations.csv'}.")
    row = matched.iloc[0]
    settings = {key: row[key] for key in SEGMENTATION_COLUMNS if key in row and pd.notna(row[key])}
    return configuration_id, _validate_settings(settings)


def _relative_error(value: float, reference: float) -> float:
    return (value - reference) / reference if reference > 0 else float("nan")


def _volume_cm3(mask: np.ndarray, spacing) -> float:
    return float(np.count_nonzero(mask) * np.prod(spacing) / 1000.0)


def _measure_case(case_id: int, ct_dir: Path, settings: dict, cfg: dict,
                  min_component_voxels: int, output_dir: Path,
                  configuration_id: str) -> tuple[dict, list[dict]]:
    predicted_planes, reference_planes, spacing = [], [], None
    for slice_index, hu, labels, spacing in iter_case_slices(case_id, data_dir=ct_dir):
        liver = labels > 0
        image = prepare_denoised_slice(hu, cfg)
        predicted_planes.append(segment_slice(image, liver, settings))
        reference_planes.append(labels == 2)
        if slice_index % 50 == 0:
            print(f"  Caso {case_id:03d}: corte {slice_index}", flush=True)
    if not predicted_planes or spacing is None:
        raise ValueError(f"El caso {case_id} no contiene cortes axiales.")

    prediction_raw = np.stack(predicted_planes, axis=2)
    reference = np.stack(reference_planes, axis=2)
    prediction_clean = remove_small_components_3d(prediction_raw, min_component_voxels)
    ref_components, _ = tumor_components_3d(np.where(reference, 2, 0).astype(np.uint8), spacing)
    raw_components, _ = tumor_components_3d(np.where(prediction_raw, 2, 0).astype(np.uint8), spacing)
    clean_components, _ = tumor_components_3d(np.where(prediction_clean, 2, 0).astype(np.uint8), spacing)

    ref_volume = _volume_cm3(reference, spacing)
    clean_volume = _volume_cm3(prediction_clean, spacing)
    ref_diameter = max((item["max_axial_feret_mm"] for item in ref_components), default=0.0)
    raw_diameter = max((item["max_axial_feret_mm"] for item in raw_components), default=0.0)
    clean_diameter = max((item["max_axial_feret_mm"] for item in clean_components), default=0.0)
    row = {
        "case_id": case_id, "configuration_id": configuration_id,
        "n_slices": prediction_raw.shape[2],
        "spacing_x_mm": spacing[0], "spacing_y_mm": spacing[1], "spacing_z_mm": spacing[2],
        "reference_tumor_voxels": int(reference.sum()),
        "prediction_raw_voxels": int(prediction_raw.sum()),
        "prediction_clean_voxels": int(prediction_clean.sum()),
        "reference_volume_cm3": ref_volume,
        "prediction_raw_volume_cm3": _volume_cm3(prediction_raw, spacing),
        "prediction_clean_volume_cm3": clean_volume,
        "clean_volume_error_cm3": clean_volume - ref_volume,
        "clean_volume_relative_error": _relative_error(clean_volume, ref_volume),
        "dice_3d_raw": dice_score(prediction_raw, reference),
        "iou_3d_raw": iou_score(prediction_raw, reference),
        "dice_3d_clean": dice_score(prediction_clean, reference),
        "iou_3d_clean": iou_score(prediction_clean, reference),
        "reference_components_6": len(ref_components),
        "prediction_components_raw_6": len(raw_components),
        "prediction_components_clean_6": len(clean_components),
        "reference_max_axial_feret_mm": ref_diameter,
        "prediction_raw_max_axial_feret_mm": raw_diameter,
        "prediction_clean_max_axial_feret_mm": clean_diameter,
        "clean_axial_feret_error_mm": clean_diameter - ref_diameter,
        "clean_axial_feret_relative_error": _relative_error(clean_diameter, ref_diameter),
    }
    np.savez_compressed(output_dir / f"case_{case_id:03d}_masks_3d.npz",
                        reference=reference, prediction_raw=prediction_raw,
                        prediction_clean=prediction_clean)
    lesion_rows = []
    for region, components in (("reference", ref_components),
                               ("prediction_raw", raw_components),
                               ("prediction_clean", clean_components)):
        lesion_rows.extend({"case_id": case_id, "region": region, **component}
                           for component in components)
    del predicted_planes, reference_planes, prediction_raw, reference, prediction_clean
    return row, lesion_rows


def plot_comparison(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    try:
        comparisons = (("reference_volume_cm3", "prediction_clean_volume_cm3", "Tumor volume (cm³)"),
                       ("reference_max_axial_feret_mm", "prediction_clean_max_axial_feret_mm",
                        "Maximum axial Feret (mm)"))
        for ax, (reference, prediction, label) in zip(axes, comparisons):
            x, y = frame[reference].to_numpy(float), frame[prediction].to_numpy(float)
            ax.scatter(x, y, color="#3478b8", alpha=0.85)
            upper = max(float(np.max(x, initial=0)), float(np.max(y, initial=0)))
            ax.plot([0, upper], [0, upper], color="#555555", linestyle="--", linewidth=1)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("symlog", linthresh=1)
            ax.set(xlabel=f"Referencia · {label}", ylabel=f"Predicción · {label}", title=label)
            ax.grid(alpha=0.25)
        fig.suptitle("Escala symlog (umbral 1); identificar casos con case_id en el CSV", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def evaluate(run_dir: Path, data_dir: Path, min_component_voxels: int = 8,
             configuration_id: str = "best", output_name: str = "morphology3d_prediction") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        raise ValueError("output_name solo admite letras ASCII, números, guiones y guiones bajos.")
    if (isinstance(min_component_voxels, bool) or not isinstance(min_component_voxels, int)
            or min_component_voxels < 1):
        raise ValueError("min_component_voxels debe ser un entero positivo.")
    state_path = run_dir / "run.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError(f"La corrida {run_dir.name} no figura como completada.")
    selection_path = run_dir / "selected_cases_slices.csv"
    selection = pd.read_csv(selection_path)
    if selection.empty or "case_id" not in selection:
        raise ValueError("La selección de casos no tiene casos para procesar.")
    configuration_id, settings = resolve_configuration(run_dir, configuration_id)
    cfg = yaml.safe_load((run_dir / "effective_config.yaml").read_text(encoding="utf-8"))
    output = run_dir / output_name
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; use otro --output-name para preservar resultados previos.")
    output.mkdir(parents=True)
    try:
        rows, lesion_rows = [], []
        for case_id in sorted(pd.to_numeric(selection.case_id, errors="raise").astype(int).unique()):
            print(f"Segmentación volumétrica estimada · caso {case_id:03d}", flush=True)
            row, components = _measure_case(int(case_id), data_dir, settings, cfg,
                                            min_component_voxels, output, configuration_id)
            rows.append(row)
            lesion_rows.extend(components)
        per_case = pd.DataFrame(rows, columns=MEASURE_COLUMNS)
        per_case.to_csv(output / "morphology_3d_prediction_per_case.csv", index=False)
        pd.DataFrame(lesion_rows, columns=("case_id", "region", "component_id", "voxel_count",
                  "volume_cm3", "extent_x_mm", "extent_y_mm", "extent_z_mm",
                  "max_feret_3d_mm", "max_axial_feret_mm", "equivalent_diameter_mm",
                  "bbox_diagonal_mm", "bbox_start_x", "bbox_start_y", "bbox_start_z"
                  )).to_csv(output / "morphology_3d_prediction_per_lesion.csv", index=False)
        plot_comparison(per_case, output / "comparison_3d.png")
        completed = {"status": "completed", "configuration_id": configuration_id,
                     "min_component_voxels": min_component_voxels,
                     "n_cases": len(rows), "finished_at": datetime.now(timezone.utc).isoformat(),
                     "interpretation": "Exploratory full-volume stacking of 2D predictions; liver ROI uses reference labels."}
        (output / "run.json").write_text(json.dumps(completed, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        failed = {"status": "failed", "configuration_id": configuration_id,
                  "error": f"{type(exc).__name__}: {exc}",
                  "finished_at": datetime.now(timezone.utc).isoformat()}
        (output / "run.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    print(f"Resultados guardados en {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True,
                        help="ID de una corrida completada de scripts/run_segmentation.py")
    parser.add_argument("--configuration-id", default="best", help="ID cfg_xxx o 'best' por Dice medio")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--min-component-voxels", type=int, default=None,
                        help="Mínimo de vóxeles por componente 3D; por defecto usa config")
    parser.add_argument("--output-name", default="morphology3d_prediction")
    args = parser.parse_args()
    try:
        cfg_path = args.config.resolve()
        cfg = load_config(cfg_path)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
            raise ValueError("run-id solo admite letras ASCII, números, guiones y guiones bajos.")
        run_dir = Path(cfg["_output_dir"]) / "segmentation" / args.run_id
        section = cfg.get("morphology3d_prediction", {})
        min_voxels = args.min_component_voxels if args.min_component_voxels is not None else int(
            section.get("min_component_voxels", 8))
        evaluate(run_dir, Path(cfg["data_dir"]), min_voxels, args.configuration_id, args.output_name)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error en estimación morfológica 3D: {exc}\n")


if __name__ == "__main__":
    main()
