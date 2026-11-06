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
from psib.detection import configured_prediction, validate_settings as validate_detection
from psib.data_io import iter_case_slices
from psib.metrics import dice_score, iou_score
from psib.morphology3d import (filter_components_3d, match_lesions_3d,
                               summarize_lesions_by_size, tumor_components_3d)
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
    "reference_lesions", "detected_lesions", "lesion_detection_rate", "covered_lesions",
    "reference_lesions_ge10mm", "detected_lesions_ge10mm",
    "reference_lesions_ge20mm", "detected_lesions_ge20mm",
    "false_positive_components", "false_positive_volume_cm3", "liver_roi",
)
LESION_COLUMNS = (
    "case_id", "component_id", "voxel_count", "volume_cm3", "extent_x_mm", "extent_y_mm",
    "extent_z_mm", "max_feret_3d_mm", "max_axial_feret_mm", "equivalent_diameter_mm",
    "bbox_diagonal_mm", "bbox_start_x", "bbox_start_y", "bbox_start_z",
    "matched_prediction_components", "matched_prediction_voxels", "intersection_voxels",
    "coverage", "lesion_dice", "lesion_iou", "detected", "covered",
)


def resolve_configuration(run_dir: Path, configuration_id: str) -> tuple[str, dict]:
    configs = pd.read_csv(run_dir / "configurations.csv")
    if configs.empty or "configuration_id" not in configs:
        raise ValueError("La corrida no contiene configuraciones de segmentación.")
    if configuration_id == "best":
        if (run_dir / "folds.json").exists():
            raise ValueError("La validación anidada requiere un ID explícito; no tiene un ganador global.")
        if (run_dir / "best_configurations.csv").exists():
            best = pd.read_csv(run_dir / "best_configurations.csv")
            if "selection_metric" in best:
                return resolve_configuration(run_dir, str(best.iloc[0].configuration_id))
        if "detector_settings" in configs and len(configs) > 1:
            raise ValueError("No hay ganador factible del detector; indique un ID explícito.")
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
    if "pipeline_config" in row and pd.notna(row["pipeline_config"]):
        return configuration_id, {"pipeline_config": json.loads(row["pipeline_config"])}
    settings = {key: row[key] for key in SEGMENTATION_COLUMNS if key in row and pd.notna(row[key])}
    if "detector_settings" in row and pd.notna(row["detector_settings"]):
        return configuration_id, {"detection": validate_detection(json.loads(row["detector_settings"]))}
    return configuration_id, _validate_settings(settings)


def _relative_error(value: float, reference: float) -> float:
    return (value - reference) / reference if reference > 0 else float("nan")


def _volume_cm3(mask: np.ndarray, spacing) -> float:
    return float(np.count_nonzero(mask) * np.prod(spacing) / 1000.0)


def _measure_case(case_id: int, ct_dir: Path, settings: dict, cfg: dict,
                  min_component_voxels: int, output_dir: Path,
                  configuration_id: str, *, min_volume_mm3: float = 0.0, min_slices: int = 1,
                  min_iou: float = 0.0, save_masks: bool = True) -> tuple[dict, list[dict], list[dict]]:
    predicted_planes, reference_planes, spacing = [], [], None
    for slice_index, hu, labels, spacing in iter_case_slices(case_id, data_dir=ct_dir):
        liver = labels > 0
        active_cfg = ({**cfg, "detection": {"enabled": True, **settings["detection"]}} if "detection" in settings
                      else settings.get("pipeline_config", cfg))
        predicted_planes.append(configured_prediction(hu, liver, spacing, active_cfg,
            None if "pipeline_config" in settings else settings)["mask"])
        reference_planes.append(labels == 2)
        if slice_index % 50 == 0:
            print(f"  Caso {case_id:03d}: corte {slice_index}", flush=True)
    if not predicted_planes or spacing is None:
        raise ValueError(f"El caso {case_id} no contiene cortes axiales.")

    prediction_raw = np.stack(predicted_planes, axis=2)
    reference = np.stack(reference_planes, axis=2)
    prediction_clean = (prediction_raw.copy() if "detection" in settings else
                        filter_components_3d(prediction_raw, spacing, min_component_voxels,
                                             min_volume_mm3, min_slices))
    ref_result = tumor_components_3d(np.where(reference, 2, 0).astype(np.uint8), spacing)
    raw_components, _ = tumor_components_3d(np.where(prediction_raw, 2, 0).astype(np.uint8), spacing)
    clean_result = tumor_components_3d(np.where(prediction_clean, 2, 0).astype(np.uint8), spacing)
    ref_components, clean_components = ref_result[0], clean_result[0]
    # Lesions are matched with 26-connectivity (own labelling); descriptive counts stay 6-connected.
    lesions, false_positives = match_lesions_3d(reference, prediction_clean, spacing, min_iou)
    lesions = [{"case_id": case_id, **item} for item in lesions]

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
        "reference_lesions": len(lesions),
        "detected_lesions": sum(item["detected"] for item in lesions),
        "lesion_detection_rate": (sum(item["detected"] for item in lesions) / len(lesions)
                                  if lesions else float("nan")),
        "covered_lesions": sum(item["covered"] for item in lesions),
        "false_positive_components": len(false_positives),
        "false_positive_volume_cm3": float(sum(item["volume_cm3"] for item in false_positives)),
        # The liver ROI that restricts the 2D predictor comes from the reference labels.
        "liver_roi": "reference",
    }
    for threshold in (10, 20):
        group = [item for item in lesions if item["max_axial_feret_mm"] >= threshold]
        row[f"reference_lesions_ge{threshold}mm"] = len(group)
        row[f"detected_lesions_ge{threshold}mm"] = sum(item["detected"] for item in group)
    if save_masks:
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
    return row, lesion_rows, lesions


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
             configuration_id: str = "best", output_name: str = "morphology3d_prediction", *,
             case_ids=None, min_volume_mm3: float = 0.0, min_slices: int = 1,
             min_iou: float = 0.0, save_masks: bool = True) -> Path:
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
    available = sorted(pd.to_numeric(selection.case_id, errors="raise").astype(int).unique())
    if case_ids:
        missing = sorted(set(map(int, case_ids)) - set(available))
        if missing:
            raise ValueError(f"Casos no presentes en la selección de la corrida: {missing}")
        available = sorted(set(map(int, case_ids)))
    configuration_id, settings = resolve_configuration(run_dir, configuration_id)
    cfg = yaml.safe_load((run_dir / "effective_config.yaml").read_text(encoding="utf-8"))
    output = run_dir / output_name
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; use otro --output-name para preservar resultados previos.")
    output.mkdir(parents=True)
    try:
        rows, lesion_rows, matched = [], [], []
        for position, case_id in enumerate(available, start=1):
            print(f"Segmentación volumétrica estimada · caso {case_id:03d} ({position}/{len(available)})",
                  flush=True)
            row, components, lesions = _measure_case(
                int(case_id), data_dir, settings, cfg, min_component_voxels, output, configuration_id,
                min_volume_mm3=min_volume_mm3, min_slices=min_slices, min_iou=min_iou,
                save_masks=save_masks)
            rows.append(row)
            lesion_rows.extend(components)
            matched.extend(lesions)
        per_case = pd.DataFrame(rows, columns=MEASURE_COLUMNS)
        per_case.to_csv(output / "morphology_3d_prediction_per_case.csv", index=False)
        pd.DataFrame(lesion_rows, columns=("case_id", "region", "component_id", "voxel_count",
                  "volume_cm3", "extent_x_mm", "extent_y_mm", "extent_z_mm",
                  "max_feret_3d_mm", "max_axial_feret_mm", "equivalent_diameter_mm",
                  "bbox_diagonal_mm", "bbox_start_x", "bbox_start_y", "bbox_start_z"
                  )).to_csv(output / "morphology_3d_prediction_per_lesion.csv", index=False)
        pd.DataFrame(matched, columns=LESION_COLUMNS).to_csv(
            output / "morphology_3d_prediction_lesions.csv", index=False)
        by_size = pd.DataFrame(summarize_lesions_by_size(matched))
        by_size.to_csv(output / "morphology_3d_prediction_by_size.csv", index=False)
        print(by_size[["size_group", "n_lesions", "detection_rate", "covered_rate", "mean_lesion_dice"]]
              .to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
        plot_comparison(per_case, output / "comparison_3d.png")
        completed = {"status": "completed", "configuration_id": configuration_id,
                     "min_component_voxels": min_component_voxels if "detection" not in settings else None,
                     "postprocessing_3d_applied": "detection" not in settings,
                     "min_component_mm3": min_volume_mm3 if "detection" not in settings else None,
                     "min_slices": min_slices if "detection" not in settings else None,
                     "lesion_min_iou": min_iou, "liver_roi": "reference", "masks_saved": save_masks,
                     "case_ids": [int(case) for case in available],
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
    parser.add_argument("--cases", type=int, nargs="+", default=None,
                        help="Evaluar solo estos casos de la selección (por defecto, todos)")
    parser.add_argument("--min-component-mm3", type=float, default=None,
                        help="Volumen mínimo por componente 3D en mm³; por defecto usa config (0 = sin filtro)")
    parser.add_argument("--min-slices", type=int, default=None,
                        help="Cortes axiales mínimos por componente 3D; por defecto usa config (1 = sin filtro)")
    parser.add_argument("--save-masks", action="store_true",
                        help="Guardar las máscaras 3D completas (.npz) de cada caso")
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
        min_mm3 = args.min_component_mm3 if args.min_component_mm3 is not None else float(
            section.get("min_component_mm3", 0.0))
        min_slices = args.min_slices if args.min_slices is not None else int(section.get("min_slices", 1))
        min_iou = float(cfg.get("lesion_eval", {}).get("min_iou", 0.0))
        evaluate(run_dir, Path(cfg["data_dir"]), min_voxels, args.configuration_id, args.output_name,
                 case_ids=args.cases, min_volume_mm3=min_mm3, min_slices=min_slices,
                 min_iou=min_iou, save_masks=args.save_masks)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Error en estimación morfológica 3D: {exc}\n")


if __name__ == "__main__":
    main()
