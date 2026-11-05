"""Evaluate 2D lesion detections against reference masks for one segmentation run."""
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
from psib.data_io import iter_case_slices
from psib.lesion_eval import match_lesions, match_lesions_one_to_one


SIZE_CLASSES = ("<10", "10-20", ">=20")


def resolve_configuration(run_dir: Path, configuration_id: str) -> str:
    configs_path = run_dir / "configurations.csv"
    configs = pd.read_csv(configs_path)
    if configuration_id == "best":
        if (run_dir / "folds.json").exists():
            raise ValueError("La validación anidada tiene una configuración por fold; indique un ID explícito.")
        best_path = run_dir / "best_configurations.csv"
        if "detector_settings" in configs and len(configs) > 1 and not best_path.exists():
            raise ValueError("No hay ganador factible del detector; indique un ID explícito.")
        candidates = pd.read_csv(best_path) if best_path.is_file() else pd.read_csv(run_dir / "summary.csv")
        winner = candidates.sort_values(["dice_mean", "iou_mean", "configuration_id"],
                                        ascending=[False, False, True]).iloc[0]
        configuration_id = str(winner.configuration_id)
    if configuration_id not in set(configs.configuration_id.astype(str)):
        raise ValueError(f"La configuración {configuration_id!r} no figura en {configs_path}.")
    return configuration_id


def _case_detection_summary(matches: pd.DataFrame, case_ids) -> pd.DataFrame:
    rows = []
    for case_id in sorted(set(map(int, case_ids))):
        group = matches.loc[matches.case_id == case_id] if not matches.empty else matches
        tp = int((group.status == "TP").sum())
        fp = int((group.status == "FP").sum())
        fn = int((group.status == "FN").sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        rows.append({"case_id": int(case_id), "TP": tp, "FP": fp, "FN": fn,
                     "precision": precision, "recall": recall, "F1": f1})
    return pd.DataFrame(rows, columns=("case_id", "TP", "FP", "FN", "precision", "recall", "F1"))


def _detection_summary(per_case: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("precision", "recall", "F1"):
        values = per_case[metric].dropna()
        rows.append({"scope": "case_mean", "size_class": "all", "metric": metric,
                     "n": len(values), "value": float(values.mean()) if len(values) else np.nan})
    for size_class in SIZE_CLASSES:
        subset = matches[(matches.size_class == size_class) & matches.status.isin(("TP", "FN"))]
        tp = int((subset.status == "TP").sum())
        fn = int((subset.status == "FN").sum())
        rows.append({"scope": "lesion", "size_class": size_class, "metric": "recall",
                     "n": tp + fn, "value": tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(rows, columns=("scope", "size_class", "metric", "n", "value"))


def evaluate(run_dir: Path, data_dir: Path, min_iou: float = 0.0,
             configuration_id: str = "best", output_name: str = "lesion_eval") -> Path:
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name debe ser un nombre de carpeta, sin rutas.")
    try:
        run_state = pd.read_json(run_dir / "run.json", typ="series")
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"No se pudo leer el estado de la corrida en {run_dir}.") from exc
    if run_state.get("status") != "completed":
        raise ValueError(f"La corrida {run_dir.name} no figura como completada.")
    configuration_id = resolve_configuration(run_dir, configuration_id)
    output = run_dir / output_name
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; use otro --output-name para conservar el resultado anterior.")
    selection_path = run_dir / "selected_cases_slices.csv"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Falta la selección copiada en {selection_path}.")
    selection = pd.read_csv(selection_path)
    if selection.empty or not {"case_id", "slice_index"}.issubset(selection.columns):
        raise ValueError("La selección debe tener filas y columnas case_id/slice_index.")

    rows = []
    one_to_one_rows = []
    for case_id, selected in selection.groupby("case_id", sort=True):
        indices = sorted(pd.to_numeric(selected.slice_index, errors="raise").astype(int).unique())
        expected = set(indices)
        prediction_dir = run_dir / "predictions" / configuration_id
        predictions = {}
        for slice_index in indices:
            path = prediction_dir / f"case_{int(case_id):03d}_slice_{slice_index:03d}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Falta la máscara de predicción: {path}")
            predictions[slice_index] = np.load(path, allow_pickle=False)
        for slice_index, _, segmentation, spacing in iter_case_slices(
                int(case_id), indices, data_dir, mask_only=True):
            if slice_index not in expected:
                continue
            groups = match_lesions(predictions[slice_index], segmentation == 2,
                                   spacing[:2], min_iou=min_iou)
            for criterion in (.10, .25):
                exact = match_lesions_one_to_one(predictions[slice_index], segmentation == 2,
                                                 spacing[:2], min_iou=criterion)
                one_to_one_rows.extend({"case_id": int(case_id), "slice_index": slice_index,
                                       "min_iou": criterion, **item} for item in exact.to_dict("records"))
            for item in groups.to_dict("records"):
                rows.append({"case_id": int(case_id), "slice_index": slice_index,
                             "min_iou": min_iou, **item})

    match_columns = ("case_id", "slice_index", "min_iou", "group_id", "reference_components",
                     "predicted_components", "status", "size_class", "area_ref_pixels",
                     "area_pred_pixels", "area_ref_mm2", "area_pred_mm2", "relative_area_error",
                     "max_feret_ref_mm", "max_feret_pred_mm", "diameter_error_mm", "group_dice")
    matches = pd.DataFrame(rows, columns=match_columns)
    cases = _case_detection_summary(matches, selection.case_id.unique())
    summary = _detection_summary(cases, matches) if not matches.empty else pd.DataFrame(
        [{"scope": "case_mean", "size_class": "all", "metric": metric, "n": 0, "value": np.nan}
         for metric in ("precision", "recall", "F1")] +
        [{"scope": "lesion", "size_class": size, "metric": "recall", "n": 0, "value": np.nan}
         for size in SIZE_CLASSES], columns=("scope", "size_class", "metric", "n", "value"))

    output.mkdir(parents=True)
    exact = pd.DataFrame(one_to_one_rows, columns=("case_id", "slice_index", "min_iou", "reference_id",
        "predicted_id", "status", "iou", "diameter_ref_mm", "size_class"))
    exact.to_csv(output / "one_to_one_matches.csv", index=False)
    exact_cases = []
    for criterion in (.10, .25):
        table = _case_detection_summary(exact[exact.min_iou == criterion], selection.case_id.unique())
        table["min_iou"] = criterion
        table["n_slices"] = table.case_id.map(selection.groupby("case_id").size())
        table["fp_per_slice"] = table.FP / table.n_slices
        exact_cases.append(table)
    pd.concat(exact_cases).to_csv(output / "one_to_one_per_case.csv", index=False)
    matches.to_csv(output / "lesion_matches.csv", index=False)
    cases.to_csv(output / "detection_per_case.csv", index=False)
    summary.to_csv(output / "detection_summary.csv", index=False)
    _plot_recall(summary, output / "recall_by_size_class.png")
    print(f"Lesion evaluation for {configuration_id} saved to {output}", flush=True)
    return output


def _plot_recall(summary: pd.DataFrame, path: Path) -> None:
    by_size = summary[(summary.scope == "lesion") & (summary.metric == "recall")]
    values = [float(by_size.loc[by_size.size_class == size, "value"].iloc[0])
              if (by_size.size_class == size).any() else np.nan for size in SIZE_CLASSES]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ax.bar(SIZE_CLASSES, values, color=("#4C9F70", "#E3A63B", "#5878B9"))
        ax.set_ylim(0, 1.05)
        ax.set(xlabel="Diámetro de referencia (Feret máximo)", ylabel="Recall por lesión",
               title="Detección por clase de tamaño")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--configuration-id", default="best", help="cfg_xxx o best")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--min-iou", type=float, default=None)
    parser.add_argument("--output-name", default="lesion_eval")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        run_dir = Path(cfg["_output_dir"]) / "segmentation" / args.run_id
        min_iou = float(args.min_iou if args.min_iou is not None else
                        cfg.get("lesion_eval", {}).get("min_iou", 0.0))
        evaluate(run_dir, Path(cfg["data_dir"]), min_iou, args.configuration_id, args.output_name)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error en evaluación por lesión: {exc}\n")


if __name__ == "__main__":
    main()
