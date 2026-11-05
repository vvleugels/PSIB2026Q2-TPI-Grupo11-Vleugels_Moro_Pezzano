"""Run reproducible 2D segmentation experiments on the selected axial slices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from psib.config import load_config
from psib.data_io import iter_case_slices
from psib.edge_analysis import (edge_map, measure_edge_metrics, sharpen_with_laplacian,
                                watershed_refine)
from psib.metrics import dice_score, iou_score, reference_coverage
from psib.preprocessing import normalize_slice_hu, prepare_enhanced_slice
from psib.segmentation import _validate_settings, segment_slice


COUNT_COLUMNS = ["pred_area", "reference_area", "intersection", "union"]


def prepare_image(hu, cfg, *, liver_mask=None):
    """Normalize, denoise, then enhance one full-size slice for analysis."""
    return prepare_enhanced_slice(hu, cfg, liver_mask=liver_mask)


def processed_image_title(cfg):
    """Describe the denoising and equalization stages applied to the panel."""
    preprocessing = cfg.get("preprocessing", {})
    denoising = preprocessing.get("denoising", {})
    enhancement = preprocessing.get("enhancement", {})
    denoising_method = str(denoising.get("method", "gaussian")).lower()
    enhancement_method = str(enhancement.get("method", "clahe")).lower()
    if enhancement_method == "histogram_equalization":
        equalization_label = "histogram equalization"
    elif enhancement_method == "clahe":
        equalization_label = "CLAHE"
    else:
        equalization_label = None
    if equalization_label:
        if denoising_method == "none":
            return f"Equalized (no denoising; {equalization_label})"
        return f"Denoised + {equalization_label}"
    return "No denoising" if denoising_method == "none" else "Denoised"


def build_configurations(cfg, sweep=False):
    """Return complete, validated settings with stable IDs (252 in sweep mode)."""
    window = cfg.get("preprocessing", {}).get("hu_window", {})
    window_enabled = window.get("enabled", True)
    if window_enabled:
        center, width = float(window.get("center", 60)), float(window.get("width", 400))
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            raise ValueError("La ventana HU debe tener centro finito y ancho positivo finito.")
        low = center - width / 2
    variants = []
    if sweep:
        if not window_enabled:
            raise ValueError("El barrido de umbrales en HU requiere una ventana HU activa.")
        thresholds = []
        for hu in (30, 40, 50):
            normalized = (hu - low) / width
            if not 0 <= normalized <= 1:
                raise ValueError(f"El umbral {hu} HU queda fuera de la ventana configurada.")
            thresholds.append({"method": "global", "threshold": normalized})
        thresholds.append({"method": "otsu"})
        thresholds.extend({"method": "statistical", "k": k} for k in (0.5, 1.0, 1.5))
        profiles = ((False, False, "disk", 1), (True, False, "disk", 1),
                    (False, True, "disk", 1), (True, True, "disk", 1),
                    (True, True, "disk", 2), (True, True, "square", 1))
        for threshold in thresholds:
            for opening, closing, footprint, radius in profiles:
                for fill in (False, True):
                    for area in (0, 25, 100):
                        variants.append({**threshold, "opening": opening, "closing": closing,
                                         "footprint": footprint, "radius": radius,
                                         "fill_holes": fill, "min_area": area})
    else:
        variants.append(cfg.get("segmentation", {}))
    records = []
    for i, variant in enumerate(variants):
        settings = _validate_settings(variant)
        records.append({"configuration_id": f"cfg_{i:03d}", **settings,
                        "threshold_hu": (low + width * settings["threshold"]
                                         if window_enabled and settings["method"] == "global" else None)})
    return records



def iter_predictions(image, liver, records):
    """Reuse morphology/labels for adjacent area variants, retaining one 2D mask."""
    previous = None
    for record in records:
        settings = {key: value for key, value in record.items()
                    if key not in ("configuration_id", "threshold_hu")}
        area = settings.pop("min_area")
        key = tuple(sorted(settings.items()))
        if key != previous:
            base = segment_slice(image, liver, {**settings, "min_area": 0})
            labels = None
            previous = key
        if area:
            if labels is None:
                labels, _ = ndimage.label(base, structure=np.ones((3, 3), dtype=bool))
                sizes = np.bincount(labels.ravel())
            keep = sizes >= area
            keep[0] = False
            pred = keep[labels]
        else:
            pred = base
        yield record, pred

def read_selection(path):
    if not path.is_file():
        raise FileNotFoundError("Falta selected_cases_slices.csv. Ejecute scripts/explore_dataset.py "
                                "y luego scripts/select_cases.py con las rutas de esta configuración.")
    frame = pd.read_csv(path)
    if frame.empty or not {"case_id", "slice_index"}.issubset(frame.columns):
        raise ValueError("La selección debe tener filas y columnas case_id y slice_index.")
    for column in ("case_id", "slice_index"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if (not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any()
                or (values % 1 != 0).any() or pd.api.types.is_bool_dtype(values.dtype)):
            raise ValueError(f"{column} debe contener enteros no negativos.")
        frame[column] = values.astype(np.int64)
    if frame.duplicated(["case_id", "slice_index"]).any():
        raise ValueError("La selección contiene pares case_id/slice_index duplicados.")
    return frame


def scores_from_counts(pred_area, reference_area, intersection, union):
    total = pred_area + reference_area
    return (2 * intersection / total if total else 1.0,
            intersection / union if union else 1.0)


def aggregate_metrics(slices):
    """Pool pixel counts within each case; summarize equally weighted cases."""
    cases = slices.groupby(["configuration_id", "case_id"], sort=True).agg(
        n_slices=("slice_index", "size"), **{key: (key, "sum") for key in COUNT_COLUMNS}).reset_index()
    scores = [scores_from_counts(*(getattr(row, key) for key in COUNT_COLUMNS))
              for row in cases.itertuples(index=False)]
    cases["dice"], cases["iou"] = np.asarray(scores).T
    cases["reference_coverage"] = cases.intersection / cases.reference_area.replace(0, np.nan)
    summary = cases.groupby("configuration_id", sort=True).agg(
        n_cases=("case_id", "size"), dice_mean=("dice", "mean"),
        dice_std=("dice", lambda x: x.std(ddof=0)), iou_mean=("iou", "mean"),
        iou_std=("iou", lambda x: x.std(ddof=0)),
        reference_coverage_mean=("reference_coverage", "mean"),
        reference_coverage_n_cases=("reference_coverage", "count")).reset_index()
    return cases, summary


def aggregate_edge_metrics(slices):
    """Pool per-case results for the one selected high-pass operator."""
    count_columns = ["pred_area", "reference_area", "intersection", "union"]
    has_sharpen = "sharpened_pred_area" in slices.columns
    has_watershed = "watershed_pred_area" in slices.columns
    if has_sharpen:
        count_columns.extend(("sharpened_pred_area", "sharpened_intersection", "sharpened_union"))
    if has_watershed:
        count_columns.extend(("watershed_pred_area", "watershed_intersection", "watershed_union"))
    edge_columns = [column for column in slices.columns if column.endswith((
        "_edge_gradient", "_gradient_mean", "_gradient_std"))]
    cases = slices.groupby(["configuration_id", "case_id"], sort=True).agg(
        n_slices=("slice_index", "size"),
        **{column: (column, "sum") for column in count_columns},
        **{column: (column, "mean") for column in edge_columns},
    ).reset_index()
    baseline_scores = [scores_from_counts(row.pred_area, row.reference_area,
                                           row.intersection, row.union)
                       for row in cases.itertuples(index=False)]
    cases["dice_no_sharpen"], cases["iou_no_sharpen"] = np.asarray(baseline_scores).T
    summary_columns = {"dice_no_sharpen": "mean", "iou_no_sharpen": "mean", "n_slices": "sum"}
    if has_sharpen:
        sharpened_scores = [scores_from_counts(row.sharpened_pred_area, row.reference_area,
                                                row.sharpened_intersection, row.sharpened_union)
                            for row in cases.itertuples(index=False)]
        cases["dice_sharpened"], cases["iou_sharpened"] = np.asarray(sharpened_scores).T
        cases["dice_change"] = cases.dice_sharpened - cases.dice_no_sharpen
        summary_columns.update({"dice_sharpened": "mean", "dice_change": "mean",
                                "iou_sharpened": "mean"})
    if has_watershed:
        watershed_scores = [scores_from_counts(row.watershed_pred_area, row.reference_area,
                                                row.watershed_intersection, row.watershed_union)
                            for row in cases.itertuples(index=False)]
        cases["dice_watershed"], cases["iou_watershed"] = np.asarray(watershed_scores).T
        summary_columns.update({"dice_watershed": "mean", "iou_watershed": "mean"})
    summary_columns.update({column: "mean" for column in edge_columns})
    summary = cases.groupby("configuration_id", sort=True).agg(
        n_cases=("case_id", "size"),
        **{f"{column}_{aggregation}": (column, aggregation)
           for column, aggregation in summary_columns.items()},
    ).reset_index()
    return cases, summary


def choose_best(summary, configurations):
    ranked = summary.merge(pd.DataFrame(configurations), on="configuration_id", validate="one_to_one")
    return ranked.sort_values(["dice_mean", "iou_mean", "configuration_id"],
                              ascending=[False, False, True]).drop_duplicates("method").reset_index(drop=True)


def export_slice_comparison(out, configuration_id, case_id, k, original, denoised,
                            operator, response, reference, prediction, *,
                            watershed_labels=None, watershed_prediction=None,
                            sharpened=None, alpha=0.5, processed_title="Denoised",
                            original_title="Original (HU window)"):
    """Save all images for one slice in a composite.

    ``original`` is the normalized CT slice; ``denoised`` is the filtered slice
    used by segmentation.
    Exactly one selected high-pass operator is shown in the figure.
    """
    directory = out / "predictions" / configuration_id
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"case_{case_id:03d}_slice_{k:03d}"
    np.save(directory / f"{stem}.npy", prediction, allow_pickle=False)
    if watershed_prediction is not None:
        np.save(directory / f"{stem}_watershed.npy", watershed_prediction, allow_pickle=False)
    if sharpened is not None:
        np.save(directory / f"{stem}_sharpened.npy", sharpened, allow_pickle=False)

    common = [(original, original_title, "gray", (0, 1)),
              (denoised, processed_title, "gray", (0, 1))]
    if operator == "sobel":
        panels = common + [
            (response, "Sobel gradient", "magma", (0, max(float(np.percentile(response, 99)), 1e-6))),
            (prediction, "Final mask (before watershed)", "gray", (0, 1)),
            (reference, "Tumor reference", "gray", (0, 1)),
            (watershed_prediction, "Watershed prediction", "gray", (0, 1)),
        ]
    elif operator == "prewitt":
        panels = common + [
            (response, "Prewitt gradient", "magma",
             (0, max(float(np.percentile(response, 99)), 1e-6))),
            (reference, "Tumor reference", "gray", (0, 1)),
            (prediction, "Prediction", "gray", (0, 1)),
        ]
    elif operator == "laplace":
        laplace_limit = max(float(np.percentile(np.abs(response), 99)), 1e-6)
        panels = common + [
            (response, "Laplacian response", "coolwarm", (-laplace_limit, laplace_limit)),
            (sharpened, f"Laplacian sharpen · α={alpha:g}", "gray", (0, 1)),
            (reference, "Tumor reference", "gray", (0, 1)),
            (prediction, "Prediction", "gray", (0, 1)),
        ]
    else:
        panels = common + [
            (reference, "Tumor reference", "gray", (0, 1)),
            (prediction, "Prediction", "gray", (0, 1)),
        ]
    panels = [(array, title, cmap, limits) for array, title, cmap, limits in panels if array is not None]
    figure, axes = plt.subplots(2, 3, figsize=(10, 10))
    try:
        for ax, (array, title, cmap, limits) in zip(axes.flat, panels):
            kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
            ax.imshow(array.T, cmap=cmap, origin="lower", **kwargs)
            ax.set_title(title)
            ax.axis("off")
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        details = f"Dice={dice_score(prediction, reference):.4f}; IoU={iou_score(prediction, reference):.4f}"
        coverage = reference_coverage(prediction, reference)
        details += (f"; Ref. coverage={coverage:.1%}" if np.isfinite(coverage)
                    else "; Ref. coverage=N/A (no tumor reference)")
        if watershed_prediction is not None:
            details += f"; watershed Dice={dice_score(watershed_prediction, reference):.4f}"
        figure.suptitle(f"Case {case_id} / slice {k} / {configuration_id} / {operator} — {details}")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(directory / f"{stem}_{operator}.png", dpi=100)
    finally:
        plt.close(figure)


def _metadata(config_path, run_id, sweep):
    dependencies = {name: version(name) for name in
                    ("numpy", "scipy", "scikit-image", "nibabel", "matplotlib", "pandas", "PyYAML")}
    revision, dirty = None, None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"status": "running", "arguments": {"config": str(config_path), "run_id": run_id,
                                                "sweep": sweep},
            "started_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "dependencies": dependencies, "git_revision": revision, "git_dirty": dirty,
            "interpretation": "Exploratory metrics on reference-selected slices, not full volumes."}


def _write_status(out, state):
    temporary = out / "run.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(out / "run.json")


def run_experiment(config_path, run_id, sweep=False):
    """Execute an experiment, leaving explicit failed/completed status on disk."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id solo admite letras ASCII, números, guiones y guiones bajos.")
    cfg = load_config(config_path)
    if cfg.get("detection", {}).get("enabled", False):
        if str(cfg['detection'].get('method', '')).startswith('local_') or cfg['detection'].get('method') == 'hybrid_size':
            if sweep:
                raise ValueError('Para el barrido local use scripts/run_local_search.py; no se reutiliza la grilla histórica.')
            from scripts.run_local_segmentation import run_experiment as run_local
            return run_local(config_path, run_id)
        from scripts.run_detection import run_experiment as run_detection
        return run_detection(config_path, run_id, sweep=sweep)
    out = Path(cfg["_output_dir"]) / "segmentation" / run_id
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()  # Exclusive: never overwrite a previous run, even a failed one.
    state = {"status": "running", "arguments": {"config": str(config_path), "run_id": run_id,
                                                "sweep": sweep}}
    _write_status(out, state)
    context = "preparación"
    try:
        state.update(_metadata(config_path, run_id, sweep))
        _write_status(out, state)
        configurations = build_configurations(cfg, sweep)
        effective = {key: value for key, value in cfg.items() if not key.startswith("_")}
        effective["preprocessing_scope"] = "liver"
        effective["segmentation"] = _validate_settings(cfg.get("segmentation", {}))
        (out / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        source = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
        selected = read_selection(source)
        shutil.copyfile(source, out / source.name)
        pd.DataFrame(configurations).to_csv(out / "configurations.csv", index=False)
        rows = []
        edge_rows = []
        edge_settings = cfg.get("edge_analysis", {})
        edge_operator = str(edge_settings.get("operator", "sobel")).lower()
        if edge_operator not in ("none", "sobel", "prewitt", "laplace"):
            raise ValueError("edge_analysis.operator must be 'none', 'sobel', 'prewitt', or 'laplace'")
        edge_alpha = float(edge_settings.get("sharpen_alpha", 0.5))
        if edge_operator == "laplace" and (not np.isfinite(edge_alpha) or edge_alpha < 0):
            raise ValueError("edge_analysis.sharpen_alpha must be finite and nonnegative")

        def process(active, export_only=False):
            nonlocal context
            for case_id, group in selected.groupby("case_id", sort=True):
                case_id = int(case_id)
                context = f"caso {case_id}, carga"
                indices = [int(k) for k in group.slice_index]
                context = f"caso {case_id}, cortes {indices}"
                for k, hu, labels, _ in iter_case_slices(case_id, indices, cfg["data_dir"]):
                    context = f"caso {case_id}, corte {k}"
                    liver, reference = labels > 0, labels == 2
                    window = cfg.get("preprocessing", {}).get("hu_window", {})
                    original = normalize_slice_hu(hu, window, liver_mask=liver)
                    image = prepare_image(hu, cfg, liver_mask=liver)
                    response = edge_map(image, edge_operator, liver_mask=liver) if edge_operator != "none" else None
                    sharpened = (sharpen_with_laplacian(image, edge_alpha, response=response)
                                 if edge_operator == "laplace" else None)
                    baseline_predictions = iter_predictions(image, liver, active)
                    if sharpened is not None:
                        sharp_predictions = iter_predictions(sharpened, liver, active)
                        def paired_predictions():
                            for (record, pred), (sharp_record, sharp_pred) in zip(
                                    baseline_predictions, sharp_predictions):
                                if record["configuration_id"] != sharp_record["configuration_id"]:
                                    raise RuntimeError("Baseline and sharpened configurations became misaligned")
                                yield record, pred, sharp_pred
                        prediction_pairs = paired_predictions()
                    else:
                        prediction_pairs = ((record, pred, None)
                                            for record, pred in baseline_predictions)
                    for record, pred, sharp_pred in prediction_pairs:
                        cid = record["configuration_id"]
                        context = f"caso {case_id}, corte {k}, configuración {cid}"
                        if edge_operator == "sobel":
                            watershed_pred, watershed_labels = watershed_refine(response, pred, liver)
                        else:
                            watershed_pred = watershed_labels = None
                        if not export_only:
                            rows.append({"configuration_id": cid, "case_id": case_id, "slice_index": k,
                                         "pred_area": int(pred.sum()), "reference_area": int(reference.sum()),
                                         "intersection": int(np.count_nonzero(pred & reference)),
                                         "union": int(np.count_nonzero(pred | reference)),
                                         "dice": dice_score(pred, reference), "iou": iou_score(pred, reference),
                                         "reference_coverage": reference_coverage(pred, reference)})
                            edge_row = {
                                "configuration_id": cid, "case_id": case_id, "slice_index": k,
                                "edge_operator": edge_operator,
                                "pred_area": int(pred.sum()), "reference_area": int(reference.sum()),
                                "intersection": int(np.count_nonzero(pred & reference)),
                                "union": int(np.count_nonzero(pred | reference)),
                                "dice_no_sharpen": dice_score(pred, reference),
                                "iou_no_sharpen": iou_score(pred, reference),
                            }
                            if response is not None:
                                edge_row.update(measure_edge_metrics(
                                    image, liver, reference, pred, edge_operator, response))
                            if sharp_pred is not None:
                                edge_row.update({
                                    "sharpened_pred_area": int(sharp_pred.sum()),
                                    "sharpened_intersection": int(np.count_nonzero(sharp_pred & reference)),
                                    "sharpened_union": int(np.count_nonzero(sharp_pred | reference)),
                                    "dice_sharpened": dice_score(sharp_pred, reference),
                                    "iou_sharpened": iou_score(sharp_pred, reference),
                                })
                            if watershed_pred is not None:
                                edge_row.update({
                                    "watershed_pred_area": int(watershed_pred.sum()),
                                    "watershed_intersection": int(np.count_nonzero(watershed_pred & reference)),
                                    "watershed_union": int(np.count_nonzero(watershed_pred | reference)),
                                    "watershed_edge_gradient": measure_edge_metrics(
                                        image, liver, reference, watershed_pred,
                                        edge_operator, response)["prediction_edge_gradient"],
                                })
                            edge_rows.append(edge_row)
                        if export_only or not sweep:
                            export_slice_comparison(
                                out, cid, case_id, k, original, image, edge_operator, response,
                                reference, pred, watershed_labels=watershed_labels,
                                watershed_prediction=watershed_pred, sharpened=sharpened,
                                alpha=edge_alpha, processed_title=processed_image_title(cfg),
                                original_title=("Original (HU window)" if window.get("enabled", True)
                                                else "Original (liver intensity range)"))
                del hu, labels, image, liver, reference, pred
                print(f"{'Exportado' if export_only else 'Evaluado'} caso {case_id}", flush=True)

        process(configurations)
        context = "agregación y exportación de tablas"
        slices = pd.DataFrame(rows).sort_values(["configuration_id", "case_id", "slice_index"])
        cases, summary = aggregate_metrics(slices)
        for name, frame in (("metrics_per_slice", slices), ("metrics_per_case", cases), ("summary", summary)):
            frame.to_csv(out / f"{name}.csv", index=False)
        edge_slices = pd.DataFrame(edge_rows).sort_values(
            ["configuration_id", "case_id", "slice_index"])
        edge_cases, edge_summary = aggregate_edge_metrics(edge_slices)
        edge_cases["edge_operator"] = edge_operator
        edge_summary["edge_operator"] = edge_operator
        edge_slices.to_csv(out / "edge_metrics_per_slice.csv", index=False)
        edge_cases.to_csv(out / "edge_metrics_per_case.csv", index=False)
        edge_summary.to_csv(out / "edge_metrics_summary.csv", index=False)
        if sweep:
            best = choose_best(summary, configurations)
            best.to_csv(out / "best_configurations.csv", index=False)
            winners = set(best.configuration_id)
            process([record for record in configurations if record["configuration_id"] in winners], export_only=True)
        state.update(status="completed", n_cases=int(selected.case_id.nunique()),
                     n_slices=len(selected), n_configurations=len(configurations),
                     finished_at=datetime.now(timezone.utc).isoformat())
        _write_status(out, state)
    except Exception as exc:
        state.update(status="failed", context=context, error=f"{type(exc).__name__}: {exc}",
                     finished_at=datetime.now(timezone.utc).isoformat())
        _write_status(out, state)
        raise RuntimeError(f"Ejecución fallida ({context}): {exc}. Estado: {out / 'run.json'}") from exc
    print(f"Ejecución completa: {out}", flush=True)
    return out


def main():
    def interrupted(signum, frame):
        raise RuntimeError(f"Proceso interrumpido por señal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    try:
        run_experiment(args.config.resolve(), args.run_id, args.sweep)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
