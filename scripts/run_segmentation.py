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
from psib.metrics import dice_score, iou_score
from psib.preprocessing import apply_hu_window, denoise
from psib.segmentation import _validate_settings, segment_slice


COUNT_COLUMNS = ["pred_area", "reference_area", "intersection", "union"]


def prepare_image(hu, cfg):
    """Compute exactly the full-size denoised stage, without enhancement/crop."""
    prep = cfg.get("preprocessing", {})
    window = prep.get("hu_window", {})
    return denoise(apply_hu_window(hu, window.get("center", 60), window.get("width", 400)),
                   prep.get("denoising", {}))


def build_configurations(cfg, sweep=False):
    """Return complete, validated settings with stable IDs (252 in sweep mode)."""
    window = cfg.get("preprocessing", {}).get("hu_window", {})
    center, width = float(window.get("center", 60)), float(window.get("width", 400))
    if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
        raise ValueError("La ventana HU debe tener centro finito y ancho positivo finito.")
    low = center - width / 2
    variants = []
    if sweep:
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
                                         if settings["method"] == "global" else None)})
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
    summary = cases.groupby("configuration_id", sort=True).agg(
        n_cases=("case_id", "size"), dice_mean=("dice", "mean"),
        dice_std=("dice", lambda x: x.std(ddof=0)), iou_mean=("iou", "mean"),
        iou_std=("iou", lambda x: x.std(ddof=0))).reset_index()
    return cases, summary


def choose_best(summary, configurations):
    ranked = summary.merge(pd.DataFrame(configurations), on="configuration_id", validate="one_to_one")
    return ranked.sort_values(["dice_mean", "iou_mean", "configuration_id"],
                              ascending=[False, False, True]).drop_duplicates("method").reset_index(drop=True)


def export_prediction(out, configuration_id, case_id, k, image, reference, pred):
    directory = out / "predictions" / configuration_id
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"case_{case_id:03d}_slice_{k:03d}"
    np.save(directory / f"{stem}.npy", pred, allow_pickle=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    try:
        for ax, array, title in zip(axes, (image, reference, pred),
                                    ("Denoised", "Tumor reference", "Prediction")):
            ax.imshow(array.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Case {case_id} / slice {k} / {configuration_id} — "
                     f"Dice={dice_score(pred, reference):.4f}; IoU={iou_score(pred, reference):.4f}")
        fig.tight_layout()
        fig.savefig(directory / f"{stem}.png", dpi=100)
    finally:
        plt.close(fig)


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
        effective["segmentation"] = _validate_settings(cfg.get("segmentation", {}))
        (out / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        source = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
        selected = read_selection(source)
        shutil.copyfile(source, out / source.name)
        pd.DataFrame(configurations).to_csv(out / "configurations.csv", index=False)
        rows = []

        def process(active, export_only=False):
            nonlocal context
            for case_id, group in selected.groupby("case_id", sort=True):
                case_id = int(case_id)
                context = f"caso {case_id}, carga"
                indices = [int(k) for k in group.slice_index]
                context = f"caso {case_id}, cortes {indices}"
                for k, hu, labels, _ in iter_case_slices(case_id, indices, cfg["data_dir"]):
                    context = f"caso {case_id}, corte {k}"
                    image = prepare_image(hu, cfg)
                    liver, reference = labels > 0, labels == 2
                    for record, pred in iter_predictions(image, liver, active):
                        cid = record["configuration_id"]
                        context = f"caso {case_id}, corte {k}, configuración {cid}"
                        if not export_only:
                            rows.append({"configuration_id": cid, "case_id": case_id, "slice_index": k,
                                         "pred_area": int(pred.sum()), "reference_area": int(reference.sum()),
                                         "intersection": int(np.count_nonzero(pred & reference)),
                                         "union": int(np.count_nonzero(pred | reference)),
                                         "dice": dice_score(pred, reference), "iou": iou_score(pred, reference)})
                        if export_only or not sweep:
                            export_prediction(out, cid, case_id, k, image, reference, pred)
                del hu, labels, image, liver, reference, pred
                print(f"{'Exportado' if export_only else 'Evaluado'} caso {case_id}", flush=True)

        process(configurations)
        context = "agregación y exportación de tablas"
        slices = pd.DataFrame(rows).sort_values(["configuration_id", "case_id", "slice_index"])
        cases, summary = aggregate_metrics(slices)
        for name, frame in (("metrics_per_slice", slices), ("metrics_per_case", cases), ("summary", summary)):
            frame.to_csv(out / f"{name}.csv", index=False)
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
