"""Simulate tomographic acquisition and compare classical reconstructions on LiTS slices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
import yaml

from psib.config import load_config
from psib.data_io import iter_case_slices
from psib.metrics import dice_score, iou_score
from psib.preprocessing import apply_hu_window, liver_bounding_box, prepare_denoised_slice
from psib.reconstruction import reconstruct_liver_roi, psnr
from psib.segmentation import segment_slice
from psib.detection import configured_prediction


DEFAULT_ANGLES = (180, 60, 20)
DEFAULT_ROLES = ("largest", "small", "tumor_free")


def _settings(cfg):
    settings = cfg.get("reconstruction", {})
    angles = settings.get("angles", DEFAULT_ANGLES)
    if (not isinstance(angles, (list, tuple)) or not angles
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in angles)):
        raise ValueError("reconstruction.angles must be a nonempty list of positive integers")
    if len(set(angles)) != len(angles):
        raise ValueError("reconstruction.angles must not contain duplicates")
    downsample = settings.get("downsample", 1)
    iterations = settings.get("sart_iterations", 3)
    relaxation = settings.get("sart_relaxation", 0.15)
    margin = settings.get("margin", 8)
    for name, value, minimum in (("downsample", downsample, 1), ("sart_iterations", iterations, 1),
                                 ("margin", margin, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"reconstruction.{name} must be an integer >= {minimum}")
    if not np.isfinite(relaxation) or relaxation <= 0:
        raise ValueError("reconstruction.sart_relaxation must be positive and finite")
    cases_value = settings.get("cases", settings.get("n_cases", 3))
    case_ids = cases_value if isinstance(cases_value, list) else None
    n_cases = cases_value if isinstance(cases_value, int) and not isinstance(cases_value, bool) else 3
    if case_ids is not None and (not case_ids or any(
            isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 0 for case_id in case_ids)):
        raise ValueError("reconstruction.cases list must contain nonnegative case IDs")
    if case_ids is None and n_cases < 1:
        raise ValueError("reconstruction.cases must be a positive count or explicit case ID list")
    slices_per_case = settings.get("slices_per_case", 3)
    if isinstance(slices_per_case, bool) or not isinstance(slices_per_case, int) or slices_per_case < 1:
        raise ValueError("reconstruction.slices_per_case must be a positive integer")
    return {"angles": list(angles), "downsample": downsample, "sart_iterations": iterations,
            "sart_relaxation": float(relaxation), "margin": margin,
            "n_cases": int(n_cases),
            "slices_per_case": slices_per_case, "case_ids": case_ids}


def _case_scope_guard(image, liver, segmentation_settings):
    if segmentation_settings.get("stats_scope", "slice") != "case":
        return
    try:
        module = importlib.import_module("psib.case_stats")
    except ImportError as exc:
        raise RuntimeError("segmentation.stats_scope='case' requires psib.case_stats, which is not importable") from exc
    calculator = getattr(module, "compute_case_stats", None)
    if calculator is None:
        raise RuntimeError("psib.case_stats lacks compute_case_stats; case-scope segmentation cannot run")
    calculator(image=image, liver_mask=liver)
    raise RuntimeError("The current segment_slice API cannot apply case-level statistics. "
                       "Integrate the case_stats return value before this comparison.")


def _choose_slices(selection, settings):
    if "selection_rank" in selection:
        cases = selection.sort_values(["selection_rank", "case_id"]).case_id.drop_duplicates().tolist()
    else:
        cases = sorted(selection.case_id.unique().tolist())
    chosen_cases = settings["case_ids"] if settings["case_ids"] is not None else cases[:settings["n_cases"]]
    selected = selection[selection.case_id.isin(chosen_cases)]
    chosen_rows = []
    roles_available = "slice_role" in selected.columns
    for case_id, group in selected.groupby("case_id", sort=False):
        group = group.sort_values("slice_index")
        case_rows = []
        if roles_available:
            for role in DEFAULT_ROLES:
                matches = group[group.slice_role == role]
                if not matches.empty:
                    case_rows.append(matches.iloc[0])
        for _, row in group.iterrows():
            if len(case_rows) >= settings["slices_per_case"]:
                break
            if not any(int(row.slice_index) == int(existing.slice_index) for existing in case_rows):
                case_rows.append(row)
        if roles_available and len(case_rows) < settings["slices_per_case"]:
            warnings.warn(f"Case {case_id}: only {len(case_rows)} requested reconstruction-role slices were available",
                          RuntimeWarning)
        chosen_rows.extend(case_rows[:settings["slices_per_case"]])
    if not chosen_rows:
        raise ValueError("No slices remain after applying the reconstruction subset")
    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def _roi_similarity(reference_hu, reconstructed_hu, box, hu_window):
    x0, y0, x1, y1 = box
    center, width = hu_window
    first = apply_hu_window(reference_hu[x0:x1, y0:y1], center, width)
    second = apply_hu_window(reconstructed_hu[x0:x1, y0:y1], center, width)
    score = psnr(first, second)
    min_dim = min(first.shape)
    win_size = min(7, min_dim if min_dim % 2 else min_dim - 1)
    similarity = (float(structural_similarity(first, second, data_range=1.0, win_size=win_size))
                  if win_size >= 3 else float("nan"))
    return score, similarity


def _save_figures(out, per_slice, example):
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(example["sinogram"], cmap="gray", aspect="auto", origin="lower",
              extent=(example["theta"][0], example["theta"][-1], 0, example["sinogram"].shape[0]))
    ax.set(xlabel="Projection angle (degrees)", ylabel="Detector bin", title="Simulated sinogram")
    fig.tight_layout()
    fig.savefig(figures / "sinogram.png", dpi=160)
    plt.close(fig)

    panels = [("Original", example["original"])]
    for angle in sorted(example["reconstructions"]):
        for method in ("sbp", "fbp_ramp", "fbp_hann", "sart"):
            panels.append((f"{method}, {angle} angles", example["reconstructions"][angle][method]))
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 3))
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "reconstruction_grid.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = per_slice.groupby(["method", "angle_count"]).dice.mean().reset_index()
    for method, group in means.groupby("method"):
        group = group.sort_values("angle_count")
        ax.plot(group.angle_count, group.dice, marker="o", label=method)
    ax.set(xlabel="Number of projection angles", ylabel="Mean Dice", title="Tumor Dice vs projection count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "dice_vs_angles.png", dpi=160)
    plt.close(fig)


def run_experiment(config_path: Path, run_id: str | None = None) -> Path:
    cfg = load_config(config_path)
    settings = _settings(cfg)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id must contain only ASCII letters, digits, hyphens or underscores")
    out = Path(cfg["_output_dir"]) / "reconstruction" / run_id
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    start = time.perf_counter()
    state = {"status": "running", "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
             "interpretation": "Simulation of acquisition and reconstruction; LiTS images are pre-reconstructed CTs."}
    (out / "run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        selection_path = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
        if not selection_path.is_file():
            raise FileNotFoundError("Missing selected_cases_slices.csv; run the slice-selection workflow first")
        selection = pd.read_csv(selection_path)
        if selection.empty or not {"case_id", "slice_index"}.issubset(selection.columns):
            raise ValueError("Selection CSV must contain case_id and slice_index rows")
        selected = _choose_slices(selection, settings)
        selected.to_csv(out / "selected_cases_slices.csv", index=False)
        effective = {key: value for key, value in cfg.items() if not key.startswith("_")}
        effective["reconstruction"] = {**cfg.get("reconstruction", {}), **settings}
        (out / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        seg_settings = dict(cfg.get("segmentation", {}))
        window = cfg.get("preprocessing", {}).get("hu_window", {})
        center, width = float(window.get("center", 60)), float(window.get("width", 400))
        if width <= 0:
            raise ValueError("preprocessing.hu_window.width must be positive")
        rows, example = [], None
        expected = len(selected)
        processed = 0
        for case_id, group in selected.groupby("case_id", sort=False):
            indices = group.slice_index.astype(int).tolist()
            role_by_index = {int(row.slice_index): getattr(row, "slice_role", "unspecified")
                             for row in group.itertuples(index=False)}
            for k, hu, labels, _spacing in iter_case_slices(int(case_id), indices, cfg["data_dir"]):
                liver, reference = labels > 0, labels == 2
                original_prepared = prepare_denoised_slice(hu, cfg)
                _case_scope_guard(original_prepared, liver, seg_settings)
                original_settings = {key: value for key, value in seg_settings.items() if key != "stats_scope"}
                baseline_prediction = (configured_prediction(hu, liver, _spacing, cfg)["mask"] if cfg.get("detection", {}).get("enabled")
                                       else segment_slice(original_prepared, liver, original_settings))
                baseline_dice = dice_score(baseline_prediction, reference)
                all_recons, sinogram, theta, box = {}, None, None, None
                for angle in settings["angles"]:
                    angle_methods, angle_box, angle_sinogram, angle_theta = reconstruct_liver_roi(
                        hu, liver, margin=settings["margin"], angle_count=angle,
                        downsample=settings["downsample"], sart_iterations=settings["sart_iterations"],
                        sart_relaxation=settings["sart_relaxation"])
                    all_recons[angle] = angle_methods
                    if sinogram is None:
                        sinogram, theta, box = angle_sinogram, angle_theta, angle_box
                x0, y0, x1, y1 = box
                baseline_scale = apply_hu_window(hu, center, width)
                for angle in settings["angles"]:
                    for method, reconstructed_roi in all_recons[angle].items():
                        reconstructed_hu = hu.astype(np.float64, copy=True)
                        reconstructed_hu[x0:x1, y0:y1] = reconstructed_roi
                        reconstructed_prepared = prepare_denoised_slice(reconstructed_hu, cfg)
                        prediction = (configured_prediction(reconstructed_hu, liver, _spacing, cfg)["mask"] if cfg.get("detection", {}).get("enabled")
                                      else segment_slice(reconstructed_prepared, liver, original_settings))
                        score, similarity = _roi_similarity(hu, reconstructed_hu, box, (center, width))
                        rows.append({"case_id": int(case_id), "slice_index": int(k),
                                     "slice_role": role_by_index[int(k)], "method": method,
                                     "angle_count": int(angle), "psnr_db": score, "ssim": similarity,
                                     "dice": dice_score(prediction, reference),
                                     "iou": iou_score(prediction, reference), "baseline_dice": baseline_dice,
                                     "delta_dice": dice_score(prediction, reference) - baseline_dice,
                                     "tumor_area_px": int(reference.sum()), "liver_area_px": int(liver.sum())})
                if example is None:
                    example = {"original": baseline_scale[x0:x1, y0:y1], "sinogram": sinogram,
                               "theta": theta,
                               "reconstructions": {
                                   angle: {method: apply_hu_window(image, center, width)
                                           for method, image in recons.items()}
                                   for angle, recons in all_recons.items()}}
                processed += 1
                elapsed = time.perf_counter() - start
                print(f"Estimated total: {elapsed / processed * expected:.1f} s; elapsed: {elapsed:.1f} s",
                      flush=True)
            del hu, labels
        per_slice = pd.DataFrame(rows)
        per_slice.to_csv(out / "per_slice.csv", index=False)
        numeric = ["psnr_db", "ssim", "dice", "iou", "baseline_dice", "delta_dice",
                   "tumor_area_px", "liver_area_px"]
        per_case = per_slice.groupby(["case_id", "method", "angle_count"], sort=True)[numeric].mean().reset_index()
        per_case.to_csv(out / "per_case.csv", index=False)
        summary = per_case.groupby(["method", "angle_count"], sort=True)[numeric].agg(["count", "mean", "std"])
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        summary.reset_index().to_csv(out / "summary.csv", index=False)
        _save_figures(out, per_slice, example)
        elapsed = time.perf_counter() - start
        state.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=elapsed, n_cases=int(per_slice.case_id.nunique()),
                     n_slices=int(per_slice[["case_id", "slice_index"]].drop_duplicates().shape[0]),
                     methods=sorted(per_slice.method.unique()), angle_counts=settings["angles"])
        (out / "run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                     finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start)
        (out / "run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise
    print(f"Reconstruction simulation complete: {out}; measured {elapsed:.1f} s", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_experiment(args.config.resolve(), args.run_id)


if __name__ == "__main__":
    main()
