"""Compare lossless and lossy compression on selected LiTS axial slices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
import yaml

from psib.compression import (entropy_bits_per_pixel, huffman_code_table,
                              jpeg_like_dct, row_rle_size_bytes)
from psib.config import load_config
from psib.data_io import iter_case_slices
from psib.features import extract_texture_features
from psib.metrics import dice_score, iou_score
from psib.preprocessing import prepare_denoised_slice
from psib.segmentation import segment_slice


BOOKKEEPING_TEXTURE_KEYS = {"pair_count", "n_offsets"}
DEFAULT_QUALITIES = (95, 75, 50, 25, 10)


def quantize_8bit(image: np.ndarray) -> np.ndarray:
    """Map normalized floating-point pixels to 8-bit using nearest integer."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("Input image must be a finite 2D array in [0, 1]")
    return np.rint(255 * array).astype(np.uint8)


def _case_statistics(image: np.ndarray, liver: np.ndarray, settings: dict) -> None:
    """Fail clearly until an importable case-scope statistics API is available."""
    if settings.get("stats_scope", "slice") != "case":
        return
    try:
        module = importlib.import_module("psib.case_stats")
    except ImportError as exc:
        raise RuntimeError("segmentation.stats_scope='case' requires the psib.case_stats module, "
                           "which is not installed/importable") from exc
    calculator = getattr(module, "compute_case_stats", None)
    if calculator is None:
        raise RuntimeError("psib.case_stats is importable but does not expose compute_case_stats; "
                           "the case-scope API must be integrated before this experiment can run")
    # Keep this call explicit and fail rather than silently substituting per-slice statistics.
    calculator(image=image, liver_mask=liver)
    raise RuntimeError("Case statistics were computed, but segment_slice has no public parameter for "
                       "applying case-level statistics. Integrate that API before comparing results.")


def _quality_values(cfg: dict) -> list[int]:
    values = cfg.get("compression", {}).get("qualities", DEFAULT_QUALITIES)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("compression.qualities must be a nonempty list of integer qualities")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= 100:
            raise ValueError("Each compression quality must be an integer in [1, 100]")
        if int(value) in result:
            raise ValueError("compression.qualities must not contain duplicates")
        result.append(int(value))
    return result


def _metrics(image_u8, original_u8):
    difference = image_u8.astype(np.float64) - original_u8.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    ssim = float(structural_similarity(original_u8, image_u8, data_range=255))
    return psnr, ssim


def _encode_png(image_u8):
    ok, encoded = cv2.imencode(".png", image_u8, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("OpenCV failed to encode the PNG lossless condition")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None or not np.array_equal(decoded, image_u8):
        raise RuntimeError("PNG codec failed exact 8-bit round trip")
    return decoded, int(encoded.nbytes)


def _texture_values(image, reference, region_masks):
    values = {}
    for region, mask in region_masks.items():
        features = extract_texture_features(image, mask)
        values.update({f"texture_{region}_{key}": float(value)
                       for key, value in features.items() if key not in BOOKKEEPING_TEXTURE_KEYS})
    return values


def _condition_result(condition, quality, image_u8, original_u8, reference, liver,
                      seg_settings, compressed_bytes, texture_keys, extra=None):
    reconstructed = image_u8.astype(np.float32) / 255.0
    _case_statistics(reconstructed, liver, seg_settings)
    prediction_settings = {key: value for key, value in seg_settings.items() if key != "stats_scope"}
    prediction = segment_slice(reconstructed, liver, prediction_settings)
    psnr, ssim = _metrics(image_u8, original_u8)
    row = {
        "condition": condition, "quality": quality,
        "compressed_size_bytes": int(compressed_bytes),
        "compression_ratio": (original_u8.size * 8) / (compressed_bytes * 8) if compressed_bytes else float("inf"),
        "relative_redundancy": 1 - compressed_bytes / original_u8.size if original_u8.size else float("nan"),
        "bits_per_pixel": compressed_bytes * 8 / original_u8.size,
        "psnr_db": psnr, "ssim": ssim,
        "dice": dice_score(prediction, reference), "iou": iou_score(prediction, reference),
        "pred_area_px": int(prediction.sum()), "reference_area_px": int(reference.sum()),
        **(extra or {}),
    }
    regions = {"tumor": reference, "healthy_liver": (liver & ~reference)}
    row.update(_texture_values(reconstructed, reference, regions))
    return row, {key: row.get(key, float("nan")) for key in texture_keys}, prediction


def process_slice(case_id, k, hu, labels, spacing, cfg, qualities):
    """Evaluate one selected slice, returning dynamic-property rows and images."""
    prepared = prepare_denoised_slice(hu, cfg)
    control = quantize_8bit(prepared)
    reference, liver = labels == 2, labels > 0
    seg_settings = dict(cfg.get("segmentation", {}))
    regions = {"tumor": reference, "healthy_liver": (labels == 1)}
    control_texture = _texture_values(control.astype(np.float32) / 255, reference, regions)
    texture_keys = list(control_texture)
    conditions = [("uncompressed_8bit", None, control, control.size, {})]
    png, png_bytes = _encode_png(control)
    conditions.append(("png_lossless", None, png, png_bytes, {"entropy_bits_per_pixel": entropy_bits_per_pixel(control)}))
    frequencies = Counter(control.ravel().tolist())
    table = huffman_code_table(frequencies)
    huffman_bits = sum(frequency * len(table[symbol]) for symbol, frequency in frequencies.items())
    conditions.append(("huffman_estimate", None, control, max(1, int(np.ceil(huffman_bits / 8))),
                       {"entropy_bits_per_pixel": entropy_bits_per_pixel(control),
                        "huffman_average_bits_per_pixel": huffman_bits / control.size,
                        "huffman_symbol_count": len(table),
                        "huffman_code_table_json": json.dumps(table, sort_keys=True)}))
    rle_bytes = row_rle_size_bytes(control)
    conditions.append(("row_rle_estimate", None, control, max(1, rle_bytes),
                       {"entropy_bits_per_pixel": entropy_bits_per_pixel(control), "rle_estimated_bytes": rle_bytes}))
    jpeg_images = {}
    for quality in qualities:
        restored, estimated_bits = jpeg_like_dct(control, quality)
        ok, encoded = cv2.imencode(".jpg", control, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"OpenCV failed to encode JPEG quality {quality}")
        jpeg_bytes = int(encoded.nbytes)
        jpeg_images[quality] = restored
        conditions.append((f"jpeg_q{quality}", quality, restored, jpeg_bytes,
                           {"jpeg_estimated_bytes": int(np.ceil(estimated_bits / 8)),
                            "jpeg_actual_bytes": jpeg_bytes,
                            "entropy_bits_per_pixel": entropy_bits_per_pixel(control)}))

    rows, predictions, images = [], {}, {"control": control, **{f"jpeg_q{q}": im for q, im in jpeg_images.items()}}
    control_row = None
    for condition, quality, image, size, extra in conditions:
        row, texture, prediction = _condition_result(condition, quality, image, control, reference,
                                                     liver, seg_settings, size, texture_keys, extra)
        row.update(case_id=int(case_id), slice_index=int(k), spacing_x_mm=spacing[0], spacing_y_mm=spacing[1],
                   tumor_area_px=int(reference.sum()), liver_area_px=int(liver.sum()))
        rows.append(row)
        predictions[condition] = prediction
        if condition == "uncompressed_8bit":
            control_row = row
    if control_row is not None:
        for row in rows:
            for key in texture_keys:
                baseline = control_row.get(key, float("nan"))
                current = row.get(key, float("nan"))
                relative = ((current - baseline) / abs(baseline)
                            if np.isfinite(current) and np.isfinite(baseline) and baseline != 0 else float("nan"))
                row[f"{key}_relative_change"] = relative
    return rows, images, reference, predictions


def _save_figures(out, per_slice, example):
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    lossy = per_slice[per_slice.quality.notna()]
    for metric, title, filename in (("compression_ratio", "Compression ratio vs JPEG quality", "compression_ratio_vs_quality.png"),
                                    ("dice", "Tumor Dice vs JPEG quality", "dice_vs_quality.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for case_id, group in lossy.groupby("case_id"):
            means = group.groupby("quality")[metric].mean().sort_index(ascending=False)
            ax.plot(means.index, means.values, marker="o", alpha=.6, label=f"Case {case_id}")
        if lossy.case_id.nunique() <= 10:
            ax.legend(fontsize=7, ncol=2)
        ax.set(xlabel="JPEG quality", ylabel=metric.replace("_", " "), title=title)
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
    change_columns = [column for column in per_slice if column.startswith("texture_") and column.endswith("_relative_change")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in change_columns:
        means = lossy.groupby("quality")[column].mean().sort_index(ascending=False)
        ax.plot(means.index, means.values, marker="o", label=column.removesuffix("_relative_change"))
    if change_columns:
        ax.legend(fontsize=7, ncol=2)
    ax.set(xlabel="JPEG quality", ylabel="Relative change from 8-bit control", title="Texture change vs JPEG quality")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(figures / "texture_relative_change_vs_quality.png", dpi=160)
    plt.close(fig)
    control, restored, reference = example
    qualities = sorted(restored, reverse=True)
    panels = [("8-bit control", control)]
    panels.extend((f"JPEG q{q}", restored[q]) for q in qualities)
    panels.extend((f"|q{q} − control|", np.abs(restored[q].astype(np.int16) - control.astype(np.int16)))
                  for q in qualities)
    columns = 4
    rows_count = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(12, rows_count * 3))
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image.T, cmap="gray", origin="lower", vmin=0, vmax=255)
        ax.contour(reference.T, levels=[.5], colors="#ff6b35", linewidths=.6, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "compression_example_grid.png", dpi=160)
    plt.close(fig)


def run_experiment(config_path: Path, run_id: str | None = None) -> Path:
    cfg = load_config(config_path)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id must contain only ASCII letters, digits, hyphens or underscores")
    out = Path(cfg["_output_dir"]) / "compression" / run_id
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    started = time.perf_counter()
    state = {"status": "running", "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat()}
    (out / "run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        qualities = _quality_values(cfg)
        selection_path = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
        if not selection_path.is_file():
            raise FileNotFoundError("Missing selected_cases_slices.csv; run the slice-selection workflow first")
        selection = pd.read_csv(selection_path)
        if selection.empty or not {"case_id", "slice_index"}.issubset(selection.columns):
            raise ValueError("Selection CSV must contain case_id and slice_index rows")
        selection.to_csv(out / "selected_cases_slices.csv", index=False)
        effective = {key: value for key, value in cfg.items() if not key.startswith("_")}
        effective.setdefault("compression", {})["qualities"] = qualities
        (out / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        rows, example = [], None
        selected_cases = selection.groupby("case_id", sort=True)
        total_slices = len(selection)
        for case_id, group in selected_cases:
            indices = group.slice_index.astype(int).tolist()
            row_by_index = {int(row.slice_index): row._asdict() for row in group.itertuples(index=False)}
            for k, hu, labels, spacing in iter_case_slices(int(case_id), indices, cfg["data_dir"]):
                slice_rows, images, reference, _predictions = process_slice(int(case_id), k, hu, labels,
                                                                              spacing, cfg, qualities)
                rows.extend(slice_rows)
                if example is None and np.any(reference):
                    example = (images["control"],
                               {q: images[f"jpeg_q{q}"] for q in (75, 25, 10) if f"jpeg_q{q}" in images},
                               reference)
            del hu, labels
            print(f"Compression: evaluated case {case_id}", flush=True)
            elapsed = time.perf_counter() - started
            processed = len({(r["case_id"], r["slice_index"]) for r in rows})
            estimate = elapsed / max(1, processed) * total_slices
            print(f"Estimated total: {estimate:.1f} s; elapsed: {elapsed:.1f} s", flush=True)
        if not rows:
            raise ValueError("No selected slices were processed")
        per_slice = pd.DataFrame(rows)
        per_slice.to_csv(out / "per_slice.csv", index=False)
        numeric = [c for c in per_slice.select_dtypes(include=[np.number]).columns
                   if c not in ("case_id", "slice_index", "quality", "spacing_x_mm", "spacing_y_mm")]
        per_case = per_slice.groupby(["case_id", "condition", "quality"], dropna=False)[numeric].mean().reset_index()
        per_case.to_csv(out / "per_case.csv", index=False)
        summary = per_slice.groupby(["condition", "quality"], dropna=False)[numeric].agg(["count", "mean", "std"])
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        summary.reset_index().to_csv(out / "summary.csv", index=False)
        if example is None:
            first = per_slice.iloc[0]
            case_id, k = int(first.case_id), int(first.slice_index)
            for k0, hu, labels, _ in iter_case_slices(case_id, [k], cfg["data_dir"]):
                _, images, reference, _ = process_slice(case_id, k0, hu, labels, _, cfg, qualities)
                example = (images["control"], {q: images[f"jpeg_q{q}"] for q in (75, 25, 10)
                                                if f"jpeg_q{q}" in images}, reference)
                break
        _save_figures(out, per_slice, example)
        state.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.perf_counter() - started, n_slices=total_slices,
                     n_cases=int(selection.case_id.nunique()))
        (out / "run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                     finished_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.perf_counter() - started)
        (out / "run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise
    print(f"Compression experiment complete: {out}; measured {state['elapsed_seconds']:.1f} s", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_experiment(args.config.resolve(), args.run_id)


if __name__ == "__main__":
    main()
