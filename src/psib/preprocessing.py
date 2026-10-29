"""Pure 2D CT windowing, denoising, enhancement, and liver cropping helpers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import exposure, restoration


def apply_hu_window(slice_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Clip Hounsfield values to a window and scale the result to [0, 1]."""
    if width <= 0:
        raise ValueError("Window width must be positive")
    low, high = center - width / 2.0, center + width / 2.0
    return (np.clip(np.asarray(slice_hu, dtype=np.float32), low, high) - low) / width


def denoise_median(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a 2D median filter with an odd window size."""
    if size < 1 or size % 2 == 0:
        raise ValueError("Median filter size must be a positive odd integer")
    return ndimage.median_filter(np.asarray(image, dtype=np.float32), size=size).astype(np.float32)


def denoise_gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply a 2D Gaussian filter."""
    if sigma < 0:
        raise ValueError("Gaussian sigma must be nonnegative")
    return ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), sigma=sigma).astype(np.float32)


def denoise_bilateral(image: np.ndarray, sigma_color: float = 0.08, sigma_spatial: float = 3.0) -> np.ndarray:
    """Apply edge-preserving bilateral denoising to a [0, 1] image."""
    source = np.asarray(image, dtype=np.float32)
    result = restoration.denoise_bilateral(source, sigma_color=sigma_color,
                                           sigma_spatial=sigma_spatial, channel_axis=None)
    return np.asarray(result, dtype=np.float32)


def denoise(image: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    """Dispatch the configured median, Gaussian, or bilateral filter."""
    method = str(settings.get("method", "median")).lower()
    if method == "median":
        return denoise_median(image, int(settings.get("median_size", 3)))
    if method == "gaussian":
        return denoise_gaussian(image, float(settings.get("gaussian_sigma", 1.0)))
    if method == "bilateral":
        return denoise_bilateral(image, float(settings.get("bilateral_sigma_color", 0.08)),
                                 float(settings.get("bilateral_sigma_spatial", 3.0)))
    raise ValueError(f"Unknown denoising method {method!r}; use median, gaussian, or bilateral")


def enhance(image: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    """Apply optional histogram equalization or CLAHE; 'none' returns a copy."""
    source = np.asarray(image, dtype=np.float32)
    method = str(settings.get("method", "none")).lower()
    if method == "none":
        return source.copy()
    if method == "histogram_equalization":
        return np.asarray(exposure.equalize_hist(source), dtype=np.float32)
    if method == "clahe":
        return np.asarray(exposure.equalize_adapthist(
            source, clip_limit=float(settings.get("clahe_clip_limit", 0.01)),
            kernel_size=int(settings.get("clahe_kernel_size", 8))), dtype=np.float32)
    raise ValueError(f"Unknown enhancement method {method!r}")


def liver_bounding_box(liver_mask: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    """Return clipped ``(x0, y0, x1, y1)`` bounds around a nonempty 2D mask."""
    mask = np.asarray(liver_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("liver_mask must be 2D")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot crop to an empty liver mask")
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return (max(0, int(x0) - margin), max(0, int(y0) - margin),
            min(mask.shape[0], int(x1) + margin), min(mask.shape[1], int(y1) + margin))


def crop_to_liver(slice_image: np.ndarray, liver_mask: np.ndarray, margin: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a 2D image around the liver and return crop plus mapping box."""
    image, mask = np.asarray(slice_image), np.asarray(liver_mask)
    if image.ndim != 2 or image.shape != mask.shape:
        raise ValueError("slice_image and liver_mask must be 2D arrays with matching shapes")
    box = liver_bounding_box(mask, margin)
    x0, y0, x1, y1 = box
    return image[x0:x1, y0:y1].copy(), box


def preprocess_slice(slice_hu: np.ndarray, liver_mask: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Run configured window, denoising, enhancement, and liver crop steps."""
    settings = cfg.get("preprocessing", cfg)
    window = settings.get("hu_window", {})
    windowed = apply_hu_window(slice_hu, float(window.get("center", 60)), float(window.get("width", 400)))
    denoised = denoise(windowed, settings.get("denoising", {}))
    enhanced = enhance(denoised, settings.get("enhancement", {}))
    cropped, box = crop_to_liver(enhanced, liver_mask, int(settings.get("crop_margin", 8)))
    return {"windowed": windowed, "denoised": denoised, "enhanced": enhanced,
            "cropped": cropped, "crop_box": box}
