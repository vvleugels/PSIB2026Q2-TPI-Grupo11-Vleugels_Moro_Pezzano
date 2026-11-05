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


def _scale_intensity(image: np.ndarray) -> np.ndarray:
    if not image.size:
        return image.copy()
    minimum, maximum = image.min(), image.max()
    if minimum == maximum:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)


def normalize_slice_hu(slice_hu: np.ndarray, window: Mapping[str, Any], *, liver_mask=None) -> np.ndarray:
    """Scale to [0, 1] using the supplied window or the range inside the ROI."""
    image = np.asarray(slice_hu, dtype=np.float32)
    mask = None if liver_mask is None else np.asarray(liver_mask, dtype=bool)
    values = image if mask is None else image[mask]
    if window.get("enabled", True):
        normalized = apply_hu_window(values, float(window.get("center", 60)), float(window.get("width", 400)))
    else:
        normalized = _scale_intensity(values)
    if mask is None:
        return normalized
    result = np.zeros_like(image)
    result[mask] = normalized
    return result


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


def denoise_mean(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a 2D arithmetic-mean filter with an odd window size."""
    if size < 1 or size % 2 == 0:
        raise ValueError("Mean filter size must be a positive odd integer")
    return ndimage.uniform_filter(np.asarray(image, dtype=np.float32), size=size).astype(np.float32)


def denoise_bilateral(image: np.ndarray, sigma_color: float = 0.08,
                      sigma_spatial: float = 3.0) -> np.ndarray:
    """Apply edge-preserving bilateral denoising to a [0, 1] image."""
    if not np.isfinite(sigma_color) or sigma_color <= 0:
        raise ValueError("Bilateral sigma_color must be finite and positive")
    if not np.isfinite(sigma_spatial) or sigma_spatial <= 0:
        raise ValueError("Bilateral sigma_spatial must be finite and positive")
    source = np.asarray(image, dtype=np.float32)
    result = restoration.denoise_bilateral(
        source, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None)
    return np.asarray(result, dtype=np.float32)


def denoise(image: np.ndarray, settings: Mapping[str, Any], *, liver_mask=None) -> np.ndarray:
    """Dispatch the configured bilateral, Gaussian, or mean filter.

    Median remains accepted for old saved configuration files, but is no
    longer offered as a GUI or default configuration option.
    """
    method = str(settings.get("method", "gaussian")).lower()
    if liver_mask is not None:
        mask = np.asarray(liver_mask, dtype=bool)
        if not mask.all():
            return _denoise_masked(np.asarray(image, dtype=np.float32), mask, settings, method)
    if method == "none":
        return np.asarray(image, dtype=np.float32).copy()
    if method == "bilateral":
        return denoise_bilateral(
            image, float(settings.get("bilateral_sigma_color", 0.08)),
            float(settings.get("bilateral_sigma_spatial", 3.0)))
    if method == "median":
        return denoise_median(image, int(settings.get("median_size", 3)))
    if method == "gaussian":
        return denoise_gaussian(image, float(settings.get("gaussian_sigma", 0.6)))
    if method == "mean":
        return denoise_mean(image, int(settings.get("mean_size", 3)))
    raise ValueError(f"Unknown denoising method {method!r}; use none, bilateral, gaussian, or mean")


def _denoise_masked(image, mask, settings, method):
    source = np.where(mask, image, 0)
    if method == "none" or not mask.any():
        return source
    if method == "bilateral":
        x0, y0, x1, y1 = liver_bounding_box(mask)
        region = np.s_[x0:x1, y0:y1]
        source[region] = _bilateral_masked(source[region], mask[region],
                                           float(settings.get("bilateral_sigma_color", 0.08)),
                                           float(settings.get("bilateral_sigma_spatial", 3.0)))
        return source
    if method == "median":
        size = int(settings.get("median_size", 3))
        if size < 1 or size % 2 == 0:
            raise ValueError("Median filter size must be a positive odd integer")
        padded = np.pad(np.where(mask, image, np.nan), size // 2, constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        for row in np.flatnonzero(mask.any(axis=1)):
            source[row, mask[row]] = np.nanmedian(windows[row, mask[row]], axis=(1, 2))
        return source
    if method in ("gaussian", "mean"):
        weights = denoise(mask.astype(np.float32), settings)
        filtered = denoise(source, settings)
        return np.divide(filtered, weights, out=source, where=mask)
    return denoise(source, settings)


def _bilateral_masked(image, mask, sigma_color, sigma_spatial):
    if not np.isfinite(sigma_color) or sigma_color <= 0:
        raise ValueError("Bilateral sigma_color must be finite and positive")
    if not np.isfinite(sigma_spatial) or sigma_spatial <= 0:
        raise ValueError("Bilateral sigma_spatial must be finite and positive")
    rows, columns = image.shape
    radius = max(2, int(np.ceil(min(3 * sigma_spatial, max(image.shape)))))
    total = np.zeros_like(image)
    weights = np.zeros_like(image)
    for dx in range(-min(radius, rows - 1), min(radius, rows - 1) + 1):
        for dy in range(-min(radius, columns - 1), min(radius, columns - 1) + 1):
            target = np.s_[max(0, -dx):min(rows, rows-dx), max(0, -dy):min(columns, columns-dy)]
            neighbor = np.s_[max(0, dx):min(rows, rows+dx), max(0, dy):min(columns, columns+dy)]
            delta = image[target] - image[neighbor]
            spatial = (dx / sigma_spatial)**2 + (dy / sigma_spatial)**2
            weight = np.exp(-0.5 * ((delta / sigma_color)**2 + spatial))
            weight *= mask[neighbor]
            weights[target] += weight
            total[target] += weight * image[neighbor]
    return np.divide(total, weights, out=np.zeros_like(image), where=mask)


def prepare_denoised_slice(slice_hu: np.ndarray, cfg: Mapping[str, Any], *, liver_mask=None) -> np.ndarray:
    """Normalize and denoise one full-size HU slice for analysis or segmentation."""
    settings = cfg.get("preprocessing", cfg)
    windowed = normalize_slice_hu(slice_hu, settings.get("hu_window", {}), liver_mask=liver_mask)
    return denoise(windowed, settings.get("denoising", {}), liver_mask=liver_mask)


def _clahe_masked(source, mask, kernel_size, clip_limit):
    if kernel_size < 1 or not np.isfinite(clip_limit) or not 0 <= clip_limit <= 1:
        raise ValueError("CLAHE requiere un tamaño positivo y un límite entre 0 y 1.")
    rows, columns = np.nonzero(mask)
    values = source[mask]
    if kernel_size == 1 or values.min() == values.max():
        return values

    # Anchor contextual tiles to the liver's bounding box and count only ROI pixels.
    coordinates = np.stack((rows - rows.min(), columns - columns.min()))
    tile_shape = coordinates.max(axis=1) // kernel_size + 1
    tile_ids = np.ravel_multi_index(coordinates // kernel_size, tile_shape)
    active, sample_tiles = np.unique(tile_ids, return_inverse=True)
    bins = 256
    levels = np.rint(_scale_intensity(values) * (bins - 1)).astype(np.intp)
    histograms = np.bincount(sample_tiles * bins + levels, minlength=len(active) * bins).reshape(-1, bins)
    counts = histograms.sum(axis=1, keepdims=True)
    if clip_limit > 0:
        cap = np.maximum(1, np.floor(clip_limit * counts))
        histograms = np.minimum(histograms, cap)
        histograms += (counts - histograms.sum(axis=1, keepdims=True)) / bins
    cdfs = np.cumsum(histograms, axis=1) / counts

    # Interpolate at tile centers, renormalizing weights when a tile has no liver pixels.
    lookup = np.full(np.prod(tile_shape), -1, dtype=np.intp)
    lookup[active] = np.arange(len(active))
    centers = np.clip((coordinates + 0.5) / kernel_size - 0.5, 0, tile_shape[:, None] - 1)
    lower = centers.astype(np.intp)
    upper = np.minimum(lower + 1, tile_shape[:, None] - 1)
    fraction = centers - lower
    total = np.zeros(len(values))
    weight_sum = np.zeros(len(values))
    for x, wx in ((lower[0], 1 - fraction[0]), (upper[0], fraction[0])):
        for y, wy in ((lower[1], 1 - fraction[1]), (upper[1], fraction[1])):
            tile = lookup[np.ravel_multi_index((x, y), tile_shape)]
            valid = tile >= 0
            weight = wx[valid] * wy[valid]
            total[valid] += weight * cdfs[tile[valid], levels[valid]]
            weight_sum[valid] += weight
    return _scale_intensity(total / weight_sum)


def enhance(image: np.ndarray, settings: Mapping[str, Any], *, liver_mask=None) -> np.ndarray:
    """Equalize if configured, then adjust contrast around normalized intensity 0.5."""
    source = np.asarray(image, dtype=np.float32)
    mask = None if liver_mask is None else np.asarray(liver_mask, dtype=bool)
    contrast = float(settings.get("contrast", 1.0))
    if not np.isfinite(contrast) or contrast < 0:
        raise ValueError("El contraste debe ser un número finito mayor o igual a cero.")
    method = str(settings.get("method", "clahe")).lower()
    if mask is not None and not mask.any():
        return np.zeros_like(source)
    partial_mask = mask is not None and not mask.all()
    values = source[mask] if partial_mask else source
    if method == "none":
        enhanced = values.copy()
    elif method == "histogram_equalization":
        enhanced = np.asarray(exposure.equalize_hist(values), dtype=np.float32)
    elif method == "clahe":
        clip_limit = float(settings.get("clahe_clip_limit", 0.01))
        kernel_size = int(settings.get("clahe_kernel_size", 8))
        if partial_mask:
            enhanced = _clahe_masked(source, mask, kernel_size, clip_limit)
        else:
            enhanced = np.asarray(exposure.equalize_adapthist(
                source, clip_limit=clip_limit, kernel_size=kernel_size), dtype=np.float32)
    else:
        raise ValueError(f"Unknown enhancement method {method!r}")
    if contrast != 1.0:
        adjusted = (enhanced.astype(np.float64) - 0.5) * contrast + 0.5
        enhanced = np.clip(adjusted, 0, 1).astype(np.float32)
    if not partial_mask:
        return enhanced
    result = np.zeros_like(source)
    result[mask] = enhanced
    return result


def prepare_enhanced_slice(slice_hu, cfg, *, liver_mask=None):
    """Run the shared preprocessing path used by the GUI and classical segmentation."""
    settings = cfg.get("preprocessing", cfg)
    denoised = prepare_denoised_slice(slice_hu, settings, liver_mask=liver_mask)
    return enhance(denoised, settings.get("enhancement", {}), liver_mask=liver_mask)


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
    windowed = normalize_slice_hu(slice_hu, settings.get("hu_window", {}), liver_mask=liver_mask)
    denoised = denoise(windowed, settings.get("denoising", {}), liver_mask=liver_mask)
    enhanced = enhance(denoised, settings.get("enhancement", {}), liver_mask=liver_mask)
    cropped, box = crop_to_liver(enhanced, liver_mask, int(settings.get("crop_margin", 8)))
    return {"windowed": windowed, "denoised": denoised, "enhanced": enhanced,
            "cropped": cropped, "crop_box": box}
