"""Pure 2D threshold segmentation inside a known liver region."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import disk


_DEFAULTS = {
    "method": "global", "threshold": 0.45, "k": 1.0, "direction": "lower",
    "opening": False, "closing": False, "fill_holes": False,
    "footprint": "disk", "radius": 1, "min_area": 0,
}
_CONNECTIVITY = np.ones((3, 3), dtype=bool)


def _validate_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise ValueError("settings debe ser un diccionario de parámetros.")
    if any(key not in _DEFAULTS for key in settings):
        raise ValueError("settings contiene parámetros desconocidos.")
    cfg = {**_DEFAULTS, **settings}
    for key, choices in (("method", ("global", "otsu", "statistical")),
                         ("direction", ("lower", "upper")),
                         ("footprint", ("disk", "square"))):
        if not isinstance(cfg[key], str) or cfg[key] not in choices:
            raise ValueError(f"{key} debe ser una de las opciones {choices}.")
    for key in ("threshold", "k"):
        value = cfg[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{key} debe ser un número real finito no negativo.")
    if cfg["threshold"] > 1:
        raise ValueError("threshold debe estar en [0, 1].")
    for key, minimum in (("radius", 1), ("min_area", 0)):
        value = cfg[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} debe ser un entero mayor o igual a {minimum}.")
    for key in ("opening", "closing", "fill_holes"):
        if not isinstance(cfg[key], (bool, np.bool_)):
            raise ValueError(f"{key} debe ser booleano.")
    return cfg


def segment_slice(image: np.ndarray, liver_mask: np.ndarray,
                  settings: Mapping[str, Any]) -> np.ndarray:
    """Segment dark candidates in an aligned 2D normalized image and liver mask.

    Use the full-size ``denoised`` image from ``preprocess_slice`` and a binary
    liver mask (``seg > 0``). Images must be real, finite and in [0, 1]. Shapes
    must match; invalid inputs/settings raise ValueError. Neither inputs nor
    settings are modified. No tumor reference is used.

    Settings default to global threshold 0.45, statistical k=1/direction='lower',
    no morphology, footprint='disk', radius=1 and min_area=0. Alternatives are
    method='otsu'/'statistical', direction='upper', footprint='square'. Square
    side length is 2*radius+1. Convert HU thresholds before calling with
    (HU - window_lower_bound) / window_width; 40 HU maps to 0.45 for the current
    center=60, width=400 window. Threshold comparisons are strict.

    Opening, closing and hole filling run in that order, clipped to the liver
    after each step. Connected components and hole filling use 8 neighbors;
    components smaller than min_area pixels are discarded (0 disables this).
    Morphology treats pixels outside the image as background. Output is a
    full-size boolean mask. Empty livers return empty masks after validation;
    constant liver intensities return empty masks for Otsu/statistical methods.
    """
    image, liver = np.asarray(image), np.asarray(liver_mask)
    if image.shape != liver.shape:
        raise ValueError(
            "La imagen y la máscara hepática deben tener la misma forma: "
            f"imagen {image.shape}, máscara hepática {liver.shape}."
        )
    if image.ndim != 2 or liver.ndim != 2:
        raise ValueError("La imagen y la máscara hepática deben ser 2D.")
    if (image.dtype.kind not in "uif" or not np.all(np.isfinite(image))
            or np.any(image < 0) or np.any(image > 1)):
        raise ValueError("La imagen debe contener números reales finitos en [0, 1].")
    if liver.dtype.kind not in "buif" or not np.all((liver == 0) | (liver == 1)):
        raise ValueError("La máscara hepática debe ser binaria (0 y 1).")
    cfg = _validate_settings(settings)
    liver = liver.astype(bool, copy=False)
    result = np.zeros(image.shape, dtype=bool)
    if not np.any(liver):
        return result
    values = image[liver].astype(np.float64)
    method = cfg["method"]
    if method != "global" and values.min() == values.max():
        return result
    if method == "global":
        threshold = cfg["threshold"]
    elif method == "otsu":
        threshold = float(threshold_otsu(values))
    else:
        offset = cfg["k"] * values.std(ddof=0)
        threshold = values.mean() + (offset if cfg["direction"] == "upper" else -offset)
    if method == "statistical" and cfg["direction"] == "upper":
        result[liver] = values > threshold
    else:
        result[liver] = values < threshold

    if cfg["opening"] or cfg["closing"]:
        radius = int(cfg["radius"])
        footprint = (disk(radius).astype(bool) if cfg["footprint"] == "disk"
                     else np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool))
        if cfg["opening"]:
            result = ndimage.binary_opening(result, structure=footprint) & liver
        if cfg["closing"]:
            result = ndimage.binary_closing(result, structure=footprint) & liver
    if cfg["fill_holes"]:
        result = ndimage.binary_fill_holes(result, structure=_CONNECTIVITY) & liver
    if cfg["min_area"] > 0:
        labels, _ = ndimage.label(result, structure=_CONNECTIVITY)
        keep = np.bincount(labels.ravel()) >= cfg["min_area"]
        keep[0] = False
        result = keep[labels]
    return result
