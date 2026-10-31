"""Geometric features for tumor regions in axial 2D masks."""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


GEOMETRY_COLUMNS = (
    "component_id",
    "area_pixels",
    "area_mm2",
    "eccentricity",
    "solidity",
    "max_diameter_mm",
    "centroid_x_mm",
    "centroid_y_mm",
)


def _validate_mask_and_spacing(
    mask: np.ndarray,
    spacing: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"La máscara tumoral debe ser 2D; se recibió {mask.shape}.")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("La máscara tumoral debe contener solo valores binarios 0 y 1.")
    if not isinstance(spacing, (tuple, list, np.ndarray)) or len(spacing) != 2:
        raise ValueError("spacing debe contener (sx, sy) en milímetros para el corte 2D.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("Cada valor de spacing debe ser un número real finito positivo.")
    return mask.astype(bool, copy=False), (float(spacing[0]), float(spacing[1]))


def extract_geometric_features(
    tumor_mask: np.ndarray,
    spacing: tuple[float, float],
) -> pd.DataFrame:
    """Measure each 8-connected lesion in one axial tumor mask.

    ``spacing`` follows the mask axes ``(X, Y)`` and is expressed in mm. Areas,
    centroids, and maximum Feret diameter are reported in physical units where
    applicable. Eccentricity and solidity are dimensionless. An empty mask
    returns an empty table with the same columns.
    """
    mask, spacing = _validate_mask_and_spacing(tumor_mask, spacing)
    labels = label(mask, connectivity=2)
    pixel_counts = np.bincount(labels.ravel())
    rows = []
    for region in regionprops(labels, spacing=spacing):
        rows.append({
            "component_id": int(region.label),
            "area_pixels": int(pixel_counts[region.label]),
            "area_mm2": float(region.area),
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "max_diameter_mm": float(region.feret_diameter_max),
            "centroid_x_mm": float(region.centroid[0]),
            "centroid_y_mm": float(region.centroid[1]),
        })
    return pd.DataFrame(rows, columns=GEOMETRY_COLUMNS)
