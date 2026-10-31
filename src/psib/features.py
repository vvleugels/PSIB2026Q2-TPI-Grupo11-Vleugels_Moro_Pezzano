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
TEXTURE_COLUMNS = ("contrast", "homogeneity", "energy", "correlation", "pair_count", "n_offsets")


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


def extract_texture_features(
    image: np.ndarray,
    roi_mask: np.ndarray,
    *,
    levels: int = 32,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, float | int]:
    """Return masked 2D GLCM texture features for one region of interest.

    ``image`` must be a normalized axial image in [0, 1]. Intensities are
    quantized into ``levels`` bins. For each distance/angle, only pixel pairs
    whose two endpoints are inside ``roi_mask`` contribute to the symmetric
    co-occurrence matrix. The four properties are averaged over offsets with
    at least one valid pair. If there are no such pairs, properties are NaN.
    """
    image = np.asarray(image)
    mask = np.asarray(roi_mask)
    if image.ndim != 2:
        raise ValueError(f"La imagen de textura debe ser 2D; se recibió {image.shape}.")
    if mask.shape != image.shape:
        raise ValueError(f"La imagen y roi_mask deben coincidir: {image.shape} vs. {mask.shape}.")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
        raise ValueError("La imagen de textura debe ser real, finita y estar normalizada en [0, 1].")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("roi_mask debe contener solo valores binarios 0 y 1.")
    if (isinstance(levels, (bool, np.bool_)) or not isinstance(levels, (int, np.integer))
            or not 2 <= levels <= 256):
        raise ValueError("levels debe ser un entero entre 2 y 256.")
    if not isinstance(distances, (tuple, list, np.ndarray)) or len(distances) == 0:
        raise ValueError("distances debe contener al menos una distancia positiva.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1
           for value in distances):
        raise ValueError("Cada distancia debe ser un entero positivo.")
    if not isinstance(angles, (tuple, list, np.ndarray)) or len(angles) == 0:
        raise ValueError("angles debe contener al menos un ángulo en radianes.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)
           for value in angles):
        raise ValueError("Cada ángulo debe ser un número real finito en radianes.")

    quantized = np.minimum((image * levels).astype(np.int32), levels - 1)
    height, width = image.shape
    gray_levels = np.arange(levels, dtype=float)
    difference = gray_levels[:, None] - gray_levels[None, :]
    offset_features: list[tuple[float, float, float, float]] = []
    pair_count = 0

    for distance in distances:
        for angle in angles:
            row_shift = int(round(int(distance) * np.sin(float(angle))))
            col_shift = int(round(int(distance) * np.cos(float(angle))))
            if row_shift == 0 and col_shift == 0:
                continue
            row_start, row_stop = max(0, -row_shift), min(height, height - row_shift)
            col_start, col_stop = max(0, -col_shift), min(width, width - col_shift)
            if row_start >= row_stop or col_start >= col_stop:
                continue
            source_mask = mask[row_start:row_stop, col_start:col_stop].astype(bool, copy=False)
            target_mask = mask[row_start + row_shift:row_stop + row_shift,
                               col_start + col_shift:col_stop + col_shift].astype(bool, copy=False)
            valid = source_mask & target_mask
            count = int(np.count_nonzero(valid))
            if count == 0:
                continue

            source = quantized[row_start:row_stop, col_start:col_stop][valid]
            target = quantized[row_start + row_shift:row_stop + row_shift,
                               col_start + col_shift:col_stop + col_shift][valid]
            cooccurrence = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(cooccurrence, (source, target), 1)
            cooccurrence += cooccurrence.T
            probabilities = cooccurrence / cooccurrence.sum()

            row_marginal, col_marginal = probabilities.sum(axis=1), probabilities.sum(axis=0)
            row_mean = float(np.dot(gray_levels, row_marginal))
            col_mean = float(np.dot(gray_levels, col_marginal))
            row_std = float(np.sqrt(np.dot((gray_levels - row_mean) ** 2, row_marginal)))
            col_std = float(np.sqrt(np.dot((gray_levels - col_mean) ** 2, col_marginal)))
            if row_std == 0 or col_std == 0:
                correlation = 1.0
            else:
                correlation = float(np.sum(
                    probabilities * (gray_levels[:, None] - row_mean)
                    * (gray_levels[None, :] - col_mean)
                ) / (row_std * col_std))
            offset_features.append((
                float(np.sum(probabilities * difference ** 2)),
                float(np.sum(probabilities / (1 + difference ** 2))),
                float(np.sqrt(np.sum(probabilities ** 2))),
                correlation,
            ))
            pair_count += count

    if not offset_features:
        return {"contrast": float("nan"), "homogeneity": float("nan"), "energy": float("nan"),
                "correlation": float("nan"), "pair_count": 0, "n_offsets": 0}
    means = np.asarray(offset_features, dtype=float).mean(axis=0)
    return {"contrast": float(means[0]), "homogeneity": float(means[1]),
            "energy": float(means[2]), "correlation": float(means[3]),
            "pair_count": pair_count, "n_offsets": len(offset_features)}
