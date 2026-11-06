"""Descriptive 3D morphology for reference tumor masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist


def tumor_components_3d(segmentation: np.ndarray, spacing_mm: tuple[float, float, float],
                        connectivity: int = 6):
    """Return physical measurements for each connected tumor component.

    Array axes are canonical RAS ``(X, Y, Z)`` and spacing follows those axes.
    Component dimensions are axis-aligned bounding-box extents in millimetres.
    ``connectivity`` is 6 (faces, the descriptive default) or 26 (faces, edges
    and corners, used for lesion matching so diagonal fragments join a lesion).
    """
    if connectivity not in (6, 26):
        raise ValueError("connectivity debe ser 6 o 26.")
    if segmentation.ndim != 3:
        raise ValueError(f"Expected a 3D label volume, got shape {segmentation.shape}")
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing_mm must contain three positive finite values")

    tumor = segmentation == 2
    voxel_volume_mm3 = float(np.prod(spacing))
    occupied = [np.flatnonzero(np.any(tumor, axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    if any(axis.size == 0 for axis in occupied):
        return [], None

    bounds = [(int(axis[0]), int(axis[-1]) + 1) for axis in occupied]
    crop_slices = tuple(slice(start, stop) for start, stop in bounds)
    roi = tumor[crop_slices]
    labels, count = ndimage.label(roi, structure=ndimage.generate_binary_structure(
        3, 1 if connectivity == 6 else 3))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels, max_label=count)

    records = []
    for component_id, (size, obj) in enumerate(zip(sizes[1:], objects), start=1):
        if obj is None:
            continue
        extents_voxels = tuple(int(axis.stop - axis.start) for axis in obj)
        component = labels[obj] == component_id
        local_coords = np.argwhere(component).astype(np.float64)
        physical_coords = (local_coords + np.asarray([axis.start for axis in obj])) * spacing
        max_feret = _maximum_pair_distance(physical_coords)
        axial_feret = _maximum_axial_feret(component, spacing[:2])
        volume_mm3 = float(size * voxel_volume_mm3)
        bbox_extents = np.asarray(extents_voxels, dtype=float) * spacing
        records.append({
            "component_id": component_id,
            "voxel_count": int(size),
            "volume_cm3": volume_mm3 / 1000.0,
            "extent_x_mm": extents_voxels[0] * spacing[0],
            "extent_y_mm": extents_voxels[1] * spacing[1],
            "extent_z_mm": extents_voxels[2] * spacing[2],
            "max_feret_3d_mm": max_feret,
            "max_axial_feret_mm": axial_feret,
            "equivalent_diameter_mm": float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)),
            "bbox_diagonal_mm": float(np.linalg.norm(bbox_extents)),
            "bbox_start_x": bounds[0][0] + int(obj[0].start),
            "bbox_start_y": bounds[1][0] + int(obj[1].start),
            "bbox_start_z": bounds[2][0] + int(obj[2].start),
        })
    records.sort(key=lambda item: (-item["volume_cm3"], item["component_id"]))
    return records, (labels, bounds, objects)


def remove_small_components_3d(mask: np.ndarray, min_voxels: int = 8) -> np.ndarray:
    """Remove 6-connected foreground components smaller than ``min_voxels``."""
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype.kind not in "buif" or not np.isfinite(mask).all() \
            or not np.isin(mask, (0, 1)).all():
        raise ValueError("mask debe ser un volumen 3D binario y finito.")
    if (isinstance(min_voxels, (bool, np.bool_)) or not isinstance(min_voxels, (int, np.integer))
            or min_voxels < 1):
        raise ValueError("min_voxels debe ser un entero positivo.")
    labels, count = ndimage.label(mask.astype(bool, copy=False),
                                  structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= int(min_voxels)
    keep[0] = False
    return keep[labels]


def filter_components_3d(mask: np.ndarray, spacing_mm, min_voxels: int = 1,
                         min_volume_mm3: float = 0.0, min_slices: int = 1) -> np.ndarray:
    """Keep 6-connected components meeting voxel, physical-volume and axial-span limits.

    ``min_volume_mm3`` is independent of the voxel spacing, unlike ``min_voxels``.
    ``min_slices`` removes components present in fewer axial (Z) slices, which
    targets isolated single-slice false positives of a stacked 2D predictor.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask debe ser un volumen 3D.")
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing_mm debe contener tres valores positivos y finitos.")
    if int(min_voxels) < 1 or int(min_slices) < 1 or not np.isfinite(min_volume_mm3) or min_volume_mm3 < 0:
        raise ValueError("Los umbrales 3D deben ser positivos (min_voxels, min_slices >= 1; min_volume_mm3 >= 0).")
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = (sizes >= int(min_voxels)) & (sizes * float(np.prod(spacing)) >= float(min_volume_mm3))
    if int(min_slices) > 1:
        spans = np.zeros(count + 1, dtype=int)
        for component_id, obj in enumerate(ndimage.find_objects(labels), start=1):
            if obj is not None:
                spans[component_id] = obj[2].stop - obj[2].start
        keep &= spans >= int(min_slices)
    keep[0] = False
    return keep[labels]


def full_component_labels(mask: np.ndarray, render_data) -> np.ndarray:
    """Expand the cropped labels returned by ``tumor_components_3d`` to ``mask.shape``."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    if render_data is not None:
        cropped, bounds, _ = render_data
        labels[tuple(slice(start, stop) for start, stop in bounds)] = cropped
    return labels


SIZE_GROUPS = (("all", 0.0, np.inf), ("<10 mm", 0.0, 10.0), ("10-20 mm", 10.0, 20.0),
               (">=10 mm", 10.0, np.inf), (">=20 mm", 20.0, np.inf))


def match_lesions_3d(reference: np.ndarray, prediction: np.ndarray, spacing_mm,
                     min_iou: float = 0.0, reference_components=None, prediction_components=None,
                     connectivity: int = 26, min_coverage: float = 0.5):
    """Lesion-level comparison of connected reference and predicted components.

    Lesions are 26-connected by default, so annotation fragments touching a
    tumor only by an edge or corner are not counted as separate lesions.
    A reference lesion is matched by every predicted component that overlaps it.
    ``detected``: it overlaps a predicted component and its lesion IoU (lesion
    vs. the union of those components) is >= ``min_iou``. ``covered``: at least
    ``min_coverage`` of its voxels are predicted; this stricter flag is not
    fooled by a large false-positive region that merely touches a small lesion.
    Predicted components that overlap no reference lesion are false positives.
    ``*_components`` accept precomputed ``tumor_components_3d`` outputs computed
    with the same ``connectivity``. Returns ``(lesion_rows, false_positive_rows)``.
    """
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError("reference y prediction deben ser volúmenes 3D de igual forma.")
    if not np.isfinite(min_iou) or not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou debe estar en [0, 1].")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage debe estar en (0, 1].")
    if reference_components is None:
        reference_components = tumor_components_3d(np.where(reference, 2, 0).astype(np.uint8),
                                                   spacing_mm, connectivity)
    if prediction_components is None:
        prediction_components = tumor_components_3d(np.where(prediction, 2, 0).astype(np.uint8),
                                                     spacing_mm, connectivity)
    ref_records, ref_render = reference_components
    pred_records, pred_render = prediction_components
    ref_labels = full_component_labels(reference, ref_render)
    pred_labels = full_component_labels(prediction, pred_render)
    n_ref = int(ref_labels.max(initial=0))
    n_pred = int(pred_labels.max(initial=0))
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)

    both = (ref_labels > 0) & (pred_labels > 0)
    pairs = np.unique(np.stack((ref_labels[both], pred_labels[both]), axis=1), axis=0) \
        if both.any() else np.zeros((0, 2), dtype=int)
    overlap_by_ref = np.bincount(ref_labels[both], minlength=n_ref + 1)
    touched_pred: dict[int, list[int]] = {}
    for ref_id, pred_id in pairs:
        touched_pred.setdefault(int(ref_id), []).append(int(pred_id))
    true_positive_pred = set(int(pred_id) for pred_id in pairs[:, 1])

    lesions = []
    for record in ref_records:
        ref_id = int(record["component_id"])
        matched = touched_pred.get(ref_id, [])
        intersection = int(overlap_by_ref[ref_id])
        matched_voxels = int(pred_sizes[matched].sum()) if matched else 0
        size = int(record["voxel_count"])
        union = size + matched_voxels - intersection
        iou = intersection / union if union else 0.0
        lesions.append({
            **record,
            "matched_prediction_components": len(matched),
            "matched_prediction_voxels": matched_voxels,
            "intersection_voxels": intersection,
            "coverage": intersection / size if size else float("nan"),
            "lesion_dice": 2 * intersection / (size + matched_voxels) if size + matched_voxels else 0.0,
            "lesion_iou": iou,
            "detected": bool(intersection > 0 and iou >= min_iou),
            "covered": bool(size and intersection / size >= min_coverage),
        })
    false_positives = [dict(record) for record in pred_records
                       if int(record["component_id"]) not in true_positive_pred]
    return lesions, false_positives


def summarize_lesions_by_size(lesions, diameter_column: str = "max_axial_feret_mm"):
    """Detection rate, coverage and Dice of reference lesions per size group (mm)."""
    rows = []
    for name, low, high in SIZE_GROUPS:
        group = [item for item in lesions if low <= float(item[diameter_column]) < high]
        n = len(group)
        mean = (lambda key: float(np.mean([float(item[key]) for item in group])) if n else float("nan"))
        rows.append({"size_group": name, "diameter_column": diameter_column, "n_lesions": n,
                     "n_detected": int(sum(bool(item["detected"]) for item in group)),
                     "detection_rate": mean("detected"),
                     "n_covered": int(sum(bool(item["covered"]) for item in group)),
                     "covered_rate": mean("covered"), "mean_coverage": mean("coverage"),
                     "mean_lesion_dice": mean("lesion_dice"),
                     "median_lesion_dice": float(np.median([float(item["lesion_dice"]) for item in group]))
                     if n else float("nan")})
    return rows


def _maximum_pair_distance(points: np.ndarray) -> float:
    """Maximum Euclidean distance between voxel centers (mm), via the convex hull.

    Works in any dimension. Degenerate sets (planar in 3D, collinear) are
    projected onto their principal subspace, so the result stays exact without
    an all-pairs search over every voxel.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    try:
        points = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        centered = points - points.mean(axis=0)
        _, singular, basis = np.linalg.svd(centered, full_matrices=False)
        rank = int((singular > 1e-9 * max(float(singular[0]), 1.0)).sum())
        if rank <= 1:
            projection = centered @ basis[0]
            return float(projection.max() - projection.min())
        try:
            points = points[ConvexHull(centered @ basis[:rank].T).vertices]
        except (QhullError, ValueError):
            pass
    if len(points) <= 10_000:
        return float(pdist(points).max(initial=0.0))
    maximum = 0.0
    for start in range(0, len(points), 16):
        distances = np.linalg.norm(points[start:start + 16, None, :] - points[None, :, :], axis=2)
        maximum = max(maximum, float(distances.max(initial=0.0)))
    return maximum


def _maximum_axial_feret(mask: np.ndarray, spacing_xy: np.ndarray) -> float:
    """Largest in-plane diameter across axial slices, between voxel centers (mm).

    Uses the same voxel-center convention as the 3D Feret, so the axial value
    never exceeds the 3D value of the same component.
    """
    spacing_xy = np.asarray(spacing_xy, dtype=float)
    diameter = 0.0
    for axial_slice in range(mask.shape[2]):
        coords = np.argwhere(mask[:, :, axial_slice])
        if len(coords) >= 2:
            diameter = max(diameter, _maximum_pair_distance(coords * spacing_xy))
    return diameter


def latest_morphology_output(results_dir) -> "Path | None":
    """Newest ``morphology_3d/<run_id>`` folder with a summary (or the legacy flat folder)."""
    base = Path(results_dir) / "morphology_3d"
    runs = sorted((path for path in base.glob("*") if path.is_dir()
                   and (path / "morphology_3d_per_case.csv").is_file()),
                  key=lambda path: (path / "morphology_3d_per_case.csv").stat().st_mtime)
    if runs:
        return runs[-1]
    return base if (base / "morphology_3d_per_case.csv").is_file() else None
