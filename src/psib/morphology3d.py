"""Descriptive 3D morphology for reference tumor masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist


def tumor_components_3d(segmentation: np.ndarray, spacing_mm: tuple[float, float, float]):
    """Return physical measurements for each 6-connected tumor component.

    Array axes are canonical RAS ``(X, Y, Z)`` and spacing follows those axes.
    Component dimensions are axis-aligned bounding-box extents in millimetres.
    """
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
    labels, count = ndimage.label(roi, structure=ndimage.generate_binary_structure(3, 1))
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


def _maximum_pair_distance(points: np.ndarray) -> float:
    """Maximum Euclidean distance between convex-hull vertices (mm)."""
    if len(points) < 2:
        return 0.0
    try:
        points = points[ConvexHull(points).vertices]
    except QhullError:
        # Degenerate (e.g. planar) components have no 3D hull; use all centers.
        pass
    if len(points) < 2:
        return 0.0
    if len(points) <= 10_000:
        return float(pdist(points).max(initial=0.0))
    # Keep the exact all-pairs fallback bounded in memory for very large flat masks.
    maximum = 0.0
    for start in range(0, len(points), 16):
        distances = np.linalg.norm(points[start:start + 16, None, :] - points[None, :, :], axis=2)
        maximum = max(maximum, float(distances.max(initial=0.0)))
    return maximum


def _maximum_axial_feret(mask: np.ndarray, spacing_xy: np.ndarray) -> float:
    """Return the largest 2D maximum-Feret diameter across axial slices."""
    from .features import extract_geometric_features

    diameter = 0.0
    for axial_slice in range(mask.shape[2]):
        if not mask[:, :, axial_slice].any():
            continue
        geometries = extract_geometric_features(mask[:, :, axial_slice], tuple(spacing_xy))
        if not geometries.empty:
            diameter = max(diameter, float(geometries["max_diameter_mm"].max()))
    return diameter
