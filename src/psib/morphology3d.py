"""Descriptive 3D morphology for reference tumor masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


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
        records.append({
            "component_id": component_id,
            "voxel_count": int(size),
            "volume_cm3": float(size * voxel_volume_mm3 / 1000.0),
            "extent_x_mm": extents_voxels[0] * spacing[0],
            "extent_y_mm": extents_voxels[1] * spacing[1],
            "extent_z_mm": extents_voxels[2] * spacing[2],
            "bbox_start_x": bounds[0][0] + int(obj[0].start),
            "bbox_start_y": bounds[1][0] + int(obj[1].start),
            "bbox_start_z": bounds[2][0] + int(obj[2].start),
        })
    records.sort(key=lambda item: (-item["volume_cm3"], item["component_id"]))
    return records, (labels, bounds, objects)
