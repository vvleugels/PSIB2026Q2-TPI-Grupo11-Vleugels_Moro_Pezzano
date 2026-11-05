"""Lesion-level correspondence and physical geometry metrics for 2D masks."""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from scipy import ndimage

from .features import extract_geometric_features


def match_lesions(pred, ref, spacing, min_iou: float = 0.0) -> pd.DataFrame:
    """Group split/merged 8-connected components and measure each group.

    Every reference lesion first maps to its overlapping prediction with the
    largest pixel overlap. Reference lesions selecting the same prediction are
    merged into one correspondence group. Each remaining prediction is assigned
    to the reference group it overlaps most; predictions without an accepted
    overlap become false-positive groups. ``min_iou`` applies to component
    links and defaults to any positive overlap.
    """
    pred, ref = _validate_binary_pair(pred, ref)
    if (not isinstance(spacing, (tuple, list, np.ndarray)) or len(spacing) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                   or not np.isfinite(v) or v <= 0 for v in spacing)):
        raise ValueError("spacing debe contener dos valores positivos y finitos en mm.")
    if (isinstance(min_iou, (bool, np.bool_)) or not isinstance(min_iou, Real)
            or not np.isfinite(min_iou) or not 0 <= min_iou <= 1):
        raise ValueError("min_iou debe estar entre 0 y 1.")

    structure = np.ones((3, 3), dtype=bool)
    pred_labels, n_pred = ndimage.label(pred, structure=structure)
    ref_labels, n_ref = ndimage.label(ref, structure=structure)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)
    ref_sizes = np.bincount(ref_labels.ravel(), minlength=n_ref + 1)
    intersections = np.zeros((n_ref + 1, n_pred + 1), dtype=np.int64)
    both = (ref_labels > 0) & (pred_labels > 0)
    if both.any():
        pairs = ref_labels[both] * (n_pred + 1) + pred_labels[both]
        intersections = np.bincount(pairs, minlength=(n_ref + 1) * (n_pred + 1)).reshape(
            n_ref + 1, n_pred + 1)
    denominators = ref_sizes[:, None] + pred_sizes[None, :] - intersections
    ious = np.divide(intersections, denominators, out=np.zeros_like(intersections, dtype=float),
                     where=denominators > 0)

    groups: list[dict[str, set[int]]] = []
    ref_to_group: dict[int, int] = {}
    pred_to_group: dict[int, int] = {}
    group_for_pred: dict[int, int] = {}
    for ref_id in range(1, n_ref + 1):
        candidates = [p for p in range(1, n_pred + 1)
                      if intersections[ref_id, p] > 0 and ious[ref_id, p] >= min_iou]
        if not candidates:
            continue
        best_pred = max(candidates, key=lambda p: (intersections[ref_id, p], -p))
        group_index = group_for_pred.get(best_pred)
        if group_index is None:
            group_index = len(groups)
            groups.append({"ref": set(), "pred": {best_pred}})
            group_for_pred[best_pred] = group_index
            pred_to_group[best_pred] = group_index
        groups[group_index]["ref"].add(ref_id)
        ref_to_group[ref_id] = group_index

    for pred_id in range(1, n_pred + 1):
        if pred_id in pred_to_group:
            continue
        choices = []
        for group_index, group in enumerate(groups):
            ref_ids = sorted(group["ref"])
            overlap = int(intersections[ref_ids, pred_id].sum())
            ref_area = int(ref_sizes[ref_ids].sum())
            iou = overlap / (ref_area + int(pred_sizes[pred_id]) - overlap) if overlap else 0.0
            if overlap and iou >= min_iou:
                choices.append((overlap, -group_index, group_index))
        if choices:
            group_index = max(choices)[2]
            groups[group_index]["pred"].add(pred_id)
            pred_to_group[pred_id] = group_index

    for ref_id in range(1, n_ref + 1):
        if ref_id not in ref_to_group:
            group_index = len(groups)
            groups.append({"ref": {ref_id}, "pred": set()})
            ref_to_group[ref_id] = group_index
    for pred_id in range(1, n_pred + 1):
        if pred_id not in pred_to_group:
            groups.append({"ref": set(), "pred": {pred_id}})

    rows = []
    for group_id, group in enumerate(groups, start=1):
        ref_ids, pred_ids = sorted(group["ref"]), sorted(group["pred"])
        ref_mask = np.isin(ref_labels, ref_ids) if ref_ids else np.zeros(ref.shape, dtype=bool)
        pred_mask = np.isin(pred_labels, pred_ids) if pred_ids else np.zeros(pred.shape, dtype=bool)
        ref_geometry = extract_geometric_features(ref_mask, spacing)
        pred_geometry = extract_geometric_features(pred_mask, spacing)
        area_ref = int(ref_mask.sum())
        area_pred = int(pred_mask.sum())
        area_ref_mm2 = area_ref * float(spacing[0]) * float(spacing[1])
        area_pred_mm2 = area_pred * float(spacing[0]) * float(spacing[1])
        diameter_ref = float(ref_geometry["max_diameter_mm"].max()) if not ref_geometry.empty else np.nan
        diameter_pred = float(pred_geometry["max_diameter_mm"].max()) if not pred_geometry.empty else np.nan
        intersection = int(np.count_nonzero(ref_mask & pred_mask))
        dice = 2 * intersection / (area_ref + area_pred) if area_ref + area_pred else 0.0
        status = "TP" if ref_ids and pred_ids else ("FN" if ref_ids else "FP")
        size_class = classify_diameter(diameter_ref) if ref_ids else "none"
        rows.append({
            "group_id": group_id,
            "reference_components": ";".join(map(str, ref_ids)),
            "predicted_components": ";".join(map(str, pred_ids)),
            "status": status,
            "size_class": size_class,
            "area_ref_pixels": area_ref,
            "area_pred_pixels": area_pred,
            "area_ref_mm2": area_ref_mm2,
            "area_pred_mm2": area_pred_mm2,
            "relative_area_error": ((area_pred_mm2 - area_ref_mm2) / area_ref_mm2
                                    if area_ref_mm2 else np.nan),
            "max_feret_ref_mm": diameter_ref,
            "max_feret_pred_mm": diameter_pred,
            "diameter_error_mm": diameter_pred - diameter_ref
            if np.isfinite(diameter_ref) and np.isfinite(diameter_pred) else np.nan,
            "group_dice": dice,
        })
    return pd.DataFrame(rows, columns=(
        "group_id", "reference_components", "predicted_components", "status", "size_class",
        "area_ref_pixels", "area_pred_pixels", "area_ref_mm2", "area_pred_mm2",
        "relative_area_error", "max_feret_ref_mm", "max_feret_pred_mm",
        "diameter_error_mm", "group_dice",
    ))


def classify_diameter(diameter_mm: float) -> str:
    """Map reference maximum Feret diameter to the requested size bins."""
    if not np.isfinite(diameter_mm) or diameter_mm < 0:
        return "none"
    if diameter_mm < 10:
        return "<10"
    if diameter_mm < 20:
        return "10-20"
    return ">=20"


def _validate_binary_pair(pred, ref) -> tuple[np.ndarray, np.ndarray]:
    pred, ref = np.asarray(pred), np.asarray(ref)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError("pred y ref deben ser máscaras 2D de igual forma.")
    for name, mask in (("pred", pred), ("ref", ref)):
        if mask.dtype.kind not in "buif" or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{name} debe ser una máscara binaria finita.")
    return pred.astype(bool, copy=False), ref.astype(bool, copy=False)


def prepare_reference(ref, spacing):
    """Cache reference geometry for evaluation only; never pass it to detection."""
    ref, _ = _validate_binary_pair(ref, ref)
    labels, count = ndimage.label(ref, structure=np.ones((3, 3), dtype=bool))
    geometry = extract_geometric_features(ref, spacing)
    diameters = dict(zip(geometry.component_id, geometry.max_diameter_mm))
    return labels, np.bincount(labels.ravel(), minlength=count + 1), diameters


def match_lesions_one_to_one(pred, ref, spacing, min_iou=0.10, *, prepared_reference=None):
    """Maximize valid match cardinality first, total IoU second.

    Returns one row per reference lesion plus unmatched predictions. Unlike
    correspondence groups, splits/merges cannot count multiple detections.
    """
    pred, ref = _validate_binary_pair(pred, ref)
    if not np.isfinite(min_iou) or not 0 <= min_iou <= 1:
        raise ValueError('min_iou must be in [0, 1]')
    rlabels, rsizes, diameters = (prepare_reference(ref, spacing) if prepared_reference is None
                                else prepared_reference)
    plabels, npred = ndimage.label(pred, structure=np.ones((3, 3), dtype=bool))
    nref = len(rsizes) - 1
    psizes = np.bincount(plabels.ravel(), minlength=npred + 1)
    both = (rlabels > 0) & (plabels > 0)
    intersections = np.bincount((rlabels[both] * (npred + 1) + plabels[both]),
        minlength=(nref + 1) * (npred + 1)).reshape(nref + 1, npred + 1)[1:, 1:]
    union = rsizes[1:, None] + psizes[None, 1:] - intersections
    ious = np.divide(intersections, union, out=np.zeros_like(intersections, dtype=float), where=union > 0)
    valid = (intersections > 0) & (ious >= min_iou)
    matches = {}
    if nref and npred:
        matches = {int(r + 1): int(p + 1) for r, p in _optimal_pairs(ious, valid)}
    rows = [dict(reference_id=r, predicted_id=matches.get(r, 0),
                 status='TP' if r in matches else 'FN',
                 iou=float(ious[r-1, matches[r]-1]) if r in matches else 0.0,
                 diameter_ref_mm=float(diameters[r]), size_class=classify_diameter(diameters[r]))
            for r in range(1, nref + 1)]
    matched_predictions = set(matches.values())
    rows.extend(dict(reference_id=0, predicted_id=p, status='FP', iou=0.0,
                     diameter_ref_mm=np.nan, size_class='none') for p in range(1, npred + 1)
                if p not in matched_predictions)
    return pd.DataFrame(rows, columns=['reference_id', 'predicted_id', 'status', 'iou', 'diameter_ref_mm', 'size_class'])


def _optimal_pairs(ious, valid):
    """Cardinality dominates every possible difference in summed IoU."""
    from scipy.optimize import linear_sum_assignment
    if not all(ious.shape):
        return []
    reward = valid * (min(ious.shape) + 1 + ious)
    rr, pp = linear_sum_assignment(reward, maximize=True)
    return [(int(r), int(p)) for r, p in zip(rr, pp) if valid[r, p]]
