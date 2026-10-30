"""Dice and IoU for spatially aligned binary masks.

Inputs must describe the same spatial region: equal shapes alone do not
guarantee alignment. Masks are never resized or cropped automatically.
"""
from __future__ import annotations

import numpy as np


def _validate_masks(pred: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate matching 2D/3D binary masks and return boolean arrays."""
    pred, reference = np.asarray(pred), np.asarray(reference)
    if pred.shape != reference.shape:
        raise ValueError(
            "Las máscaras deben tener la misma forma: "
            f"predicción {pred.shape}, referencia {reference.shape}."
        )
    for name, mask in (("predicción", pred), ("referencia", reference)):
        if mask.ndim not in (2, 3):
            raise ValueError(f"La máscara de {name} debe ser 2D o 3D.")
        if mask.dtype.kind not in "buifc" or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"La máscara de {name} debe contener solo valores booleanos o numéricos 0 y 1.")
    return pred.astype(bool, copy=False), reference.astype(bool, copy=False)


def dice_score(pred: np.ndarray, reference: np.ndarray) -> float:
    """Return Dice in [0, 1]; two empty masks score 1, only one empty scores 0.

    Accept aligned 2D/3D boolean or numeric 0/1 masks. Invalid shapes or values
    raise ValueError. Inputs are not modified.
    """
    pred, reference = _validate_masks(pred, reference)
    total = int(np.count_nonzero(pred)) + int(np.count_nonzero(reference))
    if total == 0:
        return 1.0
    return 2 * int(np.count_nonzero(pred & reference)) / total


def iou_score(pred: np.ndarray, reference: np.ndarray) -> float:
    """Return IoU in [0, 1]; two empty masks score 1, only one empty scores 0.

    Accept aligned 2D/3D boolean or numeric 0/1 masks. Invalid shapes or values
    raise ValueError. Inputs are not modified.
    """
    pred, reference = _validate_masks(pred, reference)
    union = int(np.count_nonzero(pred | reference))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(pred & reference)) / union
