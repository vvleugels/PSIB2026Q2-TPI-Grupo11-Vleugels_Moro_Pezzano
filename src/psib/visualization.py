"""Visualization helpers for comparing tumor masks on axial CT slices."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np


PREDICTION_COLOR = "deepskyblue"
REFERENCE_COLOR = "orangered"


def _validate_inputs(image: np.ndarray, prediction: np.ndarray,
                     reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image)
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if image.ndim != 2:
        raise ValueError(f"La imagen debe ser 2D; se recibió {image.shape}.")
    if prediction.shape != image.shape or reference.shape != image.shape:
        raise ValueError(
            "La imagen y las máscaras deben tener la misma forma: "
            f"imagen {image.shape}, predicción {prediction.shape}, referencia {reference.shape}."
        )
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("La imagen debe contener valores numéricos reales y finitos.")
    for name, mask in (("predicción", prediction), ("referencia", reference)):
        if mask.dtype.kind not in "buif" or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"La máscara de {name} debe ser binaria (0 y 1).")
    return image, prediction.astype(bool, copy=False), reference.astype(bool, copy=False)


def plot_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw prediction and reference contours over one aligned axial image.

    Arrays follow the project's ``(X, Y)`` convention and are transposed for
    display, matching the other project figures. The automatic prediction is
    cyan and the reference annotation is orange-red.
    """
    image, prediction, reference = _validate_inputs(image, prediction, reference)
    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 6))
    else:
        figure = ax.figure

    ax.imshow(image.T, cmap="gray", origin="lower")
    for mask, color in ((prediction, PREDICTION_COLOR), (reference, REFERENCE_COLOR)):
        if mask.any():
            ax.contour(mask.T.astype(np.uint8), levels=(0.5,), colors=(color,), linewidths=1.4,
                       origin="lower")
    ax.legend(handles=(
        Line2D([0], [0], color=PREDICTION_COLOR, lw=1.8, label="Predicción"),
        Line2D([0], [0], color=REFERENCE_COLOR, lw=1.8, label="Referencia"),
    ), loc="upper right")
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return figure, ax


def save_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    """Save a prediction/reference contour overlay and return its path."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi debe ser un entero positivo.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_overlay(image, prediction, reference, title=title)
    try:
        figure.tight_layout()
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path
