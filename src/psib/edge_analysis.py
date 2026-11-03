"""Edge maps, Laplacian sharpening, and tumor-boundary measurements."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import laplace, prewitt, sobel
from skimage.segmentation import watershed


def edge_map(smoothed: np.ndarray, operator: str) -> np.ndarray:
    """Apply exactly one selected high-pass operator to a smoothed image."""
    image = np.asarray(smoothed, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("smoothed image must be 2D")
    operator = str(operator).lower()
    filters = {"sobel": sobel, "prewitt": prewitt, "laplace": laplace}
    if operator not in filters:
        raise ValueError("edge operator must be 'sobel', 'prewitt', or 'laplace'")
    return np.asarray(filters[operator](image), dtype=np.float32)


def sharpen_with_laplacian(smoothed: np.ndarray, alpha: float = 0.5,
                           response: np.ndarray | None = None) -> np.ndarray:
    """Sharpen a normalized image using scikit-image's Laplacian sign convention."""
    image = np.asarray(smoothed, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("smoothed image must be 2D")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    laplacian = laplace(image) if response is None else np.asarray(response, dtype=np.float32)
    if laplacian.shape != image.shape:
        raise ValueError("Laplacian response must match the smoothed image shape")
    # skimage.filters.laplace uses a positive center coefficient and negative
    # neighbors, so adding its response boosts peaks and deepens adjacent valleys.
    return np.clip(image + float(alpha) * laplacian, 0.0, 1.0).astype(np.float32)


def watershed_refine(gradient_map: np.ndarray, prediction: np.ndarray,
                     liver: np.ndarray, expansion: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Refine a prediction with local watershed over one selected edge map."""
    gradient = np.asarray(gradient_map, dtype=np.float32)
    pred, liver_mask = np.asarray(prediction, dtype=bool), np.asarray(liver, dtype=bool)
    if gradient.ndim != 2 or gradient.shape != pred.shape or gradient.shape != liver_mask.shape:
        raise ValueError("gradient and masks must be 2D arrays with matching shapes")
    if expansion < 1:
        raise ValueError("expansion must be positive")
    if not np.any(pred):
        return pred.copy(), np.zeros(pred.shape, dtype=np.int32)

    region = ndimage.binary_dilation(pred, iterations=expansion) & liver_mask
    components, component_count = ndimage.label(pred, structure=np.ones((3, 3), dtype=bool))
    markers = np.zeros(pred.shape, dtype=np.int32)
    outer_background = region & ~ndimage.binary_dilation(pred, iterations=1)
    markers[outer_background] = 1
    object_markers = []
    next_marker = 2
    for component_id in range(1, component_count + 1):
        component = components == component_id
        seed = ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=bool))
        if not np.any(seed):
            seed = component
        markers[seed] = next_marker
        object_markers.append(next_marker)
        next_marker += 1
    if not np.any(markers == 1):
        return pred.copy(), np.zeros(pred.shape, dtype=np.int32)
    labels = watershed(gradient, markers, mask=region)
    return np.isin(labels, object_markers), labels


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("tumor masks must be 2D")
    return binary & ~ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))


def _mean_on(mask: np.ndarray, values: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def measure_edge_metrics(smoothed: np.ndarray, liver: np.ndarray,
                         reference: np.ndarray, prediction: np.ndarray,
                         operator: str, response: np.ndarray | None = None) -> dict[str, float]:
    """Measure the selected operator response on contours and healthy liver.

    Healthy-liver gradient statistics are a noise/texture proxy, not a direct
    estimate of scanner noise. Tumor-edge sharpness is the mean gradient
    magnitude sampled on each mask's one-pixel boundary.
    """
    image = np.asarray(smoothed, dtype=np.float32)
    liver_mask = np.asarray(liver, dtype=bool)
    ref_mask, pred_mask = np.asarray(reference, dtype=bool), np.asarray(prediction, dtype=bool)
    if image.ndim != 2 or image.shape != liver_mask.shape or image.shape != ref_mask.shape or image.shape != pred_mask.shape:
        raise ValueError("image and masks must be 2D arrays with matching shapes")
    response = edge_map(image, operator) if response is None else np.asarray(response, dtype=np.float32)
    healthy = ndimage.binary_erosion(liver_mask, iterations=2)
    tumor_exclusion = ndimage.binary_dilation(ref_mask, iterations=2)
    healthy &= ~tumor_exclusion
    result: dict[str, float] = {}
    if response.shape != image.shape:
        raise ValueError("edge response must match the smoothed image shape")
    result["reference_edge_gradient"] = _mean_on(_boundary(ref_mask), np.abs(response))
    result["prediction_edge_gradient"] = _mean_on(_boundary(pred_mask), np.abs(response))
    result["healthy_liver_gradient_mean"] = _mean_on(healthy, np.abs(response))
    result["healthy_liver_gradient_std"] = (
        float(np.std(np.abs(response[healthy]))) if np.any(healthy) else float("nan"))
    return result
