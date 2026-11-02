"""Simulated tomographic acquisition and reconstruction helpers for 2D slices."""
from __future__ import annotations

import numpy as np
from skimage.transform import iradon, iradon_sart, radon, resize

from .preprocessing import liver_bounding_box


def hu_to_mu(hu: np.ndarray) -> np.ndarray:
    """Map Hounsfield units to relative attenuation (air 0, water 1)."""
    array = np.asarray(hu, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("HU values must be finite")
    return np.maximum(array + 1000.0, 0.0) / 1000.0


def mu_to_hu(mu: np.ndarray) -> np.ndarray:
    """Map relative attenuation to HU using the inverse linear relation."""
    array = np.asarray(mu, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("attenuation values must be finite")
    return array * 1000.0 - 1000.0


def psnr(reference: np.ndarray, image: np.ndarray, data_range: float = 1.0) -> float:
    """Compute PSNR, returning positive infinity for identical arrays."""
    reference, image = np.asarray(reference, dtype=np.float64), np.asarray(image, dtype=np.float64)
    if reference.shape != image.shape or reference.size == 0:
        raise ValueError("reference and image must be nonempty arrays with equal shapes")
    if not np.isfinite(reference).all() or not np.isfinite(image).all():
        raise ValueError("PSNR inputs must be finite")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be positive and finite")
    mse = float(np.mean((reference - image) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def pad_object_to_square(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Center-pad a 2D ROI so its rectangle lies inside the square's inscribed circle.

    ``radon(circle=True)`` treats values outside the inscribed circle as zero;
    choosing a side at least the ROI diagonal prevents the rectangular ROI
    corners from being discarded. The returned box maps the ROI into the pad.
    """
    array = np.asarray(image)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("image must be a nonempty finite 2D array")
    height, width = array.shape
    # Two extra pixels keep boundary pixel centers strictly inside the
    # inscribed circle despite the integer center convention in scikit-image.
    side = int(np.ceil(np.hypot(height, width))) + 2
    side = max(side, height, width)
    top, left = (side - height) // 2, (side - width) // 2
    padded = np.zeros((side, side), dtype=array.dtype)
    padded[top:top + height, left:left + width] = array
    return padded, (top, left, top + height, left + width)


def simulate_reconstruction(hu_roi: np.ndarray, *, angle_count: int,
                            downsample: int = 1, sart_iterations: int = 3,
                            sart_relaxation: float = 0.15) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Simulate projections and return SBP, two FBP variants and SART in HU.

    Returns ``(method_images, sinogram, theta)`` at the original ROI shape.
    ``hu_roi`` should contain the liver bounding box plus its requested margin;
    the function performs the attenuation conversion and circle-safe padding.
    """
    source = np.asarray(hu_roi, dtype=np.float64)
    if source.ndim != 2 or not source.size or not np.isfinite(source).all():
        raise ValueError("hu_roi must be a nonempty finite 2D array")
    if isinstance(angle_count, bool) or not isinstance(angle_count, (int, np.integer)) or angle_count < 1:
        raise ValueError("angle_count must be a positive integer")
    if isinstance(downsample, bool) or not isinstance(downsample, (int, np.integer)) or downsample < 1:
        raise ValueError("downsample must be a positive integer")
    if isinstance(sart_iterations, bool) or not isinstance(sart_iterations, (int, np.integer)) or sart_iterations < 1:
        raise ValueError("sart_iterations must be a positive integer")
    if not np.isfinite(sart_relaxation) or sart_relaxation <= 0:
        raise ValueError("sart_relaxation must be positive and finite")
    mu = hu_to_mu(source)
    if downsample > 1:
        shape = (max(1, int(np.ceil(mu.shape[0] / downsample))),
                 max(1, int(np.ceil(mu.shape[1] / downsample))))
        mu = resize(mu, shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float64)
    padded, roi_box = pad_object_to_square(mu)
    theta = np.linspace(0.0, 180.0, int(angle_count), endpoint=False)
    sinogram = radon(padded, theta=theta, circle=True, preserve_range=True)
    reconstructions = {
        "sbp": iradon(sinogram, theta=theta, output_size=padded.shape[0], filter_name=None,
                      circle=True, preserve_range=True),
        "fbp_ramp": iradon(sinogram, theta=theta, output_size=padded.shape[0], filter_name="ramp",
                            circle=True, preserve_range=True),
        "fbp_hann": iradon(sinogram, theta=theta, output_size=padded.shape[0], filter_name="hann",
                            circle=True, preserve_range=True),
    }
    sart = np.zeros_like(padded, dtype=np.float64)
    for _ in range(int(sart_iterations)):
        sart = iradon_sart(sinogram, theta=theta, image=sart, relaxation=float(sart_relaxation))
    reconstructions["sart"] = sart
    top, left, bottom, right = roi_box
    roi_recons = {name: image[top:bottom, left:right] for name, image in reconstructions.items()}
    if downsample > 1:
        roi_recons = {name: resize(image, source.shape, order=1, preserve_range=True,
                                   anti_aliasing=False).astype(np.float64)
                      for name, image in roi_recons.items()}
    return {name: mu_to_hu(image) for name, image in roi_recons.items()}, sinogram, theta


def reconstruct_liver_roi(slice_hu: np.ndarray, liver_mask: np.ndarray, *, margin: int = 8,
                          **settings) -> tuple[dict[str, np.ndarray], tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Reconstruct the liver bounding box and return outputs plus its full-slice box."""
    image, liver = np.asarray(slice_hu), np.asarray(liver_mask, dtype=bool)
    if image.ndim != 2 or image.shape != liver.shape:
        raise ValueError("slice_hu and liver_mask must be aligned 2D arrays")
    box = liver_bounding_box(liver, margin)
    x0, y0, x1, y1 = box
    methods, sinogram, theta = simulate_reconstruction(image[x0:x1, y0:y1], **settings)
    return methods, box, sinogram, theta
