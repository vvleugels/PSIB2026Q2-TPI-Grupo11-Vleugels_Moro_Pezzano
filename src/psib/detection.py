"""Experimental, reference-free hypodense lesion detection in a known liver ROI.

HU analysis is independent of display window/CLAHE. Physical units follow the
canonical array axes. No model is trained and no tumor labels enter this API.
"""
from itertools import product
from numbers import Real
from collections.abc import Mapping

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

DEFAULTS = dict(method='hysteresis', sigma_mm=0.6, background_sigma_mm=10.0,
                high=1.0, low=0.5, min_area_mm2=0.0, noise_floor_hu=5.0)
CONNECTIVITY = np.ones((3, 3), dtype=bool)


def validate_settings(settings):
    if not isinstance(settings, Mapping) or set(settings) - set(DEFAULTS):
        raise ValueError('Unknown detector settings')
    cfg = {**DEFAULTS, **settings}
    if cfg['method'] not in ('hysteresis', 'watershed'):
        raise ValueError('Detector method must be hysteresis or watershed')
    for key in DEFAULTS.keys() - {'method'}:
        v = cfg[key]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not np.isfinite(v) or v < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
        cfg[key] = float(v)
    if cfg['high'] <= cfg['low'] or cfg['background_sigma_mm'] <= 0 or cfg['noise_floor_hu'] <= 0:
        raise ValueError('Require high > low, positive background sigma and noise floor')
    return cfg


def configurations():
    return [dict(configuration_id=f'det_{i:03d}', **validate_settings(dict(
        sigma_mm=sigma, high=high, low=low, min_area_mm2=area, method=method)))
        for i, (sigma, high, low, area, method) in enumerate(product(
            (0.6, 1.0), (1.0, 1.5, 2.0), (0.5, 0.75), (0.0, 3.0), ('hysteresis', 'watershed')))]


def prepare_detection(hu, liver_mask, spacing_mm, settings):
    cfg = validate_settings(settings)
    source, liver = np.asarray(hu), np.asarray(liver_mask)
    spacing = np.asarray(spacing_mm, dtype=float)
    if (source.ndim != 2 or source.shape != liver.shape or source.dtype.kind not in 'uif'
            or not np.isfinite(source).all() or not np.isin(liver, (0, 1)).all()):
        raise ValueError('Expected aligned finite 2D HU and binary liver arrays')
    if spacing.shape not in ((2,), (3,)) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('spacing_mm must contain two or three positive finite values')
    liver = liver.astype(bool)
    spacing = spacing[:2]
    # Normalize smoothing by liver support: outside-liver HU cannot create dark
    # edge artifacts or erase peripheral lesions.
    def masked_gaussian(image, sigma):
        denominator = ndi.gaussian_filter(liver.astype(float), sigma / spacing)
        numerator = ndi.gaussian_filter(np.where(liver, image, 0.0), sigma / spacing)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    smooth = masked_gaussian(source, cfg['sigma_mm'])
    background = masked_gaussian(smooth, cfg['background_sigma_mm'])
    values = smooth[liver]
    median = float(np.median(values)) if values.size else 0.0
    scale = max(cfg['noise_floor_hu'], 1.4826 * float(np.median(np.abs(values - median)))) if values.size else cfg['noise_floor_hu']
    score = np.where(liver, np.maximum(median - smooth, background - smooth) / scale, 0.0)
    gradients = [np.gradient(smooth, spacing[axis], axis=axis) if smooth.shape[axis] > 1
                 else np.zeros_like(smooth) for axis in (0, 1)]
    gradient = np.hypot(*gradients)
    return dict(smoothed_hu=smooth, score=score, liver=liver, spacing_mm=spacing,
                gradient=gradient, median_hu=median, scale_hu=scale,
                preparation={k: cfg[k] for k in ('sigma_mm', 'background_sigma_mm', 'noise_floor_hu')})


def detect_prepared(prepared, settings):
    """Apply a variant to cached HU maps; enforce cache/settings consistency."""
    cfg = validate_settings(settings)
    if any(cfg[k] != v for k, v in prepared['preparation'].items()):
        raise ValueError('Prepared maps do not match detector settings')
    liver, score = prepared['liver'], prepared['score']
    seeds = liver & (score > cfg['high'])
    candidates = liver & (score > cfg['low'])
    labels, _ = ndi.label(candidates, CONNECTIVITY)
    keep = np.zeros(int(labels.max()) + 1, dtype=bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    raw = keep[labels]
    fallback = []
    if cfg['method'] == 'watershed':
        regions, count = ndi.label(liver, CONNECTIVITY)
        for rid, box in enumerate(ndi.find_objects(regions), 1):
            region = regions[box] == rid
            foreground = seeds[box] & region
            background = region & ~candidates[box]
            if not foreground.any() or not background.any():
                fallback.append(dict(liver_component=rid, reason='missing_foreground' if not foreground.any() else 'missing_background'))
                continue
            markers = np.zeros(region.shape, dtype=np.int32)
            markers[background], markers[foreground] = 1, 2
            refined = watershed(prepared['gradient'][box], markers, connectivity=CONNECTIVITY, mask=region) == 2
            raw[box] = np.where(region, refined, raw[box])
    components, count = ndi.label(raw, CONNECTIVITY)
    pixel_area = float(np.prod(prepared['spacing_mm']))
    records, final = [], raw.copy()
    for cid, box in enumerate(ndi.find_objects(components), 1):
        component = components[box] == cid
        coords = np.argwhere(component) + np.array([s.start for s in box])
        area = int(component.sum()) * pixel_area
        discarded = area < cfg['min_area_mm2']
        if discarded:
            final[box][component] = False
        records.append(dict(component_id=cid, area_mm2=area, area_pixels=int(component.sum()),
            centroid_x_mm=float(coords[:, 0].mean() * prepared['spacing_mm'][0]),
            centroid_y_mm=float(coords[:, 1].mean() * prepared['spacing_mm'][1]),
            contrast_hu=float(prepared['median_hu'] - np.median(prepared['smoothed_hu'][box][component])),
            mean_score=float(score[box][component].mean()), retained=not discarded,
            discard_reason='area_below_minimum' if discarded else ''))
    return dict(mask=final, candidates=candidates, seeds=seeds, score=score,
                components=records, fallback=fallback,
                effective_parameters={**cfg, 'median_hu': prepared['median_hu'], 'scale_hu': prepared['scale_hu']})


def detect_slice(hu, liver_mask, spacing_mm, settings):
    return detect_prepared(prepare_detection(hu, liver_mask, spacing_mm, settings), settings)


def configured_prediction(hu, liver, spacing, cfg, legacy_settings=None, *, context=None):
    """Shared GUI/2D/3D dispatch, with historical 2D behavior preserved."""
    detector = cfg.get('detection', {})
    if detector.get('enabled', False):
        if detector.get('method') == 'hybrid_size':
            from .hybrid_segmentation import detect_hybrid
            return detect_hybrid(hu, liver, spacing,
                                 {k: v for k, v in detector.items() if k != 'enabled'})
        if str(detector.get('method', '')).startswith('local_'):
            if cfg.get('local_pipeline', 'raw_hu') == 'configured':
                from .local_pipeline import detect_configured_local
                return detect_configured_local(hu, liver, spacing, cfg,
                                               legacy_settings, context=context)
            if cfg.get('local_pipeline', 'raw_hu') != 'raw_hu':
                raise ValueError('Unknown local_pipeline')
            from .local_segmentation import detect_local
            return detect_local(hu, liver, spacing,
                                {k: v for k, v in detector.items() if k != 'enabled'},
                                context=context)
        return detect_slice(hu, liver, spacing, {k: v for k, v in detector.items() if k != 'enabled'})
    from .preprocessing import prepare_enhanced_slice
    from .segmentation import segment_slice
    image = prepare_enhanced_slice(hu, cfg, liver_mask=liver)
    return dict(mask=segment_slice(image, liver, legacy_settings or cfg.get('segmentation', {})))
