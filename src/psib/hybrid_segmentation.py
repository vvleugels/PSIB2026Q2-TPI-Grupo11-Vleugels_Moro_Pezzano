"""Reference-free size-gated fusion of a classical mask and HU candidates."""
from copy import deepcopy
from collections.abc import Mapping
from itertools import product
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from .detection import prepare_detection, detect_prepared, validate_settings as validate_hu
from .preprocessing import prepare_enhanced_slice
from .segmentation import segment_slice, _validate_settings

CONNECTIVITY = np.ones((3, 3), dtype=bool)


def validate_settings(settings):
    allowed = {'method', 'base_config', 'hu_settings', 'max_diameter_mm', 'addition_rule'}
    if not isinstance(settings, Mapping) or set(settings) - allowed:
        raise ValueError('Unknown hybrid settings')
    cfg = dict(method='hybrid_size', max_diameter_mm=10., addition_rule='disjoint', **{
        k: v for k, v in settings.items() if k not in ('method', 'max_diameter_mm', 'addition_rule')})
    cfg.update(settings)
    if cfg['method'] != 'hybrid_size' or cfg['addition_rule'] not in ('complete', 'disjoint'):
        raise ValueError('Invalid hybrid method or addition rule')
    limit = cfg['max_diameter_mm']
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, float)) or not np.isfinite(limit) or limit <= 0:
        raise ValueError('max_diameter_mm must be finite and positive')
    base = cfg.get('base_config')
    if not isinstance(base, Mapping) or set(base) != {'preprocessing', 'segmentation'}:
        raise ValueError('base_config requires preprocessing and segmentation only')
    cfg['base_config'] = deepcopy(base)
    cfg['base_config']['segmentation'] = _validate_settings(base['segmentation'])
    cfg['hu_settings'] = validate_hu(cfg.get('hu_settings', dict(method='watershed', high=1.5, low=.5)))
    if cfg['hu_settings']['method'] != 'watershed' or cfg['hu_settings']['min_area_mm2'] != 0:
        raise ValueError('Hybrid candidates require watershed and no area removal')
    return cfg


def prepare_components(candidate_mask, liver, spacing):
    candidate, liver = np.asarray(candidate_mask), np.asarray(liver)
    spacing = np.asarray(spacing, dtype=float)
    if (candidate.ndim != 2 or candidate.shape != liver.shape or
            not np.isin(candidate, (0, 1)).all() or not np.isin(liver, (0, 1)).all()):
        raise ValueError('Expected aligned binary 2D masks')
    if spacing.shape not in ((2,), (3,)) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Expected positive finite spacing')
    labels, count = ndi.label(candidate.astype(bool) & liver.astype(bool), CONNECTIVITY)
    diameters = np.zeros(count + 1)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    for region in regionprops(labels, spacing=tuple(spacing[:2])):
        diameters[region.label] = region.feret_diameter_max
    return labels, diameters, areas, float(np.prod(spacing[:2]))


def combine_prepared(base_mask, liver, prepared, max_diameter_mm, addition_rule):
    base, liver = np.asarray(base_mask), np.asarray(liver)
    labels, diameters, areas, pixel_area = prepared
    if (base.shape != labels.shape or base.shape != liver.shape or
            not np.isin(base, (0, 1)).all() or not np.isin(liver, (0, 1)).all()):
        raise ValueError('Expected aligned binary base and liver masks')
    if addition_rule not in ('complete', 'disjoint') or not np.isfinite(max_diameter_mm) or max_diameter_mm <= 0:
        raise ValueError('Invalid fusion rule or diameter')
    liver = liver.astype(bool)
    base = base.astype(bool) & liver
    touching = np.zeros(len(diameters), dtype=bool)
    touching[np.unique(labels[ndi.binary_dilation(base, structure=CONNECTIVITY)])] = True
    keep = (diameters < max_diameter_mm)
    if addition_rule == 'disjoint':
        keep &= ~touching
    keep[0] = False
    addition = keep[labels] & liver
    decisions = [dict(component_id=i, diameter_mm=float(diameters[i]),
                      area_mm2=float(areas[i]*pixel_area), touches_base=bool(touching[i]),
                      retained=bool(keep[i]), reason='retained' if keep[i] else
                      'diameter_at_or_above_limit' if diameters[i] >= max_diameter_mm else 'touches_base')
                 for i in range(1, len(diameters))]
    return dict(mask=base | addition, base_mask=base, candidate_mask=(labels > 0) & liver,
                added_mask=addition & ~base, components=decisions)


def combine_masks(base, candidates, liver, spacing, max_diameter_mm=10., addition_rule='disjoint'):
    return combine_prepared(base, liver, prepare_components(candidates, liver, spacing),
                            max_diameter_mm, addition_rule)


def detect_hybrid(hu, liver, spacing, settings):
    cfg = validate_settings(settings)
    prepared = prepare_detection(hu, liver, spacing, cfg['hu_settings'])
    candidate = detect_prepared(prepared, cfg['hu_settings'])
    base_cfg = cfg['base_config']
    base = segment_slice(prepare_enhanced_slice(hu, base_cfg, liver_mask=liver), liver, base_cfg['segmentation'])
    result = combine_masks(base, candidate['mask'], liver, spacing, cfg['max_diameter_mm'], cfg['addition_rule'])
    result.update(seeds=candidate['seeds'], candidates=candidate['mask'], score=candidate['score'],
                  fallback=candidate['fallback'], effective_parameters=cfg,
                  hu_effective_parameters=candidate['effective_parameters'], failed_candidates=0,
                  context_note='Axial hybrid; no neighboring-slice context')
    return result


def configurations(controls):
    records = []
    for limit, rule in product((10., 20., 30.), ('complete', 'disjoint')):
        records.append(dict(configuration_id=f'h1_{len(records):03d}', stage=1, base='otsu_059',
                            settings=dict(method='hybrid_size', base_config={k: deepcopy(controls['otsu_059'][k]) for k in ('preprocessing', 'segmentation')},
                                          hu_settings={k:v for k,v in controls['hu_sensitive']['detection'].items() if k != 'enabled'},
                                          max_diameter_mm=limit, addition_rule=rule)))
    for i, (base, sigma, high, low, limit, rule) in enumerate(product(
            ('default', 'otsu_059'), (.6, 1.), (1., 1.5, 2.), (.33, .5), (10., 20., 30.), ('complete', 'disjoint'))):
        records.append(dict(configuration_id=f'h2_{i:03d}', stage=2, base=base,
            settings=dict(method='hybrid_size', base_config={k:deepcopy(controls[base][k]) for k in ('preprocessing', 'segmentation')},
                          hu_settings=dict(method='watershed', sigma_mm=sigma, high=high, low=low,
                                           background_sigma_mm=10., noise_floor_hu=5., min_area_mm2=0.),
                          max_diameter_mm=limit, addition_rule=rule)))
    return records
