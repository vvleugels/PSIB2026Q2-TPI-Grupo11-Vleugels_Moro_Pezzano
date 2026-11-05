"""Configurable input preprocessing and output filtering for local detectors."""
import numpy as np

from .local_segmentation import detect_local
from .preprocessing import prepare_enhanced_slice
from .segmentation import postprocess_mask


def detect_configured_local(hu, liver, spacing, cfg, settings=None, *, context=None):
    window = cfg.get('preprocessing', {}).get('hu_window', {})

    def prepare(source, support):
        image = prepare_enhanced_slice(source, cfg, liver_mask=support)
        if window.get('enabled', True):
            scale = float(window.get('width', 400))
            lower = float(window.get('center', 60)) - scale / 2
        else:
            values = np.asarray(source)[np.asarray(support, bool)]
            scale = float(np.ptp(values)) if values.size else 0.
            lower = float(values.min()) if values.size else 0.
        # Restore the intensity scale: a 5 HU noise floor must not be applied
        # directly to [0, 1]. With nonlinear enhancement these are equivalent
        # intensity units, not measured HU.
        return image, image * scale + lower

    processed, source = prepare(hu, liver)
    neighbors = None if context is None else [
        (offset, prepare(image, support)[1], support)
        for offset, image, support in context]
    result = detect_local(source, liver, spacing,
                          {k: v for k, v in cfg['detection'].items() if k != 'enabled'},
                          context=neighbors, preprocessed=True)
    result['raw_mask'] = result['mask'].copy()
    result['mask'] = postprocess_mask(result['mask'], liver,
                                      settings if settings is not None else cfg.get('segmentation', {}))
    result['processed_image'] = processed
    result['pipeline_note'] = 'configured preprocessing; local smoothing bypassed; configured morphology'
    return result
