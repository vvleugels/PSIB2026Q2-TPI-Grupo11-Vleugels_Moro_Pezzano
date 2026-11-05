"""Controlled exploratory ablation on a completed run's frozen selection.

Does not change pipeline defaults or existing results. The configurations are
compared on the same reference-selected slices, not an independent test set.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import json
import numpy as np
import pandas as pd
import yaml
from psib.data_io import iter_case_slices
from psib.preprocessing import prepare_denoised_slice, enhance
from psib.segmentation import segment_slice
from scripts.run_segmentation import aggregate_metrics, read_selection


def audit(run_dir: Path, output: Path) -> None:
    cfg = yaml.safe_load((run_dir / 'effective_config.yaml').read_text())
    if json.loads((run_dir / 'run.json').read_text()).get('status') != 'completed':
        raise ValueError('Audit requires a completed run')
    selection = read_selection(run_dir / 'selected_cases_slices.csv')
    output.mkdir(parents=True, exist_ok=False)
    rows, intensities, variants = [], [], []
    for stage in ('original', 'without_equalization', 'gaussian_1_without_equalization'):
        for method in ('saved', 'otsu', 'statistical_0.5', 'statistical_1', 'statistical_1.5'):
            for cleanup in ('saved', 'opening_min25'):
                settings = deepcopy(cfg['segmentation'])
                if method == 'otsu':
                    settings['method'] = 'otsu'
                elif method.startswith('statistical_'):
                    settings.update(method='statistical', k=float(method.split('_')[1]), direction='lower')
                if cleanup != 'saved':
                    settings.update(opening=True, closing=False, fill_holes=False,
                                    footprint='disk', radius=1, min_area=25)
                variants.append(dict(configuration_id=f'{stage}__{method}__{cleanup}',
                                     stage=stage, settings=settings))
    pd.DataFrame(variants).assign(settings=lambda x: x.settings.map(json.dumps)).to_csv(output / 'variants.csv', index=False)
    selection.to_csv(output / 'selection.csv', index=False)
    (output / 'source_config.yaml').write_text(yaml.safe_dump(cfg))
    for case_id, group in selection.groupby('case_id'):
        for k, hu, labels, spacing in iter_case_slices(int(case_id), list(group.slice_index), cfg['data_dir']):
            liver, ref = labels > 0, labels == 2
            prepared = prepare_denoised_slice(hu, cfg)
            smoother = deepcopy(cfg)
            smoother['preprocessing']['denoising'] = dict(method='gaussian', gaussian_sigma=1.0)
            images = {'original': enhance(prepared, cfg['preprocessing'].get('enhancement', {})),
                      'without_equalization': prepared,
                      'gaussian_1_without_equalization': prepare_denoised_slice(hu, smoother)}
            for region, mask in (('tumor', ref), ('healthy_liver', labels == 1)):
                values = hu[mask]
                intensities.append(dict(case_id=case_id, slice_index=k, region=region,
                    n_pixels=len(values), median_hu=float(np.median(values)) if len(values) else np.nan,
                    q10_hu=float(np.quantile(values, .1)) if len(values) else np.nan,
                    q90_hu=float(np.quantile(values, .9)) if len(values) else np.nan))
            for variant in variants:
                pred = segment_slice(images[variant['stage']], liver, variant['settings'])
                rows.append(dict(configuration_id=variant['configuration_id'], case_id=case_id,
                    slice_index=k, pred_area=int(pred.sum()), reference_area=int(ref.sum()),
                    intersection=int((pred & ref).sum()), union=int((pred | ref).sum()),
                    spacing_x_mm=spacing[0], spacing_y_mm=spacing[1]))
        print(f'Audited case {case_id}', flush=True)
    slices = pd.DataFrame(rows)
    cases, summary = aggregate_metrics(slices)
    cases['pixel_precision'] = cases.intersection / cases.pred_area.replace(0, np.nan)
    cases['pixel_recall'] = cases.intersection / cases.reference_area.replace(0, np.nan)
    negatives = slices[slices.reference_area == 0].groupby('configuration_id').agg(
        negative_slices=('slice_index', 'size'),
        negative_slices_with_fp=('pred_area', lambda x: int((x > 0).sum())),
        fp_pixels_on_negative_slices=('pred_area', 'sum')).reset_index()
    summary = summary.merge(negatives, how='left', on='configuration_id').sort_values('dice_mean', ascending=False)
    for name, frame in [('per_slice', slices), ('per_case', cases), ('summary', summary),
                        ('intensities', pd.DataFrame(intensities))]:
        frame.to_csv(output / f'{name}.csv', index=False)
    (output / 'interpretation.txt').write_text(
        f'Source: {run_dir.resolve()}\nExploratory ablation; same reference-selected slices for all variants.\n'
        'Ranking is in-sample, not a validation estimate. Precision/recall are pixel-level.\n')
    print(summary[['configuration_id', 'dice_mean', 'iou_mean']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.output)
