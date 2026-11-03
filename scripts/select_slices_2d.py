"""Select cases and slices with only 2D mask reads; no 3D connected components."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from psib.config import load_config
from psib.data_io import iter_case_slices, list_cases
from psib.dataset import select_slice_records


def select_2d(config_path):
    cfg = load_config(config_path)
    settings = cfg.get('selection', {})
    available = list_cases(cfg['data_dir'])
    n_cases = int(settings.get('n_cases', 10))
    n_slices = int(settings.get('slices_per_case', 5))
    strategy = str(settings.get('slice_strategy', 'stratified'))
    gap_mm = float(settings.get('minimum_slice_gap_mm', 10.0))
    min_tumor = int(settings.get('min_tumor_pixels', 20))
    legacy_gap = int(settings.get('minimum_slice_gap', 2))
    seed = int(settings.get('seed', 2026))
    requested_ids = settings.get('case_ids')
    if not 3 <= n_slices <= 5:
        raise ValueError('Revisar n_cases y slices_per_case (3 a 5).')
    if requested_ids is None:
        if not 1 <= n_cases <= len(available):
            raise ValueError('n_cases debe estar entre 1 y la cantidad de casos disponibles.')
        chosen = sorted(np.random.default_rng(seed).choice(available, n_cases, replace=False).tolist())
        case_selection = 'seeded random sample'
    else:
        if (not isinstance(requested_ids, list) or not requested_ids
                or any(isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 0
                       for case_id in requested_ids)):
            raise ValueError('case_ids debe ser una lista no vacía de identificadores enteros no negativos.')
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError('case_ids no puede contener identificadores duplicados.')
        unavailable = sorted(set(requested_ids) - set(available))
        if unavailable:
            raise ValueError(f'Los casos solicitados no están disponibles en data_dir: {unavailable}.')
        chosen = list(requested_ids)
        n_cases = len(chosen)
        case_selection = 'explicitly selected case IDs'
    rows = []
    for rank, case_id in enumerate(chosen, 1):
        areas = []
        spacing_z = None
        for k, _ct, labels, spacing in iter_case_slices(case_id, data_dir=cfg['data_dir'], mask_only=True):
            areas.append((k, int(np.count_nonzero(labels == 2)), int(np.count_nonzero(labels > 0))))
            spacing_z = spacing[2]
        selected, shortages = select_slice_records(areas, n_slices, spacing_z or 1.0,
            strategy=strategy, minimum_gap_mm=gap_mm, min_tumor_pixels=min_tumor,
            legacy_minimum_gap=legacy_gap)
        if shortages:
            warnings.warn(f'Case {case_id}: ' + '; '.join(shortages), RuntimeWarning)
        gap_description = (f'minimum index gap={legacy_gap}' if strategy == 'area_top'
                           else f'minimum gap={gap_mm:g} mm')
        criteria = (f'2D-only reference selection; {case_selection}; strategy={strategy}; '
                    f'{gap_description}; min_tumor_pixels={min_tumor}; '
                    + ('shortage: ' + '; '.join(shortages) if shortages else 'all requested roles filled'))
        for row in selected:
            k = row['slice_index']
            rows.append({'selection_rank': rank, 'case_id': case_id, 'slice_index': k,
                         'seed': seed, 'selection_criteria':
                         criteria, 'slice_role': row['slice_role'],
                         'tumor_area_px': row['tumor_area_px'], 'liver_area_px': row['liver_area_px'],
                         'slice_position_mm': k * (spacing_z or 1.0)})
        print(f'Selección 2D: caso {case_id}, cortes {[r["slice_index"] for r in selected]}', flush=True)
    path = Path(cfg['_output_dir']) / 'selected_cases_slices.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'Saved {len(rows)} selected slices to {path}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    args = parser.parse_args()
    select_2d(args.config)


if __name__ == '__main__':
    main()
