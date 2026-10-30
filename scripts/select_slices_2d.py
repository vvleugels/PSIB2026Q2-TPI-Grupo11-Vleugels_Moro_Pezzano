"""Select cases and slices with only 2D mask reads; no 3D connected components."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from psib.config import load_config
from psib.data_io import iter_case_slices, list_cases


def select_2d(config_path):
    cfg = load_config(config_path)
    settings = cfg.get('selection', {})
    available = list_cases(cfg['data_dir'])
    n_cases = int(settings.get('n_cases', 10))
    n_slices = int(settings.get('slices_per_case', 5))
    gap = int(settings.get('minimum_slice_gap', 2))
    seed = int(settings.get('seed', 2026))
    if not 1 <= n_cases <= len(available) or not 3 <= n_slices <= 5 or gap < 0:
        raise ValueError('Revisar n_cases, slices_per_case (3 a 5) y minimum_slice_gap.')
    chosen = sorted(np.random.default_rng(seed).choice(available, n_cases, replace=False).tolist())
    rows = []
    for rank, case_id in enumerate(chosen, 1):
        areas = []
        for k, _, labels, _ in iter_case_slices(case_id, data_dir=cfg['data_dir'], mask_only=True):
            areas.append((k, int(np.count_nonzero(labels == 2)), int(np.count_nonzero(labels > 0))))
        ranked = sorted(areas, key=lambda row: (-row[1], -row[2], row[0]))
        selected = []
        for k, _, _ in ranked:
            if all(abs(k - previous) >= gap for previous in selected):
                selected.append(k)
                if len(selected) == n_slices:
                    break
        for k in sorted(selected):
            rows.append({'selection_rank': rank, 'case_id': case_id, 'slice_index': k,
                         'seed': seed, 'selection_criteria':
                         '2D-only reference selection; seeded cases; highest tumor area; '
                         f'minimum gap={gap}; no 3D inventory or burden stratification'})
        print(f'Selección 2D: caso {case_id}, cortes {sorted(selected)}', flush=True)
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
