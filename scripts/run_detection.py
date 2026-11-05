"""Run experimental HU detection; optionally execute frozen 10-patient nested validation."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from psib.config import load_config
from psib.data_io import list_cases, load_case, load_segmentation, iter_case_slices
from psib.dataset import select_slice_records
from psib.detection import configurations, validate_settings, prepare_detection, detect_prepared, configured_prediction
from psib.detection_validation import make_folds, nested_evaluation, summarize, select_variant
from psib.lesion_eval import prepare_reference, match_lesions_one_to_one
from psib.preprocessing import prepare_denoised_slice
from psib.segmentation import segment_slice
from scripts.run_segmentation import aggregate_metrics, read_selection, _metadata, _write_status, export_slice_comparison


def freeze_selection(cfg, out):
    cases = list_cases(cfg['data_dir'])
    folds = make_folds(cases)
    (out / 'folds.json').write_text(json.dumps(folds, indent=2))
    rows = []
    for case in cases:
        labels = load_segmentation(case, cfg['data_dir'])
        _, _, _, spacing = next(iter_case_slices(case, [0], cfg['data_dir'], mask_only=True))
        stats = [(k, int((labels[:, :, k] == 2).sum()), int((labels[:, :, k] > 0).sum()))
                 for k in range(labels.shape[2])]
        selected, shortages = select_slice_records(stats, 5, spacing[2], strategy='stratified',
                                                   minimum_gap_mm=10, min_tumor_pixels=20)
        if not selected:
            raise ValueError(f'Case {case} has no eligible liver slices')
        rows.extend(dict(case_id=case, **r, selection_criteria='reference-stratified; gap=10 mm; '+ '; '.join(shortages)) for r in selected)
        print(f'Frozen selection: case {case}, {len(selected)} slices', flush=True)
    selection = pd.DataFrame(rows)
    selection.to_csv(out / 'selected_cases_slices.csv', index=False)
    return selection, folds


def append_table(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)


def run_experiment(config_path, run_id, sweep=False, *, benchmark=False, historical_run=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    cfg = load_config(config_path)
    out = Path(cfg['_output_dir']) / 'segmentation' / run_id
    out.mkdir(parents=True, exist_ok=False)
    state = _metadata(config_path, run_id, sweep)
    state['pipeline'] = 'experimental_hu_detection_v1'
    state['arguments'].update(benchmark=benchmark, historical_run=str(historical_run) if historical_run else None)
    _write_status(out, state)
    try:
        if benchmark:
            if historical_run is None:
                raise ValueError('Benchmark requires --historical-run')
            history = yaml.safe_load((Path(historical_run) / 'effective_config.yaml').read_text())
            state['historical_run'] = str(Path(historical_run).resolve())
            (out / 'historical_config.yaml').write_text(yaml.safe_dump(history))
            selection, folds = freeze_selection(cfg, out)
        else:
            selection = read_selection(Path(cfg['_output_dir']) / 'selected_cases_slices.csv')
            selection.to_csv(out / 'selected_cases_slices.csv', index=False)
        effective = {k: v for k, v in cfg.items() if not k.startswith('_')}
        if benchmark:
            effective['selection'] = {**effective.get('selection', {}), 'n_cases': 10, 'case_ids': sorted(map(int, selection.case_id.unique())), 'slices_per_case': 5, 'slice_strategy': 'stratified', 'minimum_slice_gap_mm': 10, 'min_tumor_pixels': 20}
        effective['detection'] = dict(enabled=True, **validate_settings({k:v for k,v in cfg.get('detection', {}).items() if k != 'enabled'}))
        (out / 'effective_config.yaml').write_text(yaml.safe_dump(effective, sort_keys=False))
        variants = configurations() if sweep or benchmark else [dict(configuration_id='det_000', **validate_settings(
            {k:v for k,v in cfg.get('detection', {}).items() if k != 'enabled'}))]
        configs = [dict(configuration_id=v['configuration_id'], method=v['method'], detector_settings=json.dumps(
            {k: val for k, val in v.items() if k != 'configuration_id'})) for v in variants]
        if benchmark:
            otsu_config = deepcopy(history)
            otsu_config['detection'] = {'enabled': False}
            otsu_config['preprocessing']['denoising'] = dict(method='gaussian', gaussian_sigma=1.0)
            otsu_config['preprocessing']['enhancement'] = dict(method='none')
            otsu_config['segmentation'] = dict(method='otsu', opening=True, radius=1, min_area=25)
            configs.extend([dict(configuration_id='historical', method='historical', pipeline_config=json.dumps(history)),
                            dict(configuration_id='otsu', method='otsu', pipeline_config=json.dumps(otsu_config))])
        pd.DataFrame(configs).to_csv(out / 'configurations.csv', index=False)
        rows, detection_rows = [], []
        for case, group in selection.groupby('case_id', sort=True):
            # One sequential decompression per volume, avoiding repeated gzip seeks.
            ct, seg, spacing = load_case(int(case), cfg['data_dir'])
            for k in group.slice_index.astype(int):
                hu, labels = ct[:, :, k], seg[:, :, k]
                liver, reference = labels > 0, labels == 2
                ref_info = prepare_reference(reference, spacing[:2])
                cache = {}
                predictions = []
                for variant in variants:
                    cid = variant['configuration_id']
                    settings = {key:value for key,value in variant.items() if key != 'configuration_id'}
                    key = tuple(settings[q] for q in ('sigma_mm','background_sigma_mm','noise_floor_hu'))
                    if key not in cache:
                        cache[key] = prepare_detection(hu, liver, spacing, settings)
                    result = detect_prepared(cache[key], settings)
                    predictions.append((cid, result['mask']))
                    append_table(out / 'components.csv', [dict(configuration_id=cid, case_id=case, slice_index=k, **c) for c in result['components']])
                    append_table(out / 'effective_parameters.csv', [dict(configuration_id=cid, case_id=case, slice_index=k,
                        **result['effective_parameters'], fallback=json.dumps(result['fallback']))])
                    if not sweep and not benchmark:
                        directory = out / 'diagnostics' / cid
                        directory.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(directory / f'case_{case:03d}_slice_{k:03d}.npz',
                            seeds=result['seeds'], candidates=result['candidates'], score=result['score'],
                            smoothed_hu=cache[key]['smoothed_hu'])
                if benchmark:
                    predictions.append(('historical', configured_prediction(hu, liver, spacing, {**history, 'detection': {'enabled':False}})['mask']))
                    otsu_cfg = deepcopy(history)
                    otsu_cfg['preprocessing']['denoising'] = dict(method='gaussian', gaussian_sigma=1.0)
                    predictions.append(('otsu', segment_slice(prepare_denoised_slice(hu, otsu_cfg, liver_mask=liver), liver,
                        dict(method='otsu', opening=True, radius=1, min_area=25))))
                for cid, pred in predictions:
                    directory = out / 'predictions' / cid
                    directory.mkdir(parents=True, exist_ok=True)
                    np.save(directory / f'case_{case:03d}_slice_{k:03d}.npy', pred, allow_pickle=False)
                    count = dict(configuration_id=cid, case_id=case, slice_index=k,
                        pred_area=int(pred.sum()), reference_area=int(reference.sum()),
                        intersection=int((pred & reference).sum()), union=int((pred | reference).sum()))
                    total = count['pred_area'] + count['reference_area']
                    count.update(dice=2*count['intersection']/total if total else 1,
                                 iou=count['intersection']/count['union'] if count['union'] else 1,
                                 reference_coverage=count['intersection']/count['reference_area'] if count['reference_area'] else np.nan)
                    rows.append(count)
                    for threshold in (.10, .25):
                        matches = match_lesions_one_to_one(pred, reference, spacing[:2], threshold, prepared_reference=ref_info)
                        small = matches[matches.size_class == '<10']
                        detection_rows.append(dict(**count, min_iou=threshold,
                            TP=int((matches.status=='TP').sum()), FP=int((matches.status=='FP').sum()), FN=int((matches.status=='FN').sum()),
                            small_TP=int((small.status=='TP').sum()), small_FN=int((small.status=='FN').sum())))
                    if not sweep and not benchmark:
                        from psib.preprocessing import apply_hu_window
                        window = cfg.get('preprocessing', {}).get('hu_window', {})
                        display = apply_hu_window(hu, window.get('center',30), window.get('width',150))
                        # A single-variant run has exactly one prepared map. Show
                        # the HU actually analyzed, instead of duplicating input.
                        analyzed = next(iter(cache.values()))['smoothed_hu']
                        processed = apply_hu_window(np.where(liver, analyzed, hu),
                            window.get('center',30), window.get('width',150))
                        export_slice_comparison(out, cid, case, k, display, processed, 'none', None, reference, pred,
                            processed_title='Smoothed HU in liver (display window)')
                print(f'Evaluated case {case}, slice {k}', flush=True)
            # A slice is a view: release it before loading the next volume, too.
            del ct, seg, cache, predictions, hu, labels, liver, reference, ref_info
        slices = pd.DataFrame(rows)
        cases, summary = aggregate_metrics(slices)
        for name, frame in [('metrics_per_slice',slices),('metrics_per_case',cases),('summary',summary),
                            ('detection_per_slice',pd.DataFrame(detection_rows))]:
            frame.to_csv(out / f'{name}.csv', index=False)
        exploratory = pd.DataFrame([dict(configuration_id=cid, min_iou=iou, **summarize(group))
            for (cid, iou), group in pd.DataFrame(detection_rows).groupby(['configuration_id', 'min_iou'])])
        exploratory.to_csv(out / 'exploratory_detection_summary.csv', index=False)
        if not benchmark:
            primary = pd.DataFrame(detection_rows).query('min_iou == 0.10')
            winner = select_variant(primary, 8)
            (out / 'best_selection.json').write_text(json.dumps(dict(configuration_id=winner, budget=8,
                status='selected' if winner else 'no_feasible_configuration', interpretation='in-sample exploratory selection'), indent=2))
            if winner:
                best = summary[summary.configuration_id == winner].copy()
                best['selection_metric'] = 'in_sample_lesion_recall_at_8_fp_per_slice'
                best.to_csv(out / 'best_configurations.csv', index=False)
        if benchmark:
            chosen, outer, validation, per_case, inner, decision = nested_evaluation(pd.DataFrame(detection_rows), folds)
            for name, frame in [('selected_configurations',chosen),('outer_per_slice',outer),('validation_summary',validation),
                                ('validation_per_case',per_case),('inner_validation',inner)]:
                frame.to_csv(out / f'{name}.csv', index=False)
            (out / 'promotion.json').write_text(json.dumps(decision, indent=2))
            plot_froc(validation, out)
            write_validation_report(validation, chosen, decision, out)
        state.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(), n_cases=int(selection.case_id.nunique()), n_slices=len(selection), n_configurations=len(configs))
        _write_status(out, state)
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _write_status(out, state)
        raise
    print(f'Detection results: {out}', flush=True)
    return out


def plot_froc(summary, out):
    fig, ax = plt.subplots(figsize=(7,5))
    for family, group in summary[summary.min_iou == .1].groupby('family'):
        g = group.sort_values('budget')
        unique = g.drop_duplicates(['fp_per_slice', 'recall'])
        label = family
        choices_path = out / 'selected_configurations.csv'
        unavailable = family == 'detector' and choices_path.exists() and pd.read_csv(choices_path).configuration_id.isna().all()
        if unavailable:
            label = 'detector: no feasible configuration (abstention)'
        ax.plot(unique.fp_per_slice, unique.recall, 'o-', label=label)
        if family == 'detector' and not unavailable:
            for row in unique.itertuples():
                ax.annotate(f'budget {row.budget}', (row.fp_per_slice,row.recall), fontsize=8)
    ax.set(xlabel='False positives / slice (outer patients)', ylabel='Lesion sensitivity (IoU ≥ 0.10)', ylim=(0,1), title='Internal patient-disjoint evaluation')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / 'froc.png', dpi=150); plt.close(fig)


def write_validation_report(summary, chosen, decision, out):
    lines = ["# Evaluación interna del detector clásico", "",
        f"Diez pacientes, {int(summary.iloc[0].n_slices)} cortes seleccionados por referencia; cinco folds externos.",
        "Parámetros elegidos en pacientes internos con prioridad de sensibilidad. No es validación externa.",
        "", "## Resultado al presupuesto interno de 8 FP/corte", "",
        "| Método | IoU mínimo | Sensibilidad | Sensibilidad <10 mm | FP/corte | Precisión | Dice medio |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary[summary.budget == 8].itertuples():
        lines.append(f"| {row.family} | {row.min_iou:.2f} | {row.recall:.3f} | {row.small_recall:.3f} | {row.fp_per_slice:.2f} | {row.precision:.3f} | {row.dice_mean:.3f} |")
    lines += ["", "## Decisión", "", str(decision), "",
        "Los presupuestos se aplican al ajuste interno; la tasa externa se informa tal como resulta.",
        "Las configuraciones no se vuelven a elegir con los pacientes externos.", "",
        "Si no existe configuración factible, recall=0 corresponde a abstención del selector, no al recall bruto de las 48 variantes.",
        "Ver exploratory_detection_summary.csv para resultados de cada variante fija sobre todos los casos (exploratorios).", "",
        "## Selecciones al presupuesto 8", ""]
    for row in chosen[chosen.budget == 8].itertuples():
        lines.append(f"- Fold {row.fold}: {row.configuration_id}, {row.status}.")
    lines += ["", "![Sensibilidad y falsos positivos](froc.png)", "",
        "Ver validation_per_case.csv para heterogeneidad por paciente y outer_per_slice.csv para cada corte."]
    (out / 'validation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config.yaml')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--historical-run', type=Path)
    args = parser.parse_args()
    run_experiment(args.config, args.run_id, args.sweep, benchmark=args.benchmark, historical_run=args.historical_run)
