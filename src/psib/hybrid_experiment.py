"""Frozen, patient-disjoint evaluation of size-gated hybrid masks.

References are used only here, after reference-free prediction has finished.
"""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import platform
import shutil
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi
from .hybrid_segmentation import configurations, prepare_components, combine_prepared
from .detection import configured_prediction, prepare_detection, detect_prepared
from .classical_search import score_mask, validate_folds
from .lesion_eval import prepare_reference, label_prediction_ious, _optimal_pairs
from .local_experiment import (atomic_json, digest, seal, verify_seal, safe_records,
                               summarize, size_category, DEPENDENCIES)

CONTROLS = ('default', 'otsu_059', 'hu_sensitive', 'bilateral', 'union_full')
PROCEDURES = ('stage1', 'stage2', 'combined')
CRITERIA = (.1, .25)
PROTOCOL = dict(schema=1, dice_floor=.4, match_iou=list(CRITERIA),
    selection='recall, small recall, Dice, lower false area mm2, lower FP/slice, stable ID',
    fallback='otsu_059', stages=[6, 144],
    adaptation='Hepatic HU median/MAD per slice; predicted component Feret in physical mm',
    interpretation='Previously explored, reference-selected 2D cohort; no independent validation')


def folder_for(root, row):
    return Path(root)/f'case_{int(row.case_id):03d}'/f'slice_{int(row.slice_index):04d}'


def unpack(data, cid):
    ids = list(data['ids'])
    if len(ids) != len(set(ids)) or len(ids) != len(data['predictions']):
        raise ValueError('Invalid packed mask IDs')
    shape = tuple(data['shape'])
    return np.unpackbits(data['predictions'][ids.index(cid)], count=int(np.prod(shape))).reshape(shape).astype(bool)


def read_inputs(folder):
    with np.load(folder/'inputs.npz', allow_pickle=False) as data:
        return {k: data[k] for k in ('hu', 'liver', 'reference', 'spacing')}


def preflight(root, source, bilateral):
    root, source, bilateral = map(Path, (root, source, bilateral))
    selection = pd.read_csv(source/'selection.csv')
    frozen = pd.read_csv(root/'experiments/local_candidates/selection.csv')
    pd.testing.assert_frame_equal(selection, frozen)
    if len(selection) != 50 or selection.duplicated(['case_id', 'slice_index']).any() or not (selection.groupby('case_id').size() == 5).all():
        raise ValueError('Expected 50 frozen slices in ten patients')
    folds = json.loads((source/'folds.json').read_text())
    if folds != json.loads((root/'experiments/local_candidates/folds.json').read_text()):
        raise ValueError('Frozen folds differ')
    validate_folds(folds, selection.case_id.unique())
    controls = json.loads((source/'controls.json').read_text())
    if controls != json.loads((root/'experiments/local_candidates/controls.json').read_text()):
        raise ValueError('Frozen controls differ')
    manifest = json.loads((source/'manifest.json').read_text())
    if controls != manifest['controls'] or folds != manifest['folds'] or safe_records(selection) != manifest['selection']:
        raise ValueError('Source manifest differs from protocol')
    for run in (source, bilateral):
        if json.loads((run/'run.json').read_text())['status'] != 'completed':
            raise ValueError(f'Incomplete source: {run}')
    pd.testing.assert_frame_equal(pd.read_csv(bilateral/'selected_cases_slices.csv'), selection)
    bilateral_cfg = yaml.safe_load((bilateral/'effective_config.yaml').read_text())
    if bilateral_cfg.get('preprocessing_scope') != 'liver' or bilateral_cfg['detection']['enabled']:
        raise ValueError('Expected liver-only classical bilateral control')
    if (bilateral_cfg['preprocessing']['denoising'] != dict(method='bilateral', bilateral_sigma_color=.5, bilateral_sigma_spatial=10.)
            or bilateral_cfg['preprocessing']['enhancement']['method'] != 'none'
            or bilateral_cfg['preprocessing']['enhancement']['contrast'] != .9
            or bilateral_cfg['preprocessing']['hu_window']['center'] != 30
            or bilateral_cfg['preprocessing']['hu_window']['width'] != 150):
        raise ValueError('Unexpected frozen bilateral parameters')
    from .segmentation import _validate_settings
    if _validate_settings(bilateral_cfg['segmentation']) != _validate_settings(controls['otsu_059']['segmentation']):
        raise ValueError('Bilateral cleanup differs from frozen Otsu control')
    hashes = {}
    for row in selection.itertuples():
        folder = folder_for(source, row)
        if not verify_seal(folder):
            raise ValueError(f'Incomplete source slice: {folder}')
        data = read_inputs(folder)
        hu, liver, reference, spacing = (data[k] for k in ('hu', 'liver', 'reference', 'spacing'))
        if (hu.ndim != 2 or hu.shape != liver.shape or hu.shape != reference.shape or
                not np.isfinite(hu).all() or not np.isin(liver, (0, 1)).all() or
                not np.isin(reference, (0, 1)).all() or np.any(reference & ~liver)):
            raise ValueError('Invalid frozen image/mask geometry')
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError('Invalid frozen spacing')
        if int(reference.sum()) != int(row.tumor_area_px) or int(liver.sum()) != int(row.liver_area_px):
            raise ValueError('Frozen selection areas differ')
        with np.load(folder/'masks.npz', allow_pickle=False) as masks:
            if tuple(masks['shape']) != hu.shape:
                raise ValueError('Source mask geometry differs')
            for cid in ('default', 'otsu_059', 'hu_sensitive'):
                if np.any(unpack(masks, cid) & ~liver):
                    raise ValueError('Source prediction escapes liver')
        bp = bilateral/'predictions/cfg_000'/f'case_{row.case_id:03d}_slice_{row.slice_index:03d}.npy'
        mask = np.load(bp, allow_pickle=False)
        if mask.shape != hu.shape or not np.isin(mask, (0, 1)).all() or np.any(mask & ~liver):
            raise ValueError('Invalid bilateral mask')
        hashes[f'{row.case_id}:{row.slice_index}'] = dict(source_seal=digest(folder/'complete.json'), bilateral=digest(bp))
    payload = dict(protocol=PROTOCOL, configurations=configurations(controls), controls=controls,
        bilateral_config={k:bilateral_cfg[k] for k in ('preprocessing', 'segmentation')},
        selection=safe_records(selection), folds=folds, sources=hashes,
        source_manifest_sha256=digest(source/'manifest.json'), python=platform.python_version(),
        dependencies={name:version(name) for name in DEPENDENCIES},
        code={str(p.relative_to(root)):digest(p) for p in sorted((root/'src/psib').glob('*.py'))} |
             {'scripts/run_hybrid_search.py':digest(root/'scripts/run_hybrid_search.py')})
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return selection, folds, controls, bilateral_cfg, signature, payload


def evaluate_mask(mask, reference, spacing, prepared):
    rows = score_mask(mask, prepared, CRITERIA)
    _, ious = label_prediction_ious(mask, prepared[0], prepared[1])
    overlap = ious > 0
    fusion = int((overlap.sum(axis=0) > 1).sum())
    split = int((overlap.sum(axis=1) > 1).sum())
    lesions = []
    for row in rows:
        criterion = row['min_iou']
        matches = {r+1:p+1 for r,p in _optimal_pairs(ious, overlap & (ious >= criterion))}
        row.update(pixel_area_mm2=float(np.prod(spacing[:2])), failed_candidates=0,
                   fusion_components=fusion, split_references=split,
                   positive_without_detection=int(reference.any() and not matches))
        for rid, diameter in prepared[2].items():
            pid = matches.get(rid, 0)
            lesions.append(dict(min_iou=criterion, reference_id=int(rid), predicted_id=pid,
                diameter_mm=float(diameter), size=size_category(diameter), detected=bool(pid),
                fusion=bool(pid and overlap[:, pid-1].sum() > 1),
                split=bool(overlap[rid-1].sum() > 1)))
    return rows, lesions


def process_slice(source, bilateral, out, row, controls, bilateral_cfg, records):
    source_folder = folder_for(source, row)
    target = folder_for(out, row)
    if verify_seal(target):
        return 'cached'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.'+target.name+'.partial')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    data = read_inputs(source_folder)
    hu, liver, reference, spacing = (data[k] for k in ('hu', 'liver', 'reference', 'spacing'))
    with np.load(source_folder/'masks.npz', allow_pickle=False) as saved:
        base_masks = {cid:unpack(saved, cid) for cid in ('default', 'otsu_059', 'hu_sensitive')}
    bp = Path(bilateral)/'predictions/cfg_000'/f'case_{row.case_id:03d}_slice_{row.slice_index:03d}.npy'
    base_masks['bilateral'] = np.load(bp, allow_pickle=False).astype(bool)
    # Reproduce every reused control: seals alone do not prove current-code equivalence.
    for cid in ('default', 'otsu_059', 'hu_sensitive', 'bilateral'):
        cfg = bilateral_cfg if cid == 'bilateral' else controls[cid]
        actual = configured_prediction(hu, liver, spacing, cfg)['mask']
        if not np.array_equal(actual, base_masks[cid]):
            raise ValueError(f'Reused control differs: {cid}, case {row.case_id}, slice {row.slice_index}')
    base_masks['union_full'] = base_masks['otsu_059'] | base_masks['hu_sensitive']
    reference_prepared = prepare_reference(reference, spacing[:2])
    packed, ids, metrics, decisions, hu_parameters = [], [], [], {}, {}
    maps_cache, components_cache, candidates_cache = {}, {}, {}
    def store_mask(cid, mask):
        ids.append(cid); packed.append(np.packbits(mask))
        for entry in score_mask(mask, reference_prepared, CRITERIA):
            metrics.append(dict(configuration_id=cid, case_id=int(row.case_id), slice_index=int(row.slice_index),
                                pixel_area_mm2=float(np.prod(spacing[:2])), failed_candidates=0, **entry))
    for cid, mask in base_masks.items():
        store_mask(cid, mask)
    for record in records:
        cid, cfg = record['configuration_id'], record['settings']
        h = cfg['hu_settings']; key = json.dumps(h, sort_keys=True)
        if key not in components_cache:
            prep_key = (h['sigma_mm'], h['background_sigma_mm'], h['noise_floor_hu'])
            if prep_key not in maps_cache:
                maps_cache[prep_key] = prepare_detection(hu, liver, spacing, h)
            detected = detect_prepared(maps_cache[prep_key], h)
            if record['stage'] == 1 and not np.array_equal(detected['mask'], base_masks['hu_sensitive']):
                raise ValueError('Stage-one HU mask is not the frozen control')
            candidate = base_masks['hu_sensitive'] if record['stage'] == 1 else detected['mask']
            components_cache[key] = prepare_components(candidate, liver, spacing)
            candidates_cache[key] = np.packbits(candidate)
            hu_parameters[key] = dict(effective_parameters=detected['effective_parameters'], fallback=detected['fallback'])
        result = combine_prepared(base_masks[record['base']], liver, components_cache[key],
                                  cfg['max_diameter_mm'], cfg['addition_rule'])
        decisions[cid] = result['components']
        store_mask(cid, result['mask'])
    np.savez_compressed(temporary/'inputs.npz', **data)
    np.savez_compressed(temporary/'masks.npz', ids=np.array(ids), shape=hu.shape, predictions=np.stack(packed))
    np.savez_compressed(temporary/'hu_masks.npz', ids=np.array(list(candidates_cache)), shape=hu.shape,
                        predictions=np.stack(list(candidates_cache.values())))
    atomic_json(temporary/'decisions.json', decisions)
    atomic_json(temporary/'hu_parameters.json', hu_parameters)
    pd.DataFrame(metrics).to_csv(temporary/'metrics.csv', index=False)
    seal(temporary)
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    return 'completed'


def eligible_ids(records, procedure):
    stages = {'stage1': (1,), 'stage2': (2,), 'combined': (1, 2)}[procedure]
    return {'default', 'otsu_059'} | {r['configuration_id'] for r in records if r['stage'] in stages}


def select_scores(scores, ids, floor=.4):
    feasible = [cid for cid in ids if cid in scores and np.isfinite(scores[cid]['dice_mean']) and scores[cid]['dice_mean'] >= floor]
    if not feasible:
        return 'otsu_059', 'no_variant_reaches_dice_floor'
    def key(cid):
        s = scores[cid]
        def descending(value):
            return -value if np.isfinite(value) else np.inf
        return (descending(s['recall']), descending(s['small_recall']), -s['dice_mean'],
                s['false_area_mm2'], s['fp_per_slice'], cid)
    return min(feasible, key=key), 'selected'


def choose_fold(primary, fold, records, procedure):
    train, test = set(fold['train']), set(fold['test'])
    inner = [c for group in fold['inner_validation'] for c in group]
    if train & test or sorted(inner) != sorted(train):
        raise ValueError('Invalid patient partition')
    training = primary[primary.case_id.isin(inner)]
    scores = {cid:summarize(g) for cid,g in training.groupby('configuration_id')}
    cid, reason = select_scores(scores, eligible_ids(records, procedure))
    return cid, reason, scores


def summary(frame):
    result = summarize(frame)
    result['positive_slices_without_detection'] = int(frame.positive_without_detection.sum())
    result['fusion_components'] = int(frame.fusion_components.sum())
    result['split_references'] = int(frame.split_references.sum())
    return result


def validate_run_metadata(out):
    out = Path(out)
    selection = pd.read_csv(out/'selection.csv')
    folds = json.loads((out/'folds.json').read_text())
    records = json.loads((out/'configurations.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    actual = dict(selection=safe_records(selection), folds=folds, configurations=records,
                  controls=json.loads((out/'controls.json').read_text()),
                  protocol=json.loads((out/'protocol.json').read_text()))
    if any(manifest.get(key) != value for key, value in actual.items()):
        raise ValueError('Run metadata differs from frozen generation manifest')
    validate_folds(folds, selection.case_id.unique())
    return selection, folds, records


def analyze(out, figures=True):
    out = Path(out)
    selection, folds, records = validate_run_metadata(out)
    expected_ids = set(CONTROLS) | {r['configuration_id'] for r in records}
    metrics, lesions = [], []
    for row in selection.itertuples():
        folder = folder_for(out, row)
        if not verify_seal(folder):
            raise ValueError(f'Incomplete hybrid slice: {folder}')
        data = read_inputs(folder)
        ref, spacing = data['reference'], data['spacing']
        prepared = prepare_reference(ref, spacing[:2])
        recomputed = []
        with np.load(folder/'masks.npz', allow_pickle=False) as masks:
            if set(masks['ids']) != expected_ids or len(masks['ids']) != len(expected_ids):
                raise ValueError('Incomplete hybrid mask grid')
            for cid in masks['ids']:
                mask = unpack(masks, cid)
                if np.any(mask & ~data['liver']):
                    raise ValueError('Prediction outside liver')
                scored, matched = evaluate_mask(mask, ref, spacing, prepared)
                identity = dict(configuration_id=str(cid), case_id=int(row.case_id), slice_index=int(row.slice_index))
                recomputed.extend(dict(**identity, **s) for s in scored)
                lesions.extend(dict(**identity, **s) for s in matched)
        actual = pd.DataFrame(recomputed)
        saved = pd.read_csv(folder/'metrics.csv')
        keys = ['configuration_id', 'min_iou']
        pd.testing.assert_frame_equal(actual[saved.columns].sort_values(keys).reset_index(drop=True),
                                      saved.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=1e-12)
        metrics.extend(recomputed)
    metrics, lesions = pd.DataFrame(metrics), pd.DataFrame(lesions)
    primary = metrics[metrics.min_iou == .1]
    choices, inner_rows, outer = [], [], []
    for fold in folds:
        chosen = [(cid,cid) for cid in CONTROLS]
        for procedure in PROCEDURES:
            cid, reason, scores = choose_fold(primary, fold, records, procedure)
            choices.append(dict(fold=fold['fold'], family=procedure, configuration_id=cid, reason=reason,
                                training_dice=scores[cid]['dice_mean'], training_recall=scores[cid]['recall']))
            chosen.append((procedure, cid))
        for j, cases in enumerate(fold['inner_validation']):
            for cid, group in primary[primary.case_id.isin(cases)].groupby('configuration_id'):
                inner_rows.append(dict(fold=fold['fold'], inner_fold=j, configuration_id=cid, **summary(group)))
        for family,cid in chosen:
            outer.append(metrics[(metrics.configuration_id == cid) & metrics.case_id.isin(fold['test'])].assign(family=family, fold=fold['fold']))
    outer = pd.concat(outer, ignore_index=True)
    if len(outer) != len(selection)*(len(CONTROLS)+len(PROCEDURES))*len(CRITERIA) or outer.duplicated(['family','case_id','slice_index','min_iou']).any():
        raise ValueError('Incomplete or duplicate outer evaluation')
    keys = ['configuration_id','case_id','slice_index','min_iou']
    outer_lesions = outer[keys+['family','fold']].merge(lesions, on=keys, validate='many_to_many')
    for baseline in ('default', 'otsu_059'):
        compare_keys = ['case_id', 'slice_index', 'min_iou', 'reference_id']
        base = outer_lesions[outer_lesions.family == baseline][compare_keys+['detected']].rename(columns={'detected':f'{baseline}_detected'})
        outer_lesions = outer_lesions.merge(base, on=compare_keys, validate='many_to_one')
        outer_lesions[f'gained_vs_{baseline}'] = outer_lesions.detected & ~outer_lesions[f'{baseline}_detected']
        outer_lesions[f'lost_vs_{baseline}'] = ~outer_lesions.detected & outer_lesions[f'{baseline}_detected']
    summaries = pd.DataFrame([dict(family=f,min_iou=t,**summary(g)) for (f,t),g in outer.groupby(['family','min_iou'])])
    percase = pd.DataFrame([dict(family=f,min_iou=t,case_id=c,**summary(g)) for (f,t,c),g in outer.groupby(['family','min_iou','case_id'])])
    percase['below_dice_floor'] = percase.dice_mean < .4
    exploratory = pd.DataFrame([dict(configuration_id=c,min_iou=t,**summary(g)) for (c,t),g in metrics.groupby(['configuration_id','min_iou'])])
    exploratory['meets_dice_floor'] = exploratory.dice_mean >= .4
    exploratory['pareto_sensitivity_dice'] = False
    for criterion in CRITERIA:
        panel = exploratory[exploratory.min_iou == criterion]
        for idx, r in panel.iterrows():
            dominated = ((panel.recall >= r.recall) & (panel.dice_mean >= r.dice_mean) &
                         ((panel.recall > r.recall) | (panel.dice_mean > r.dice_mean))).any()
            exploratory.loc[idx, 'pareto_sensitivity_dice'] = not dominated
    sizes=[]
    for (f,t), group in outer_lesions.groupby(['family','min_iou']):
        for label in ('very_small','small','medium','big','at_least_30'):
            g = group[group.diameter_mm >= 30] if label == 'at_least_30' else group[group['size'] == label]
            sizes.append(dict(family=f,min_iou=t,size=label,TP=int(g.detected.sum()),FN=int((~g.detected).sum()),
                              recall=float(g.detected.mean()) if len(g) else np.nan))
    sizes=pd.DataFrame(sizes)
    primary_summary=summaries[summaries.min_iou==.1].set_index('family')
    primary_sizes=sizes[sizes.min_iou==.1].set_index(['family','size'])
    success={}
    for procedure in PROCEDURES:
        s=primary_summary.loc[procedure]
        success[procedure]=dict(dice_floor=bool(s.dice_mean>=.4),
            improves_total_recall=bool(all(s.recall>primary_summary.loc[b,'recall'] for b in ('default','otsu_059'))),
            preserves_small_recall=bool(all(s.small_recall>=primary_summary.loc[b,'small_recall'] for b in ('default','otsu_059'))),
            preserves_large_recall=bool(all(primary_sizes.loc[(procedure,'at_least_30'),'recall'] >= primary_sizes.loc[(b,'at_least_30'),'recall'] for b in ('default','otsu_059'))))
        success[procedure]['success']=all(success[procedure].values())
    all_scores={cid:summarize(g) for cid,g in primary.groupby('configuration_id')}
    preset_id, reason=select_scores(all_scores,eligible_ids(records,'combined'))
    controls=json.loads((out/'controls.json').read_text())
    settings={r['configuration_id']:r['settings'] for r in records}
    preset = dict(detection=dict(enabled=True,**settings[preset_id])) if preset_id in settings else controls[preset_id]
    report=out/'analysis.partial'
    if report.exists():shutil.rmtree(report)
    report.mkdir()
    gains=outer_lesions[(outer_lesions.family.isin(PROCEDURES)) & (outer_lesions.min_iou==.1)].copy()
    gains=gains[gains[[f'{action}_vs_{base}' for action in ('gained','lost') for base in ('default','otsu_059')]].any(axis=1)]
    negatives=outer[(outer.reference_area==0) & (outer.min_iou==.1)].copy()
    negatives['false_area_mm2']=negatives.pred_area*negatives.pixel_area_mm2
    negatives=negatives.sort_values('false_area_mm2',ascending=False)
    for name, frame in [('metrics',metrics),('choices',pd.DataFrame(choices)),('inner_validation',pd.DataFrame(inner_rows)),
        ('outer_per_slice',outer),('outer_per_lesion',outer_lesions),('summary',summaries),('per_case',percase),
        ('exploratory',exploratory),('grid_lesions',lesions),('size_summary',sizes),('gains_losses',gains),('negative_slices',negatives)]:
        frame.to_csv(report/f'{name}.csv',index=False)
    atomic_json(report/'promotion.json',dict(procedures=success,defaults_changed=False,interpretation=PROTOCOL['interpretation']))
    atomic_json(report/'audit.json',dict(slices=len(selection),masks_recomputed=len(selection)*len(expected_ids),
        saved_metrics_equal=True,liver_confinement=True,all_reused_controls_reproduced=True))
    atomic_json(report/'analysis_provenance.json',dict(source_manifest_sha256=digest(out/'manifest.json'),
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        python=platform.python_version(),dependencies={name:version(name) for name in DEPENDENCIES}))
    # Complete portable config; paths resolve relative to the exported YAML.
    full_cfg=yaml.safe_load((out/'base_config.yaml').read_text())
    full_cfg.update(preset)
    (report/'preset.yaml').write_text(yaml.safe_dump(full_cfg,sort_keys=False))
    atomic_json(report/'preset_selection.json',dict(configuration_id=preset_id,reason=reason,
        interpretation='Chosen on all ten patients; not independently evaluated'))
    if figures:render_figures(out,report,outer,exploratory,selection,gains)
    lines=['# Combinación por tamaño: comparación interna exploratoria','',
        'Selección: máxima sensibilidad con Dice medio por paciente ≥0,40; sin límite de FP. '
        'Cada elección externa usa exclusivamente los ocho pacientes internos. '
        'El respaldo Otsu se registra cuando ninguna variante alcanza el piso.','',
        '| Procedimiento | IoU | Dice | TP | FN | Pequeñas TP/32 | FP/corte | Área falsa mm² |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries.itertuples():
        lines.append(f'| {r.family} | {r.min_iou:.2f} | {r.dice_mean:.4f} | {r.TP} | {r.FN} | {r.small_TP} | {r.fp_per_slice:.2f} | {r.false_area_mm2:.1f} |')
    lines.extend(['',f'Éxito observado del procedimiento combinado: **{success["combined"]["success"]}**.',
        f'Preset exploratorio elegido con toda la cohorte: `{preset_id}`. No se modificaron defaults.', '',
        'Las figuras incluyen dos variantes híbridas diagnósticas elegidas con toda la cohorte (máximo recall y máximo Dice); no son resultados externos. '
        'Las dos etapas se ejecutaron sin ampliar la grilla tras consultar los resultados externos. '
        'Se distinguen resultados externos del procedimiento y resultados descriptivos de configuraciones fijas. '
        'Las 83 referencias son componentes 2D, no lesiones volumétricas independientes.', '',
        'Ver `choices.csv` para respaldos, `per_case.csv` para pacientes bajo 0,40, '
        '`size_summary.csv`, `gains_losses.csv` y `negative_slices.csv` para los costos por tamaño y caso. '
        '`audit.json` registra la recomputación desde todas las máscaras guardadas. '
        'Las fusiones/divisiones indican solapamiento positivo de componentes, separado del umbral de correspondencia.', '',
        'La adaptación HU usa mediana/MAD hepática por corte y el tamaño de candidatos predichos. '
        'No hay ajuste manual por paciente, uso de referencia durante predicción ni contexto entre cortes. '
        'La cohorte ya fue explorada: estos resultados no constituyen validación independiente.'])
    (report/'report.md').write_text('\n'.join(lines)+'\n')
    seal(report)
    final=out/'analysis'
    if final.exists():
        backup=out/'analysis.previous'
        if backup.exists():shutil.rmtree(backup)
        final.replace(backup)
    report.replace(final)
    return dict(success=success,preset_id=preset_id,masks_recomputed=len(selection)*len(expected_ids))


def render_figures(out, report, outer, exploratory, selection, gains):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=report/'figures';directory.mkdir()
    panel=exploratory[exploratory.min_iou==.1]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(panel.dice_mean,panel.recall,c=np.where(panel.meets_dice_floor,'#237da1','#b0b0b0'),s=18)
    frontier=panel[panel.pareto_sensitivity_dice].sort_values('dice_mean')
    ax.plot(frontier.dice_mean,frontier.recall,'o-',color='#d96924',label='Exploratory Pareto frontier')
    for r in panel[panel.configuration_id.isin(CONTROLS)].itertuples():ax.annotate(r.configuration_id,(r.dice_mean,r.recall),fontsize=8)
    ax.axvline(.4,color='black',linestyle='--',label='Dice floor 0.40')
    ax.set(xlabel='Mean patient Dice',ylabel='Lesion sensitivity (IoU >=0.10)',title='All-cohort configurations; not held-out selection')
    ax.legend();fig.tight_layout();fig.savefig(directory/'frontier.png',dpi=140);plt.close(fig)
    selected=outer[outer.min_iou==.1]
    hybrid=panel[panel.configuration_id.str.startswith('h')]
    diagnostic_ids={
        'exploratory_max_recall':hybrid.sort_values(['recall','small_recall','dice_mean','configuration_id'],ascending=[False,False,False,True]).iloc[0].configuration_id,
        'exploratory_max_dice':hybrid.sort_values(['dice_mean','recall','configuration_id'],ascending=[False,False,True]).iloc[0].configuration_id}
    atomic_json(report/'diagnostic_configurations.json',dict(configurations=diagnostic_ids,scope='Selected using the entire cohort for diagnosis only; not held-out results'))
    grid_metrics=pd.read_csv(report/'metrics.csv')
    grid_metrics=grid_metrics[grid_metrics.min_iou==.1]
    index=[]
    for row in selection.itertuples():
        folder=folder_for(out,row);data=read_inputs(folder)
        hu,liver,ref=(data[k] for k in ('hu','liver','reference'))
        rows=selected[(selected.case_id==row.case_id)&(selected.slice_index==row.slice_index)].set_index('family')
        coords=np.argwhere(liver)
        box=tuple(slice(max(0,a-5),min(n,b+6)) for a,b,n in zip(coords.min(0),coords.max(0),hu.shape)) if coords.size else (slice(None),slice(None))
        for label,cid in diagnostic_ids.items():
            extra=grid_metrics[(grid_metrics.case_id==row.case_id)&(grid_metrics.slice_index==row.slice_index)&(grid_metrics.configuration_id==cid)].iloc[0]
            rows.loc[label]=extra.reindex(rows.columns)
        families=('otsu_059','default','stage1','stage2','combined',*diagnostic_ids)
        fig,axes=plt.subplots(2,4,figsize=(18,10),layout='constrained');axes=axes.ravel();axes[-1].axis('off')
        with np.load(folder/'masks.npz') as masks:
            for ax,f in zip(axes,families):
                r=rows.loc[f];mask=unpack(masks,r.configuration_id)
                ax.imshow(hu[box].T,cmap='gray',vmin=0,vmax=200,origin='lower')
                rgba=np.zeros((*mask[box].T.shape,4));rgba[mask[box].T]=[1,.4,0,.65]
                ax.imshow(rgba,origin='lower')
                if ref[box].any():ax.contour(ref[box].T,levels=[.5],colors='cyan',linewidths=.7)
                ax.set_title(f'{f}: {r.configuration_id}\nDice {r.dice:.3f}; TP {r.TP}; FN {r.FN}; FP {r.FP}',fontsize=9);ax.axis('off')
        fig.suptitle(f'Case {row.case_id}, slice {row.slice_index}; cyan reference, orange prediction')
        filename=f'{row.case_id:03d}_{row.slice_index:04d}.png'
        fig.savefig(directory/filename,dpi=110);plt.close(fig)
        changed=gains[(gains.case_id==row.case_id)&(gains.slice_index==row.slice_index)]
        index.append(dict(case_id=row.case_id,slice_index=row.slice_index,figure=filename,
                          gains_or_losses=not changed.empty,negative=not ref.any()))
    pd.DataFrame(index).to_csv(report/'figure_index.csv',index=False)
