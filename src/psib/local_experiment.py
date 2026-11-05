"""Portable experiment support. Reference-dependent code lives here, not in detection."""
from pathlib import Path
import hashlib
import json
import os
import time
import platform
try:
    import resource
except ImportError:  # GUI fixed-preset runs also remain importable on Windows.
    resource = None
import shutil
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .classical_search import score_mask, summarize_search
from .lesion_eval import prepare_reference, match_lesions_one_to_one, label_prediction_ious, _optimal_pairs
from .local_segmentation import (configurations, prepare_maps, prepare_context_maps,
    context_from_volume, build_candidates, refine_candidates)
from .detection import configured_prediction
from .data_io import load_case

DEPENDENCIES = ('numpy','scipy','scikit-image','nibabel','pandas','matplotlib','PyYAML')
SIZE_BINS = ('very_small', 'small', 'medium', 'big')
SIZE_LABELS = ('Very small (<10 mm)', 'Small (10 to <30 mm)', 'Medium (30 to 50 mm)', 'Big (>50 mm)')
GEOMETRY_COLUMNS = ('configuration_id', 'min_iou', 'reference_id', 'predicted_id', 'diameter_mm',
                    'size', 'match_iou', 'center_distance_mm', 'assd_mm', 'hd95_mm')


def atomic_json(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=_json_default)+'\n')
    temporary.replace(path)


def _json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def safe_records(frame):
    return frame.astype(object).where(pd.notna(frame),None).to_dict('records')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def fingerprint(root, pairs, selection, folds, controls):
    payload=dict(schema=1,python=platform.python_version(),grid=configurations(),selection=safe_records(selection),folds=folds,controls=controls,
        code={str(p.relative_to(root)):digest(p) for directory in ('src','scripts')
              for p in sorted((root/directory).rglob('*.py'))},
        dependencies={name:version(name) for name in DEPENDENCIES},
        data={str(case):[digest(p) for p in paths] for case,paths in sorted(pairs.items())})
    encoded=json.dumps(payload,sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(),payload


def seal(directory):
    atomic_json(directory/'complete.json',{str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*'))
                                         if p.is_file() and p != directory/'complete.json'})


def verify_seal(directory):
    manifest=directory/'complete.json'
    if not manifest.exists(): return False
    expected=json.loads(manifest.read_text())
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != manifest}
    if actual!=set(expected) or any(digest(directory/name)!=value for name,value in expected.items()):
        raise ValueError(f'Corrupt completed artifacts: {directory}')
    return True


def _detected(mask,reference,spacing,prepared):
    pairs=match_lesions_one_to_one(mask,reference,spacing[:2],.1,prepared_reference=prepared)
    return set(pairs.loc[pairs.status=='TP','reference_id'].astype(int))


def lesion_diagnostics(hu,liver,reference,spacing,prepared,result):
    final=_detected(result['mask'],reference,spacing,prepared)
    raw=_detected(result['candidates'],reference,spacing,prepared)
    labels,_,diameters=prepared
    plabels,_=ndi.label(result['mask'],np.ones((3,3),bool))
    rows=[]
    for rid,diameter in diameters.items():
        component=labels==rid
        overlaps=np.unique(plabels[component]); overlaps=overlaps[overlaps>0]
        merged=any(len(np.unique(labels[plabels==p][labels[plabels==p]>0]))>1 for p in overlaps)
        distance=ndi.distance_transform_edt(~component,sampling=spacing[:2])
        ring=liver & ~reference & (distance>0) & (distance<=5.)
        delta=float(np.median(hu[component])-np.median(hu[ring])) if ring.any() else None
        present=bool((component & result['candidates']).any())
        rows.append(dict(reference_id=int(rid),diameter_mm=float(diameter),detected=rid in final,
            candidate_overlap=present,support_detected=rid in raw,
            stage='detected' if rid in final else 'no_candidate' if not present else
                  'lost_during_refinement' if rid in raw else 'poor_delineation',
            fusion=bool(merged),delta_local_hu=delta,
            appearance='unknown' if delta is None else 'low_contrast' if abs(delta)<=5 else
                       'dark' if delta<0 else 'bright'))
    return rows


def process_patient(task):
    case,indices,data_dir,out_string,controls,*optional=task
    expected_areas=optional[0] if optional else {}
    out=Path(out_string); patient=out/f'case_{case:03d}';patient.mkdir(exist_ok=True)
    missing=[k for k in indices if not verify_seal(patient/f'slice_{k:04d}')]
    if not missing: return dict(case_id=case,status='cached')
    start=time.perf_counter()
    ct,seg,spacing=load_case(case,data_dir); liver_volume=seg>0
    records=configurations()
    for k in missing:
        target=patient/f'slice_{k:04d}'
        temporary=patient/f'.slice_{k:04d}.partial'
        if temporary.exists(): shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            hu=ct[:,:,k];liver=liver_volume[:,:,k];reference=seg[:,:,k]==2
            if k in expected_areas and int(reference.sum()) != expected_areas[k]:
                raise ValueError(f'Frozen reference area changed: patient {case}, slice {k}')
            prepared_reference=prepare_reference(reference,spacing[:2])
            maps=prepare_maps(hu,liver,spacing,records[0]['settings'])
            context=context_from_volume(ct,liver_volume,k,spacing)
            neighbors=prepare_context_maps(context,hu.shape,spacing,records[0]['settings'])
            generated_cache={};diagnostics={}; masks=[];seeds=[];supports=[];metrics=[];lesions=[];ids=[];candidate_masks={}
            for record in records:
                cid=record['configuration_id'];cfg=record['settings']
                key=(cfg['high'],cfg['polarity'],cfg['context'])
                if key not in generated_cache:
                    generated_cache[key]=build_candidates(maps,cfg,neighbors if cfg['context'] else None)
                tick=time.perf_counter()
                result=refine_candidates(maps,generated_cache[key],cfg)
                elapsed=time.perf_counter()-tick
                diagnostics[cid]=dict(components=result['components'],context_note=result['context_note'],elapsed_s=elapsed,
                                      failed_candidates=result['failed_candidates'])
                for i,(box,mask) in enumerate(result['candidate_masks']):
                    candidate_masks[f'{cid}_{i:04d}']=np.packbits(mask)
                for row in score_mask(result['mask'],prepared_reference):
                    metrics.append(dict(configuration_id=cid,case_id=case,slice_index=k,
                                        pixel_area_mm2=float(np.prod(spacing[:2])),
                                        failed_candidates=result['failed_candidates'],elapsed_s=elapsed,**row))
                lesions.extend(dict(configuration_id=cid,case_id=case,slice_index=k,**r) for r in
                    lesion_diagnostics(hu,liver,reference,spacing,prepared_reference,result))
                ids.append(cid); masks.append(np.packbits(result['mask']));seeds.append(np.packbits(result['seeds']));supports.append(np.packbits(result['candidates']))
            for cid,cfg in controls.items():
                tick=time.perf_counter();mask=configured_prediction(hu,liver,spacing,cfg)['mask']
                for row in score_mask(mask,prepared_reference):
                    metrics.append(dict(configuration_id=cid,case_id=case,slice_index=k,failed_candidates=0,
                                        pixel_area_mm2=float(np.prod(spacing[:2])),elapsed_s=time.perf_counter()-tick,**row))
                result=dict(mask=mask,candidates=mask)
                lesions.extend(dict(configuration_id=cid,case_id=case,slice_index=k,**r) for r in
                    lesion_diagnostics(hu,liver,reference,spacing,prepared_reference,result))
                ids.append(cid);masks.append(np.packbits(mask));seeds.append(np.zeros_like(masks[-1]));supports.append(np.packbits(mask))
            np.savez_compressed(temporary/'masks.npz',ids=np.array(ids),shape=hu.shape,
                                predictions=np.stack(masks),seeds=np.stack(seeds),supports=np.stack(supports))
            np.savez_compressed(temporary/'inputs.npz',hu=hu,liver=liver,reference=reference,spacing=spacing,
                                dark_score=maps['scores']['dark'],bright_score=maps['scores']['bright'])
            np.savez_compressed(temporary/'candidate_masks.npz',**candidate_masks)
            atomic_json(temporary/'candidates.json',diagnostics)
            pd.DataFrame(metrics).to_csv(temporary/'metrics.csv',index=False)
            pd.DataFrame(lesions,columns=['configuration_id','case_id','slice_index','reference_id','diameter_mm','detected',
                'candidate_overlap','support_detected','stage','fusion','delta_local_hu','appearance']).to_csv(temporary/'lesions.csv',index=False)
            seal(temporary)
            if target.exists(): shutil.rmtree(target)
            temporary.replace(target)
            print(f'Patient {case}, slice {k}: checkpoint complete',flush=True)
        except BaseException as exc:
            atomic_json(temporary/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',case_id=case,slice_index=k))
            raise
    return dict(case_id=case,status='completed',elapsed_s=time.perf_counter()-start,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024. if resource else None)


def rank(score,cid):
    return (-score['dice_mean'],score.get('false_area_mm2',score['pixel_FP']),score['fp_per_slice'],cid)


def select(scores,constrained=True):
    baseline=scores['default']; admissible=[]
    for cid,score in scores.items():
        if cid!='default' and not cid.startswith('local_'): continue
        if score.get('failed_candidates',0): continue
        if constrained and any(np.isfinite(baseline[k]) and
            (not np.isfinite(score[k]) or score[k]<baseline[k]) for k in ('recall','small_recall')): continue
        admissible.append(cid)
    # Default competes on Dice too: a novel method with worse Dice is not a winner.
    best=min(admissible,key=lambda cid:rank(scores[cid],cid))
    reason='selected' if best!='default' else 'no_local_improves_dice_under_constraints' if constrained else 'default_has_best_dice'
    return best,reason


def summarize(frame):
    score=summarize_search(frame)
    score['failed_candidates']=int(frame.failed_candidates.sum())
    score['false_area_mm2']=float(((frame.pred_area-frame.intersection)*frame.pixel_area_mm2).sum())
    return score


def boundary_tree(mask, spacing, origin=0):
    boundary = mask & ~ndi.binary_erosion(mask)
    return cKDTree((np.argwhere(boundary) + origin) * spacing)


def boundary_distances(predicted_boundary, reference_boundary):
    return (reference_boundary.query(predicted_boundary.data)[0],
            predicted_boundary.query(reference_boundary.data)[0])


def hd95_mm(prediction, reference_boundary, spacing):
    """Maximum of the two directed 95th-percentile boundary distances in 2D."""
    if not reference_boundary.n or not prediction.any():
        return np.nan
    distances = boundary_distances(boundary_tree(prediction, spacing), reference_boundary)
    return max(np.percentile(direction, 95) for direction in distances)


def component_geometry(labels, box, component, spacing):
    mask = labels[box] == component
    origin = np.array([axis.start for axis in box])
    center = (np.asarray(ndi.center_of_mass(mask)) + origin) * spacing
    return center, boundary_tree(mask, spacing, origin)


def size_category(diameter):
    if diameter < 10:
        return 'very_small'
    if diameter < 30:
        return 'small'
    if diameter <= 50:
        return 'medium'
    return 'big'


def matched_geometry(prediction, prepared, reference_geometry, spacing, criteria):
    reference_labels, reference_sizes, diameters = prepared
    predicted_labels, ious = label_prediction_ious(prediction, reference_labels, reference_sizes)
    boxes = ndi.find_objects(predicted_labels)
    pair_metrics = {}
    for criterion in criteria:
        pairs = {ref + 1: pred + 1 for ref, pred in _optimal_pairs(ious, (ious > 0) & (ious >= criterion))}
        for ref, diameter in diameters.items():
            pred = pairs.get(ref)
            row = dict(min_iou=criterion, reference_id=ref, predicted_id=pred,
                       diameter_mm=diameter, size=size_category(diameter))
            if pred is not None:
                if (ref, pred) not in pair_metrics:
                    ref_center, ref_boundary = reference_geometry[ref]
                    pred_center, pred_boundary = component_geometry(predicted_labels, boxes[pred-1], pred, spacing)
                    distances = boundary_distances(pred_boundary, ref_boundary)
                    pair_metrics[ref, pred] = dict(
                        match_iou=ious[ref-1, pred-1],
                        center_distance_mm=np.linalg.norm(pred_center - ref_center),
                        assd_mm=sum(direction.sum() for direction in distances) / sum(map(len, distances)),
                        hd95_mm=max(np.percentile(direction, 95) for direction in distances))
                row.update(pair_metrics[ref, pred])
            yield row


def slice_geometry(folder, metrics):
    with np.load(folder/'inputs.npz') as inputs:
        reference = inputs['reference']
        spacing = inputs['spacing'][:2]
    if not reference.any():
        return np.nan, []

    reference_boundary = boundary_tree(reference, spacing)
    prepared = prepare_reference(reference, spacing)
    reference_labels = prepared[0]
    reference_geometry = {rid: component_geometry(reference_labels, box, rid, spacing)
                          for rid, box in enumerate(ndi.find_objects(reference_labels), 1)}
    with np.load(folder/'masks.npz') as masks:
        ids = masks['ids']
        predictions = masks['predictions']
    expected_ids = set(metrics.configuration_id)
    if len(ids) != len(predictions) or len(ids) != len(expected_ids) or set(ids) != expected_ids:
        raise ValueError(f'Invalid prediction coverage for geometry: {folder}')

    criteria = sorted(metrics.min_iou.unique())
    distances = {}
    geometry = []
    for cid, packed in zip(ids, predictions):
        prediction = np.unpackbits(packed, count=reference.size).reshape(reference.shape).astype(bool)
        distances[cid] = hd95_mm(prediction, reference_boundary, spacing)
        geometry.extend(dict(configuration_id=cid, **row) for row in
                        matched_geometry(prediction, prepared, reference_geometry, spacing, criteria))
    return metrics.configuration_id.map(distances), geometry


def summarize_report(frame):
    score = summarize(frame)
    patient_hd95 = frame.groupby('case_id').hd95_mm.mean()
    score['hd95_mean_mm'] = float(patient_hd95.mean())
    score['hd95_n_slices'] = int(frame.hd95_mm.count())
    score['hd95_n_cases'] = int(patient_hd95.count())
    return score


def size_summary(slices, lesions, group_column, scope):
    groups = [group_column, 'min_iou']
    operating_points = slices.groupby(groups).FP.agg(fp='sum', n_slices='size')
    operating_points['fp_per_slice'] = operating_points.fp / operating_points.n_slices
    geometry = lesions.groupby([*groups, 'size']).agg(
        n_reference=('reference_id', 'size'), tp=('predicted_id', 'count'),
        center_distance_mean_mm=('center_distance_mm', 'mean'), assd_mean_mm=('assd_mm', 'mean'),
        lesion_hd95_mean_mm=('hd95_mm', 'mean')).reset_index()
    sizes = pd.DataFrame({'size': SIZE_BINS})
    summary = operating_points.reset_index().merge(sizes, how='cross')
    summary = summary.merge(geometry, how='left', on=[*groups, 'size'], validate='one_to_one')
    for column in ('n_reference', 'tp'):
        summary[column] = summary[column].fillna(0).astype(int)
    summary['fn'] = summary.n_reference - summary.tp
    summary['sensitivity'] = summary.tp / summary.n_reference.where(summary.n_reference > 0)
    return summary.rename(columns={group_column: 'operating_point'}).assign(scope=scope)


def choose_fold(primary,fold):
    """Only training-patient rows may influence either selection procedure."""
    if set(fold['train']) & set(fold['test']):raise ValueError('Overlapping patient partitions')
    training=primary[primary.case_id.isin(fold['train'])]
    scores={cid:summarize(g) for cid,g in training.groupby('configuration_id')}
    return [(family,*select(scores,constrained)) for family,constrained in
            [('constrained',True),('unconstrained',False)]]


def analyze(out,selection,folds,controls,*,figures=True):
    out=Path(out);all_metrics=[];all_lesions=[];all_geometry=[]
    for row in selection.itertuples():
        folder=out/f'case_{int(row.case_id):03d}'/f'slice_{int(row.slice_index):04d}'
        if not verify_seal(folder): raise ValueError(f'Incomplete slice: {folder}')
        slice_metrics = pd.read_csv(folder/'metrics.csv')
        slice_metrics['hd95_mm'], geometry = slice_geometry(folder, slice_metrics)
        all_geometry.extend(dict(case_id=row.case_id, slice_index=row.slice_index, **entry) for entry in geometry)
        all_metrics.append(slice_metrics)
        all_lesions.append(pd.read_csv(folder/'lesions.csv'))
    metrics=pd.concat(all_metrics,ignore_index=True);lesions=pd.concat(all_lesions,ignore_index=True)
    geometry = pd.DataFrame(all_geometry, columns=('case_id', 'slice_index', *GEOMETRY_COLUMNS))
    if len(metrics)!=len(selection)*(len(configurations())+len(controls))*2: raise ValueError('Incomplete metrics grid')
    expected_ids={r['configuration_id'] for r in configurations()}|set(controls)
    for _,group in metrics.groupby(['case_id','slice_index','min_iou']):
        if len(group)!=len(expected_ids) or set(group.configuration_id)!=expected_ids: raise ValueError('Invalid configuration coverage')
    primary=metrics[metrics.min_iou==.1]
    choices=[]; outer=[]; lesion_outer=[]
    for fold in folds:
        chosen=[]
        for family,cid,reason in choose_fold(primary,fold):
            choices.append(dict(fold=fold['fold'],family=family,configuration_id=cid,reason=reason))
            chosen.append((family,cid))
        chosen.extend((cid,cid) for cid in controls)
        for family,cid in chosen:
            held=metrics[metrics.case_id.isin(fold['test']) & (metrics.configuration_id==cid)].copy()
            outer.append(held.assign(family=family,fold=fold['fold']))
            held_lesions=lesions[lesions.case_id.isin(fold['test']) & (lesions.configuration_id==cid)].copy()
            lesion_outer.append(held_lesions.assign(family=family,fold=fold['fold']))
    outer=pd.concat(outer,ignore_index=True);lesion_outer=pd.concat(lesion_outer,ignore_index=True)
    geometry_keys = ['configuration_id', 'case_id', 'slice_index', 'min_iou']
    outer_geometry = outer[[*geometry_keys, 'family']].merge(geometry, on=geometry_keys, how='inner')
    grid_sizes = size_summary(metrics, geometry, 'configuration_id', 'grid')
    outer_sizes = size_summary(outer, outer_geometry, 'family', 'outer')
    sizes = pd.concat([outer_sizes, grid_sizes], ignore_index=True)
    summary=pd.DataFrame([dict(family=f,min_iou=t,**summarize_report(g)) for (f,t),g in outer.groupby(['family','min_iou'])])
    percase=pd.DataFrame([dict(family=f,min_iou=t,case_id=c,**summarize_report(g)) for (f,t,c),g in outer.groupby(['family','min_iou','case_id'])])
    exploratory=pd.DataFrame([dict(configuration_id=cid,min_iou=t,**summarize_report(g)) for (cid,t),g in metrics.groupby(['configuration_id','min_iou'])])
    ids=['case_id','slice_index','reference_id']
    baseline=lesion_outer[lesion_outer.family=='default'][ids+['detected']].rename(columns={'detected':'default_detected'})
    lesion_outer=lesion_outer.merge(baseline,on=ids,validate='many_to_one')
    lesion_outer['gained']=lesion_outer.detected & ~lesion_outer.default_detected
    lesion_outer['lost']=~lesion_outer.detected & lesion_outer.default_detected
    report_dir=out/'analysis.partial'
    if report_dir.exists(): shutil.rmtree(report_dir)
    report_dir.mkdir()
    atomic_json(report_dir/'analysis_provenance.json',dict(python=platform.python_version(),
        analysis_source_sha256=digest(Path(__file__)),source_manifest_sha256=digest(out/'manifest.json') if (out/'manifest.json').exists() else None,
        dependencies={name:version(name) for name in DEPENDENCIES}))
    for name,frame in [('metrics',metrics),('choices',pd.DataFrame(choices)),('outer_per_slice',outer),('outer_per_lesion',lesion_outer),
                       ('summary',summary),('per_case',percase),('exploratory',exploratory),
                       ('size_summary', sizes), ('lesion_geometry', geometry)]:
        frame.to_csv(report_dir/f'{name}.csv',index=False)
    selected=summary[(summary.family=='constrained') & (summary.min_iou==.1)].iloc[0]
    baseline=summary[(summary.family=='default') & (summary.min_iou==.1)].iloc[0]
    eligible=bool(selected.dice_mean>baseline.dice_mean and selected.recall>=baseline.recall
                  and selected.small_recall>=baseline.small_recall and selected.failed_candidates==0)
    atomic_json(report_dir/'promotion.json',dict(eligible_procedure=eligible,defaults_changed=False,
        scope='Previously explored cohort; fold selection procedure, not independent validation of a fixed preset'))
    all_scores={cid:summarize(g) for cid,g in primary.groupby('configuration_id')}
    cid,reason=select(all_scores)
    import yaml
    records={r['configuration_id']:r['settings'] for r in configurations()}
    preset={'detection':dict(enabled=True,**records[cid])} if cid.startswith('local_') else controls['default']
    (report_dir/'preset.yaml').write_text(yaml.safe_dump(preset,sort_keys=False))
    atomic_json(report_dir/'preset_selection.json',dict(configuration_id=cid,reason=reason,scope='All-cohort exploratory future-test preset'))
    # Explicit ablations keep all 108 rows, not just winners after seeing held-out outcomes.
    grid=pd.DataFrame([dict(configuration_id=r['configuration_id'],**r['settings']) for r in configurations()])
    exploratory.merge(grid,on='configuration_id').to_csv(report_dir/'ablations.csv',index=False)
    if figures:
        render_figures(out,report_dir,outer)
        render_froc(grid_sizes, grid, report_dir/'figures')
    lines=['# Local candidates: exploratory internal comparison','',
           'Selection prioritizes patient Dice, constrained by total and small-lesion recall against default. '
           'All references are 2D; neighboring slices provide image context only. '
           'Failures exclude affected variants from selection; no silent algorithm fallback.','',
           'HD95 compares the 2D boundaries of the full slice masks in mm and is independent of the IoU match threshold. '
           'Only slices with both masks non-empty contribute. The mean gives equal weight to patients with valid slices.', '',
           '| Procedure | IoU match | Dice | HD95 mean (mm) | HD95 slices / reference-positive | Recall | Small recall | FP/slice | Pixel FP | Failed candidates |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        hd95 = f'{r.hd95_mean_mm:.2f}' if r.hd95_n_slices else '—'
        coverage = f'{r.hd95_n_slices}/{r.n_slices-r.negative_slices}'
        lines.append(f'| {r.family} | {r.min_iou} | {r.dice_mean:.4f} | {hd95} | {coverage} | {r.recall:.4f} | {r.small_recall:.4f} | {r.fp_per_slice:.2f} | {r.pixel_FP} | {r.failed_candidates} |')
    lines.extend(['',f'Procedure meets promotion criteria: {eligible}. No defaults changed.',
                  f'All-cohort future-test preset: {cid}. This fixed preset has no new independent evaluation.',
                  'Inspect outer_per_lesion.csv for gains, losses, contrast classes, missed candidates and fusions. '
                  'See ablations.csv for polarity/context/method comparisons; these are exploratory.', '',
                  '## Lesion size and matched geometry', '',
                  'Bins use the reference 2D maximum Feret diameter. Geometry means include matched pairs only, '
                  'with equal weight per lesion. ASSD averages all nearest-boundary distances from both directions.', '',
                  '| Size | Procedure | Detected at IoU ≥ 0.10 | Center distance (mm) | ASSD (mm) | Lesion HD95 (mm) |',
                  '|---|---|---:|---:|---:|---:|'])
    for size, label in zip(SIZE_BINS, SIZE_LABELS):
        for row in outer_sizes[(outer_sizes['size'] == size) & (outer_sizes.min_iou == .1)].itertuples():
            means = (row.center_distance_mean_mm, row.assd_mean_mm, row.lesion_hd95_mean_mm)
            distances = ' | '.join(f'{mean:.2f}' if row.tp else '—' for mean in means)
            lines.append(f'| {label} | {row.operating_point} | {row.tp}/{row.n_reference} | {distances} |')
    lines.extend(['', '## FROC', '',
                  'Each point shows sensitivity within one reference size bin versus global false positives per slice. '
                  'Predictions matched to lesions in other bins are excluded from the false-positive count. '
                  'The step lines show the best observed grid sensitivity at each false-positive budget by method.', '',
                  '[Size summaries and FROC points](size_summary.csv) · [Per-lesion geometry](lesion_geometry.csv)'])
    if figures:
        lines.extend(['', '[IoU ≥ 0.10](figures/froc_iou_010.png) · [IoU ≥ 0.25](figures/froc_iou_025.png)'])
    (report_dir/'report.md').write_text('\n'.join(lines)+'\n')
    seal(report_dir)
    final=out/'analysis'
    if final.exists():
        backup=out/'analysis.previous'
        if backup.exists(): shutil.rmtree(backup)
        final.replace(backup)
    report_dir.replace(final)
    return dict(eligible_procedure=eligible,failed_candidates=int(primary.failed_candidates.sum()))


def render_froc(sizes, grid, destination):
    import matplotlib.pyplot as plt

    points = sizes.merge(grid[['configuration_id', 'method']], how='left',
                         left_on='operating_point', right_on='configuration_id')
    colors = {'local_growth': '#1f77b4', 'local_contour': '#d14a18', 'local_graphcut': '#258259'}
    markers = {'default': 'D', 'otsu_059': 's', 'hu_sensitive': '^'}
    for criterion, data in points.groupby('min_iou'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout='constrained')
        handles = {}
        for ax, size, label in zip(axes.flat, SIZE_BINS, SIZE_LABELS):
            panel = data[data['size'] == size]
            count = int(panel.n_reference.iloc[0])
            ax.set_title(f'{label} · {count} references')
            ax.set_xscale('symlog', linthresh=1)
            ax.set_ylim(-.03, 1.04)
            ax.grid(alpha=.25)
            if not count:
                ax.text(.5, .5, 'No reference lesions', ha='center', transform=ax.transAxes)
                continue
            for method, group in panel.groupby('method'):
                color = colors[method]
                ax.scatter(group.fp_per_slice, group.sensitivity, color=color, s=12, alpha=.4)
                best = group.groupby('fp_per_slice').sensitivity.max().cummax()
                line, = ax.step(best.index, best, where='post', color=color)
                handles[method] = line
            for row in panel[panel.method.isna()].itertuples():
                handles[row.operating_point] = ax.scatter(row.fp_per_slice, row.sensitivity, s=45,
                    color='#202020', marker=markers.get(row.operating_point, 'o'), zorder=4)
        fig.suptitle(f'FROC by reference lesion size · IoU ≥ {criterion:.2f}')
        fig.supxlabel('False positives per slice (symmetric log scale)')
        fig.supylabel('Lesion sensitivity')
        if handles:
            fig.legend(handles.values(), handles.keys(), loc='outside right center', frameon=False)
        fig.savefig(destination/f'froc_iou_{round(criterion*100):03d}.png', dpi=150)
        plt.close(fig)


def render_figures(out,destination,outer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=destination/'figures'; directory.mkdir()
    selected=outer[(outer.family=='constrained') & (outer.min_iou==.1)]
    for row in selected.itertuples():
        folder=out/f'case_{row.case_id:03d}'/f'slice_{row.slice_index:04d}'
        with np.load(folder/'inputs.npz') as data:
            hu,reference,liver=data['hu'],data['reference'],data['liver']
        with np.load(folder/'masks.npz') as data:
            i=list(data['ids']).index(row.configuration_id)
            arrays=[np.unpackbits(data[k][i],count=hu.size).reshape(hu.shape).astype(bool)
                    for k in ('seeds','supports','predictions')]
        coords=np.argwhere(liver)
        if coords.size:
            lo=np.maximum(coords.min(0)-5,0);hi=np.minimum(coords.max(0)+6,hu.shape)
            box=tuple(slice(a,b) for a,b in zip(lo,hi))
        else: box=(slice(None),slice(None))
        fig,axes=plt.subplots(1,4,figsize=(14,4))
        for ax,title,mask in zip(axes,('HU / reference','Automatic seeds','Candidate support','Final / reference'),[None,*arrays]):
            ax.imshow(hu[box].T,cmap='gray',vmin=0,vmax=200,origin='lower')
            if mask is not None:
                rgba=np.zeros((*mask[box].T.shape,4));rgba[mask[box].T]=[1,.4,0,.7];ax.imshow(rgba,origin='lower')
            if reference.any():ax.contour(reference[box].T,levels=[.5],colors='cyan',linewidths=.6)
            ax.set_title(title);ax.axis('off')
        fig.suptitle(f'Case {row.case_id} / slice {row.slice_index} / {row.configuration_id}; cyan reference, orange prediction')
        fig.tight_layout();fig.savefig(directory/f'{row.case_id:03d}_{row.slice_index:04d}.png',dpi=130);plt.close(fig)
