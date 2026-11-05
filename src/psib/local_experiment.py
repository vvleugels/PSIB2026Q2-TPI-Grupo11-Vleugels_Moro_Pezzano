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
from .classical_search import score_mask, summarize_search
from .lesion_eval import prepare_reference, match_lesions_one_to_one
from .local_segmentation import (configurations, prepare_maps, prepare_context_maps,
    context_from_volume, build_candidates, refine_candidates)
from .detection import configured_prediction
from .data_io import load_case

DEPENDENCIES = ('numpy','scipy','scikit-image','nibabel','pandas','matplotlib','PyYAML')


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


def choose_fold(primary,fold):
    """Only training-patient rows may influence either selection procedure."""
    if set(fold['train']) & set(fold['test']):raise ValueError('Overlapping patient partitions')
    training=primary[primary.case_id.isin(fold['train'])]
    scores={cid:summarize(g) for cid,g in training.groupby('configuration_id')}
    return [(family,*select(scores,constrained)) for family,constrained in
            [('constrained',True),('unconstrained',False)]]


def analyze(out,selection,folds,controls,*,figures=True):
    out=Path(out);all_metrics=[];all_lesions=[]
    for row in selection.itertuples():
        folder=out/f'case_{int(row.case_id):03d}'/f'slice_{int(row.slice_index):04d}'
        if not verify_seal(folder): raise ValueError(f'Incomplete slice: {folder}')
        all_metrics.append(pd.read_csv(folder/'metrics.csv')); all_lesions.append(pd.read_csv(folder/'lesions.csv'))
    metrics=pd.concat(all_metrics,ignore_index=True);lesions=pd.concat(all_lesions,ignore_index=True)
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
    summary=pd.DataFrame([dict(family=f,min_iou=t,**summarize(g)) for (f,t),g in outer.groupby(['family','min_iou'])])
    percase=pd.DataFrame([dict(family=f,min_iou=t,case_id=c,**summarize(g)) for (f,t,c),g in outer.groupby(['family','min_iou','case_id'])])
    exploratory=pd.DataFrame([dict(configuration_id=cid,min_iou=t,**summarize(g)) for (cid,t),g in metrics.groupby(['configuration_id','min_iou'])])
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
                       ('summary',summary),('per_case',percase),('exploratory',exploratory)]:
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
    if figures: render_figures(out,report_dir,outer)
    lines=['# Local candidates: exploratory internal comparison','',
           'Selection prioritizes patient Dice, constrained by total and small-lesion recall against default. '
           'All references are 2D; neighboring slices provide image context only. '
           'Failures exclude affected variants from selection; no silent algorithm fallback.','',
           '| Procedure | IoU match | Dice | Recall | Small recall | FP/slice | Pixel FP | Failed candidates |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.family} | {r.min_iou} | {r.dice_mean:.4f} | {r.recall:.4f} | {r.small_recall:.4f} | {r.fp_per_slice:.2f} | {r.pixel_FP} | {r.failed_candidates} |')
    lines.extend(['',f'Procedure meets promotion criteria: {eligible}. No defaults changed.',
                  f'All-cohort future-test preset: {cid}. This fixed preset has no new independent evaluation.',
                  'Inspect outer_per_lesion.csv for gains, losses, contrast classes, missed candidates and fusions. '
                  'See ablations.csv for polarity/context/method comparisons; these are exploratory.'])
    (report_dir/'report.md').write_text('\n'.join(lines)+'\n')
    seal(report_dir)
    final=out/'analysis'
    if final.exists():
        backup=out/'analysis.previous'
        if backup.exists(): shutil.rmtree(backup)
        final.replace(backup)
    report_dir.replace(final)
    return dict(eligible_procedure=eligible,failed_candidates=int(primary.failed_candidates.sum()))


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
