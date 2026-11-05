"""216 Otsu pipelines, frozen patient validation and greedy OR ensembles.

Example: python scripts/run_classical_search.py --run-id classical-search-20261104
Use --resume only for an interrupted run with the same frozen inputs/config.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psib.classical_search import configurations,score_mask,summarize_search,choose,greedy_union,validate_folds
from psib.config import load_config
from psib.data_io import load_case
from psib.preprocessing import prepare_enhanced_slice,apply_hu_window
from psib.detection import configured_prediction
from psib.lesion_eval import prepare_reference
from scripts.run_segmentation import iter_predictions


def signature(cfg):
    p=cfg['preprocessing'];s=cfg['segmentation'];d=p['denoising'];e=p['enhancement']
    return (p['hu_window'].get('enabled', True),p['hu_window']['center'],p['hu_window']['width'],e.get('contrast', 1.0),
        d.get('gaussian_sigma',0) if d['method']=='gaussian' else 0,e['method'],
        e.get('clahe_clip_limit',.01) if e['method']=='clahe' else None,
        e.get('clahe_kernel_size',8) if e['method']=='clahe' else None,
        s['method'],s['opening'],s['closing'],s['fill_holes'],s['footprint'],s['radius'],s['min_area'])


def atomic_json(path, obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=True));temp.replace(path)


class Evaluator:
    def __init__(self,out,records,selection):
        self.out=out;self.index={r['configuration_id']:i for i,r in enumerate(records)}
        self.index['default']=len(records)
        self.slices={};self.cache={};self.calls=0
        for r in selection.itertuples():
            key=(int(r.case_id),int(r.slice_index));stem=f'case_{key[0]:03d}_slice_{key[1]:03d}'
            with np.load(out/'inputs'/f'{stem}.npz') as data:
                ref=data['reference'];spacing=data['spacing'];liver=data['liver']
            self.slices[key]=dict(stem=stem,shape=ref.shape,liver=liver,
                prepared=prepare_reference(ref,spacing[:2]),
                packed=np.load(out/'predictions'/f'{stem}.npy',mmap_mode='r'))
        f=pd.read_csv(out/'individual_per_slice.csv')
        for (cid,case,k),g in f.groupby(['configuration_id','case_id','slice_index']):
            self.cache[((str(cid),),int(case),int(k))]=g.to_dict('records')
        cachefile=out/'union_metrics_cache.csv'
        if cachefile.exists():
            for (members,case,k),g in pd.read_csv(cachefile).groupby(['members','case_id','slice_index']):
                self.cache[(tuple(members.split('+')),int(case),int(k))]=g.drop(columns='members').to_dict('records')

    def mask(self,members,key):
        s=self.slices[key]
        packed=np.bitwise_or.reduce(np.asarray(s['packed'][[self.index[c] for c in members]]),axis=0)
        return np.unpackbits(packed,count=int(np.prod(s['shape']))).reshape(s['shape']).astype(bool)

    def evaluate(self,members,cases,criteria=(.1,)):
        members=tuple(sorted(members));rows=[]
        for key,s in self.slices.items():
            if key[0] not in cases:continue
            cachekey=(members,*key)
            if cachekey not in self.cache:
                pred=self.mask(members,key)
                if np.any(pred & ~s['liver']):raise AssertionError('Union outside liver')
                self.cache[cachekey]=[dict(case_id=key[0],slice_index=key[1],**r)
                    for r in score_mask(pred,s['prepared'])]
            rows.extend(r for r in self.cache[cachekey] if r['min_iou'] in criteria)
        return pd.DataFrame(rows)

    def save_cache(self):
        rows=[dict(members='+'.join(members),**r) for (members,case,k),group in self.cache.items()
              if len(members)>1 for r in group]
        if rows:pd.DataFrame(rows).to_csv(self.out/'union_metrics_cache.csv',index=False)


def generate(args,out,cfg,records,selection):
    for directory in ('inputs','predictions','slice_metrics'):(out/directory).mkdir(exist_ok=True)
    signatures={signature(r['config']):i for i,r in enumerate(records)}
    historical=[]
    for run in sorted((ROOT/'results/segmentation').glob('*/effective_config.yaml')):
        try:
            h=yaml.safe_load(run.read_text())
            if h.get('preprocessing_scope') != 'liver':continue
            if h.get('detection',{}).get('enabled') or h.get('segmentation',{}).get('method')!='otsu':continue
            index=signatures.get(signature(h))
            if index is not None:historical.append((run.parent,index))
        except (KeyError,TypeError):continue
    parity=[];physical=[]
    for case,group in selection.groupby('case_id'):
        pending=[int(k) for k in group.slice_index if not (out/'slice_metrics'/f'case_{case:03d}_slice_{k:03d}.csv').exists()]
        if not pending:continue
        print(f'Loading patient {case}; {len(pending)} slices',flush=True)
        ct,seg,spacing=load_case(int(case),cfg['data_dir'])
        for k in pending:
            stem=f'case_{case:03d}_slice_{k:03d}';hu=ct[:,:,k];liver=seg[:,:,k]>0;ref=seg[:,:,k]==2
            expected=group[group.slice_index==k].iloc[0]
            if int(ref.sum())!=int(expected.tumor_area_px):raise ValueError('Frozen reference changed')
            coords=np.argwhere(liver)
            lo=np.maximum(coords.min(0)-3,0);hi=np.minimum(coords.max(0)+4,hu.shape)
            box=tuple(slice(a,b) for a,b in zip(lo,hi));prepared=prepare_reference(ref[box],spacing[:2])
            np.savez_compressed(out/'inputs'/f'{stem}.npz',reference=ref[box],liver=liver[box],
                spacing=spacing,origin=lo,full_shape=hu.shape,display=apply_hu_window(hu,100,200)[box])
            packed=[];rows=[]
            for start in range(0,len(records),6):
                chunk=records[start:start+6];c=chunk[0]['config']
                im=prepare_enhanced_slice(hu,c,liver_mask=liver)
                variants=[dict(configuration_id=r['configuration_id'],**r['config']['segmentation']) for r in chunk]
                for record,pred in iter_predictions(im,liver,variants):
                    index=len(packed);cid=record['configuration_id']
                    if np.any(pred & ~liver):raise AssertionError('Prediction outside liver')
                    for run,j in historical:
                        path=run/'predictions/cfg_000'/f'{stem}.npy'
                        if j==index and path.exists():
                            equal=np.array_equal(pred,np.load(path))
                            parity.append(dict(run=run.name,configuration_id=cid,case_id=case,slice_index=k,equal=equal))
                            if not equal:raise AssertionError(f'Historical mismatch: {path}')
                    packed.append(np.packbits(pred[box]))
                    rows.extend(dict(configuration_id=cid,case_id=case,slice_index=k,**r) for r in score_mask(pred[box],prepared))
            default=configured_prediction(hu,liver,spacing,cfg)['mask']
            packed.append(np.packbits(default[box]))
            rows.extend(dict(configuration_id='default',case_id=case,slice_index=k,**r) for r in score_mask(default[box],prepared))
            np.save(out/'predictions'/f'{stem}.npy',np.stack(packed),allow_pickle=False)
            pd.DataFrame(rows).to_csv(out/'slice_metrics'/f'{stem}.csv',index=False)
            physical.append(dict(case_id=case,slice_index=k,spacing_x_mm=spacing[0],spacing_y_mm=spacing[1],
                area10_mm2=10*spacing[0]*spacing[1],radius1_x_mm=spacing[0],radius1_y_mm=spacing[1],
                radius2_x_mm=2*spacing[0],radius2_y_mm=2*spacing[1]))
            print(f'Generated {stem}: 216 variants + default',flush=True)
        del ct,seg,hu,liver,ref,im,pred,default
    frames=[pd.read_csv(p) for p in sorted((out/'slice_metrics').glob('*.csv'))]
    all_rows=pd.concat(frames,ignore_index=True)
    if len(all_rows)!=len(selection)*(len(records)+1)*2:raise ValueError('Incomplete grid')
    all_rows.to_csv(out/'individual_per_slice.csv',index=False)
    for name,rows in [('historical_parity',parity),('physical_units',physical)]:
        path=out/f'{name}.csv'
        if rows:
            frame=pd.DataFrame(rows)
            if path.exists():frame=pd.concat([pd.read_csv(path),frame],ignore_index=True).drop_duplicates()
            frame.to_csv(path,index=False)
    return all_rows


def validate(out,records,selection,folds):
    evaluator=Evaluator(out,records,selection);ids=list(evaluator.index)[:-1]
    cases=sorted(map(int,selection.case_id.unique()));outer=[];choices=[];traces=[]
    for fold in folds:
        train=fold['train'];test=fold['test']
        print(f"Selecting fold {fold['fold']} on {train}; held out {test}",flush=True)
        single,ss=choose(ids,train,evaluator.evaluate)
        union,us,trace=greedy_union(ids,train,evaluator.evaluate,start=(single,ss))
        sensitive,rs=choose(ids,train,evaluator.evaluate,objective='recall')
        for family,members,score in [('single',single,ss),('union',union,us),('max_sensitivity',sensitive,rs),
                                     ('default',('default',),summarize_search(evaluator.evaluate(('default',),train)))]:
            choices.append(dict(fold=fold['fold'],family=family,members='+'.join(members),train=train,test=test,**score))
            frame=evaluator.evaluate(members,test,criteria=(.1,.25)).copy()
            frame['family']=family;frame['fold']=fold['fold'];frame['members']='+'.join(members)
            outer.append(frame)
        traces.append(dict(fold=fold['fold'],trace=trace))
        evaluator.save_cache()
        pd.DataFrame(choices).to_csv(out/'selected_configurations.csv',index=False)
        pd.concat(outer,ignore_index=True).to_csv(out/'outer_per_slice.csv',index=False)
        atomic_json(out/'union_selection_trace.json',traces)
        print(f"Fold {fold['fold']}: single={single}, union={union}",flush=True)
    outer=pd.concat(outer,ignore_index=True)
    summary=pd.DataFrame([dict(family=f,min_iou=t,**summarize_search(g)) for (f,t),g in outer.groupby(['family','min_iou'])])
    percase=pd.DataFrame([dict(family=f,min_iou=t,case_id=c,**summarize_search(g)) for (f,t,c),g in outer.groupby(['family','min_iou','case_id'])])
    summary.to_csv(out/'validation_summary.csv',index=False);percase.to_csv(out/'validation_per_case.csv',index=False)
    # Inner partitions are pooled because all pipelines are fixed, with no fitted model.
    inner=[]
    for fold in folds:
        for j,patients in enumerate(fold['inner_validation']):
            for cid in ids:
                inner.append(dict(outer_fold=fold['fold'],inner_fold=j,configuration_id=cid,
                    **summarize_search(evaluator.evaluate((cid,),patients))))
    pd.DataFrame(inner).to_csv(out/'inner_validation.csv',index=False)
    print('Selecting future-test presets on all ten patients (exploratory)',flush=True)
    single,ss=choose(ids,cases,evaluator.evaluate)
    union,us,trace=greedy_union(ids,cases,evaluator.evaluate,start=(single,ss))
    sensitive,rs=choose(ids,cases,evaluator.evaluate,objective='recall')
    evaluator.save_cache()
    finalists=dict(single=single,union=union,max_sensitivity=sensitive)
    presets=out/'presets';presets.mkdir(exist_ok=True)
    lookup={r['configuration_id']:r['config'] for r in records}
    for family,members in finalists.items():
        if family!='union':(presets/f'{family}.yaml').write_text(yaml.safe_dump(lookup[members[0]],sort_keys=False))
    for cid in union:(presets/f'{cid}.yaml').write_text(yaml.safe_dump(lookup[cid],sort_keys=False))
    (presets/'union.yaml').write_text(yaml.safe_dump(dict(operation='logical_or',postprocessing='none',
        members=[f'{cid}.yaml' for cid in union],selection_scope='all ten patients; proposal for future testing'),sort_keys=False))
    atomic_json(out/'finalists.json',dict(members=finalists,scores=dict(single=ss,union=us,max_sensitivity=rs),
        union_trace=trace,note='All-cohort selection is exploratory; use validation_summary for held-out estimates.'))
    return evaluator,outer,summary,percase,finalists


def figures_and_report(out,evaluator,outer,summary,individual,records,finalists):
    exploratory=pd.DataFrame([dict(configuration_id=c,min_iou=t,**summarize_search(g))
        for (c,t),g in individual.groupby(['configuration_id','min_iou'])])
    exploratory.to_csv(out/'exploratory_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric,title in zip(axes,['recall','small_recall'],['Todas las lesiones','Lesiones <10 mm']):
        points=exploratory[(exploratory.min_iou==.1)&(exploratory.configuration_id!='default')]
        ax.scatter(points.fp_per_slice,points[metric],s=12,alpha=.25,label='216 variantes (exploratorio)')
        for r in summary[summary.min_iou==.1].itertuples():
            ax.scatter(r.fp_per_slice,getattr(r,metric),marker='X',s=70,label=r.family)
        ax.set(xlabel='FP/corte (escala logarítmica)',ylabel='Sensibilidad',ylim=(0,1),title=title)
        ax.set_xscale('symlog',linthresh=1)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'sensitivity_fp.png',dpi=160);plt.close(fig)
    figs=out/'case94';figs.mkdir(exist_ok=True)
    for key,s in evaluator.slices.items():
        if key[0]!=94:continue
        with np.load(out/'inputs'/f"{s['stem']}.npz") as data:display=data['display'];reference=data['reference']
        fig,axes=plt.subplots(1,5,figsize=(16,4))
        axes[0].imshow(reference.T,cmap='gray',origin='lower',vmin=0,vmax=1);axes[0].set_title('Referencia')
        for ax,family in zip(axes[1:],['default','single','union','max_sensitivity']):
            row=outer[(outer.case_id==94)&(outer.slice_index==key[1])&(outer.family==family)&(outer.min_iou==.1)].iloc[0]
            members=tuple(row.members.split('+'));mask=evaluator.mask(members,key)
            ax.imshow(display.T,cmap='gray',origin='lower',vmin=0,vmax=1)
            overlay=np.zeros((*mask.T.shape,4));overlay[mask.T]=[1,.2,0,.6];ax.imshow(overlay,origin='lower')
            if reference.any():ax.contour(reference.T,levels=[.5],colors='lime',linewidths=.8,origin='lower')
            coverage=f'{row.reference_coverage:.0%}' if np.isfinite(row.reference_coverage) else 'N/A'
            ax.set_title(f'{family}\nCobertura {coverage}; FP {row.FP}; Dice {row.dice:.3f}',fontsize=9)
            # Explicit saved full-size outer masks for exact GUI/measurement reuse.
            with np.load(out/'inputs'/f"{s['stem']}.npz") as data:
                full=np.zeros(tuple(data['full_shape']),bool);origin=data['origin']
            full[origin[0]:origin[0]+mask.shape[0],origin[1]:origin[1]+mask.shape[1]]=mask
            np.save(figs/f"{s['stem']}_{family}.npy",full,allow_pickle=False)
        for ax in axes:ax.axis('off')
        fig.suptitle(f'Caso 94 / corte {key[1]}: selección sin usar el paciente 94 · rojo predicción, verde referencia',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.86));fig.savefig(figs/f"{s['stem']}.png",dpi=140);plt.close(fig)
    lines=['# Barrido clásico: 216 variantes y uniones automáticas','',
        'Evaluación interna posterior a exploración previa: diez pacientes, 50 cortes congelados. '
        'F2 por lesión = 5 TP / (5 TP + 4 FN + FP). No hay límite obligatorio de FP. '
        'Selección interna a IoU ≥0,10; pacientes externos solo puntúan. '
        'Las referencias son componentes 2D, no lesiones volumétricas independientes.','',
        '| Familia | IoU mínimo | F2 | Sensibilidad | Pequeñas | FP/corte | Cobertura media/paciente | Dice medio/paciente |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.family} | {r.min_iou:.2f} | {r.F2:.3f} | {r.recall:.1%} | {r.small_recall:.1%} | '
                     f'{r.fp_per_slice:.2f} | {r.reference_coverage_mean:.1%} | {r.dice_mean:.3f} |')
    lines+=['','## Presets para pruebas futuras (seleccionados con toda la cohorte)','']
    for family,members in finalists.items():lines.append(f'- {family}: {", ".join(members)}.')
    lines+=['','Estos presets no tienen una evaluación independiente nueva. Las métricas anteriores '
            'corresponden al procedimiento de selección por folds, cuyos parámetros pueden variar. '
            'No se modificaron defaults. El grupo se combina por OR, sin limpieza posterior.', '',
            'Los CSV de configuraciones, selección, métricas internas/externas y trazas de unión '
            'permiten auditar el resultado. `historical_parity.csv` registra reproducción exacta '
            'de máscaras históricas equivalentes; `physical_units.csv` registra unidades por corte. '
            '`case94/` contiene comparaciones y máscaras externas, incluido el corte 463 pequeño y 608 negativo.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--benchmark',type=Path,default=ROOT/'results/segmentation/classical-20261104')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):raise ValueError('run-id must be a directory name')
    cfg=load_config(args.config);cfg={k:v for k,v in cfg.items() if not k.startswith('_')}
    cfg['preprocessing_scope']='liver'
    if cfg.get('detection',{}).get('enabled'):raise ValueError('Baseline must be the current classical default')
    selection=pd.read_csv(args.benchmark/'selected_cases_slices.csv');folds=json.loads((args.benchmark/'folds.json').read_text())
    if len(selection)!=50 or selection.duplicated(['case_id','slice_index']).any() or not (selection.groupby('case_id').size()==5).all():raise ValueError('Expected 50 frozen slices')
    validate_folds(folds,list(map(int,selection.case_id.unique())))
    out=Path(cfg['output_dir'])/'classical_search'/args.run_id
    records=configurations(cfg)
    fingerprint=hashlib.sha256((json.dumps(cfg,sort_keys=True)+selection.to_csv(index=False)+json.dumps(folds,sort_keys=True)).encode()).hexdigest()
    if out.exists() and not args.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    statepath=out/'run.json'
    if args.resume:
        if not statepath.exists() or json.loads(statepath.read_text())['fingerprint']!=fingerprint:raise ValueError('Resume input mismatch')
    state=dict(status='running',fingerprint=fingerprint,started_at=datetime.now(timezone.utc).isoformat(),
        config=str(args.config.resolve()),benchmark=str(args.benchmark.resolve()),defaults_changed=False)
    atomic_json(statepath,state)
    try:
        selection.to_csv(out/'selected_cases_slices.csv',index=False);atomic_json(out/'folds.json',folds)
        (out/'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        atomic_json(out/'configurations.json',records)
        pd.DataFrame([dict(configuration_id=r['configuration_id'],profile=r['profile'],
            center=r['config']['preprocessing']['hu_window']['center'],width=r['config']['preprocessing']['hu_window']['width'],
            gaussian_sigma_px=r['config']['preprocessing']['denoising'].get('gaussian_sigma',0),
            equalization=r['config']['preprocessing']['enhancement']['method'],
            **r['config']['segmentation'],pipeline_config=json.dumps(r['config'])) for r in records]).to_csv(out/'configurations.csv',index=False)
        individual=generate(args,out,cfg,records,selection)
        evaluator,outer,summary,percase,finalists=validate(out,records,selection,folds)
        figures_and_report(out,evaluator,outer,summary,individual,records,finalists)
        state.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
        print(summary.to_string(index=False));print(f'Results: {out}',flush=True)
    except BaseException as exc:
        state.update(status='failed',error=str(exc));raise
    finally:atomic_json(statepath,state)


if __name__=='__main__':main()
