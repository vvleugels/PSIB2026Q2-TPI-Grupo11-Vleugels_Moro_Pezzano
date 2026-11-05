"""Prepare/run/resume/analyze the frozen automatic local-candidate experiment.

No computation is launched by installation. --preflight reads headers only;
full runs fingerprint source volumes before segmentation. Linux CPU execution.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import fcntl
import json
import os
import re
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# Avoid multiplying BLAS threads by patient workers. Set before importing NumPy.
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import nibabel as nib
import pandas as pd
import yaml
from psib.data_io import _pairs, _slice_geometry
from psib.classical_search import validate_folds
from psib.local_segmentation import configurations
from psib.local_experiment import (atomic_json,fingerprint,process_patient,analyze,DEPENDENCIES)
from importlib.metadata import version


def resolve(config_path, override=None):
    config_path=config_path.resolve()
    cfg=yaml.safe_load(config_path.read_text())
    if not isinstance(cfg,dict):raise ValueError('Expected YAML mapping')
    result={}
    for key in ('data_dir','output_dir','selection','folds','controls','grid'):
        value=Path(override).expanduser().resolve() if key=='data_dir' and override else Path(cfg[key]).expanduser()
        result[key]=value.resolve() if value.is_absolute() else (config_path.parent/value).resolve()
    result['workers']=int(cfg.get('workers',1))
    return result


def preflight(paths):
    selection=pd.read_csv(paths['selection']);folds=json.loads(paths['folds'].read_text())
    controls=json.loads(paths['controls'].read_text())
    if json.loads(paths['grid'].read_text()) != json.loads(json.dumps(configurations())):
        raise ValueError('Frozen grid differs from implementation; prepare a new protocol instead of silently changing it')
    for column in ('case_id','slice_index'):
        values=selection[column]
        if values.isna().any() or (values<0).any() or (values%1!=0).any():raise ValueError(f'Invalid {column}')
    if len(selection)!=50 or selection.duplicated(['case_id','slice_index']).any() or not (selection.groupby('case_id').size()==5).all():
        raise ValueError('Expected 50 unique frozen central slices, five per patient')
    validate_folds(folds,sorted(map(int,selection.case_id.unique())))
    if set(controls)!= {'default','otsu_059','hu_sensitive'}:raise ValueError('Expected three frozen controls')
    pairs=_pairs(paths['data_dir']);used={}
    for case,group in selection.groupby('case_id'):
        case=int(case)
        if case not in pairs:raise ValueError(f'Missing patient {case}')
        images=[nib.load(str(p)) for p in pairs[case]]
        shapes=[_slice_geometry(image) for image in images]
        import numpy as np
        if shapes[0][0]!=shapes[1][0] or not np.allclose(shapes[0][1],shapes[1][1],atol=1e-4):
            raise ValueError(f'Geometry mismatch for patient {case}')
        if group.slice_index.max()>=shapes[0][0][2]:raise ValueError(f'Out-of-range slice in patient {case}')
        used[case]=pairs[case]
    return selection,folds,controls,used


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/local_candidates/config.yaml')
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--folds',type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    paths=resolve(args.config,args.data_dir)
    for key in ('selection','folds'):
        if getattr(args,key):paths[key]=getattr(args,key).resolve()
    workers=args.workers if args.workers is not None else paths['workers']
    if workers<1:parser.error('--workers must be positive')
    if args.preflight and (args.resume or args.report_only):parser.error('Preflight cannot resume/report')
    if args.preflight:
        selection,folds,controls,pairs=preflight(paths)
        print(json.dumps(dict(status='preflight_ok',patients=len(pairs),slices=len(selection),
            configurations=len(configurations()),controls=list(controls),workers=workers,
            dependencies={name:version(name) for name in DEPENDENCIES},
            note='Headers only; no segmentation, tests, full data hashing or output writes.'),indent=2))
        return
    if not args.run_id or not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):parser.error('Provide a safe --run-id')
    out=paths['output_dir']/args.run_id
    if args.report_only:
        if not out.is_dir():raise FileNotFoundError(out)
        with (out/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            selection=pd.read_csv(out/'selection.csv');folds=json.loads((out/'folds.json').read_text());controls=json.loads((out/'controls.json').read_text())
            result=analyze(out,selection,folds,controls)
            print(json.dumps(result,indent=2))
        return
    selection,folds,controls,pairs=preflight(paths)
    if out.exists() and not args.resume:raise FileExistsError(f'{out}; use a new ID or --resume')
    if args.resume and not out.exists():raise FileNotFoundError('No run to resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Fingerprinting data and implementation before computation...',flush=True)
        signature,manifest=fingerprint(ROOT,pairs,selection,folds,controls)
        statepath=out/'run.json'
        if args.resume:
            previous=json.loads(statepath.read_text())
            if previous['fingerprint']!=signature:raise ValueError('Resume mismatch: data, code, grid, selection, folds or dependencies changed')
        state=dict(status='running',fingerprint=signature,workers=workers,
                   started_at=datetime.now(timezone.utc).isoformat(),defaults_changed=False)
        atomic_json(statepath,state)
        try:
            atomic_json(out/'manifest.json',manifest)
            atomic_json(out/'grid.json',configurations());atomic_json(out/'folds.json',folds);atomic_json(out/'controls.json',controls)
            temp=out/'selection.csv.tmp';selection.to_csv(temp,index=False);temp.replace(out/'selection.csv')
            tasks=[(int(case),list(map(int,g.slice_index)),str(paths['data_dir']),str(out),controls,
                    dict(zip(map(int,g.slice_index),map(int,g.tumor_area_px))) if 'tumor_area_px' in g else {})
                   for case,g in selection.groupby('case_id')]
            performance=[]
            if workers==1:
                for task in tasks:
                    performance.append(process_patient(task));atomic_json(out/'performance.json',performance)
            else:
                # Spawn avoids inheriting image arrays, GUI state, or numerical thread pools.
                import multiprocessing
                with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures=[pool.submit(process_patient,task) for task in tasks]
                    for future in as_completed(futures):
                        performance.append(future.result());atomic_json(out/'performance.json',performance)
            result=analyze(out,selection,folds,controls)
            state.update(status='completed_with_candidate_failures' if result['failed_candidates'] else 'completed',**result)
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            raise
        finally:
            state['finished_at']=datetime.now(timezone.utc).isoformat();atomic_json(statepath,state)
        print(f'Results: {out}/analysis/report.md',flush=True)


if __name__=='__main__':main()
