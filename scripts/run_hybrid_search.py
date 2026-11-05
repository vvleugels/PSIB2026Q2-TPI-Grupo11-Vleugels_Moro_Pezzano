"""Run two frozen hybrid stages, resume checkpoints, or re-audit saved masks."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import yaml
from psib.hybrid_experiment import preflight, process_slice, analyze, PROTOCOL
from psib.hybrid_segmentation import configurations
from psib.local_experiment import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/local_candidates/local-v2-liver')
    parser.add_argument('--bilateral',type=Path,default=ROOT/'results/otsu-bilateral-50/segmentation/otsu-bilateral-50')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/hybrid_candidates')
    parser.add_argument('--run-id')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--resume',action='store_true')
    mode.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if not args.preflight and (not args.run_id or not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id)):
        parser.error('Provide a safe --run-id')
    out=args.output_dir/args.run_id if args.run_id else None
    if args.report_only:
        if not out.is_dir():raise FileNotFoundError(out)
        with (out/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            print(json.dumps(analyze(out),indent=2))
        return
    print('Verifying frozen selection, controls, geometry and source seals...',flush=True)
    selection,folds,controls,bilateral_cfg,signature,manifest=preflight(ROOT,args.source,args.bilateral)
    records=configurations(controls)
    if args.preflight:
        print(json.dumps(dict(status='preflight_ok',patients=10,slices=len(selection),stage1=6,stage2=144,controls=5,fingerprint=signature),indent=2))
        return
    if out.exists() and not args.resume:raise FileExistsError(f'{out}; use a new ID or --resume')
    if args.resume and not out.exists():raise FileNotFoundError('No run to resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            if json.loads((out/'run.json').read_text())['fingerprint']!=signature:
                raise ValueError('Resume mismatch: code, versions, frozen inputs or protocol changed')
        else:
            atomic_json(out/'manifest.json',manifest)
            atomic_json(out/'protocol.json',PROTOCOL)
            atomic_json(out/'configurations.json',records)
            atomic_json(out/'controls.json',controls)
            atomic_json(out/'folds.json',folds)
            selection.to_csv(out/'selection.csv',index=False)
            cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
            cfg['data_dir']=str((ROOT/'data/Selected_database').resolve())
            cfg['output_dir']=str((ROOT/'results').resolve())
            cfg['selection']['n_cases']=10
            cfg['edge_analysis']={'operator':'none'}
            (out/'base_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        state=dict(status='running',fingerprint=signature,started_at=datetime.now(timezone.utc).isoformat(),defaults_changed=False)
        atomic_json(out/'run.json',state)
        try:
            for i,row in enumerate(selection.itertuples(),1):
                status=process_slice(args.source,args.bilateral,out,row,controls,bilateral_cfg,records)
                print(f'{i}/50 case {row.case_id}, slice {row.slice_index}: {status}',flush=True)
            print('Recomputing metrics from all saved masks; selecting by held-out patient folds...',flush=True)
            state['analysis']=analyze(out)
            state['status']='completed'
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            state['finished_at']=datetime.now(timezone.utc).isoformat()
            atomic_json(out/'run.json',state)
        print(json.dumps(state,indent=2))


if __name__=='__main__':main()
