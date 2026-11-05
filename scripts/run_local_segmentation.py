"""Run one local or hybrid preset through the normal batch entrypoint."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import json
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yaml
from psib.config import load_config
from psib.data_io import load_case
from psib.detection import configured_prediction
from psib.local_segmentation import validate_settings,context_from_volume
from psib.serialization import atomic_json
from psib.metrics import dice_score,iou_score,reference_coverage
from scripts.run_segmentation import read_selection,aggregate_metrics,export_slice_comparison,_metadata
from psib.preprocessing import apply_hu_window


def run_experiment(config_path,run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):raise ValueError('Invalid run ID')
    cfg=load_config(config_path)
    validator=validate_settings
    if cfg['detection'].get('method') == 'hybrid_size':
        from psib.hybrid_segmentation import validate_settings as validator
    local=validator({k:v for k,v in cfg['detection'].items() if k!='enabled'})
    cfg['detection']=dict(enabled=True,**local)
    preset_id='hybrid_preset' if local['method']=='hybrid_size' else 'local_preset'
    selection=read_selection(Path(cfg['_output_dir'])/'selected_cases_slices.csv')
    out=Path(cfg['_output_dir'])/'segmentation'/run_id;out.mkdir(parents=True,exist_ok=False)
    state=_metadata(config_path,run_id,False);rows=[];failures=0
    atomic_json(out/'run.json',state)
    try:
        effective={k:v for k,v in cfg.items() if not k.startswith('_')}
        (out/'effective_config.yaml').write_text(yaml.safe_dump(effective,sort_keys=False))
        selection.to_csv(out/'selected_cases_slices.csv',index=False)
        pd.DataFrame([dict(configuration_id=preset_id,method=local['method'],pipeline_config=json.dumps(effective))]).to_csv(out/'configurations.csv',index=False)
        for case,group in selection.groupby('case_id'):
            volume,labels,spacing=load_case(int(case),cfg['data_dir']);liver_volume=labels>0
            for k in group.slice_index.astype(int):
                hu=volume[:,:,k];liver=liver_volume[:,:,k];reference=labels[:,:,k]==2
                context=context_from_volume(volume,liver_volume,k,spacing) if local.get('context',False) else None
                result=configured_prediction(hu,liver,spacing,cfg,context=context);mask=result['mask']
                failures+=result['failed_candidates']
                stem=f'case_{case:03d}_slice_{k:03d}'
                folder=out/'diagnostics';folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{stem}.npz',seeds=result['seeds'],candidates=result['candidates'],prediction=mask,
                                    **{key:result[key] for key in ('base_mask','candidate_mask','added_mask','raw_mask') if key in result})
                atomic_json(folder/f'{stem}.json',dict(components=result['components'],context_note=result['context_note'],
                    **{key:result[key] for key in ('hu_effective_parameters','fallback','pipeline_note') if key in result}))
                rows.append(dict(configuration_id=preset_id,case_id=case,slice_index=k,pred_area=int(mask.sum()),
                    reference_area=int(reference.sum()),intersection=int((mask & reference).sum()),union=int((mask|reference).sum()),
                    dice=dice_score(mask,reference),iou=iou_score(mask,reference),reference_coverage=reference_coverage(mask,reference)))
                display=apply_hu_window(hu,100,200)
                export_slice_comparison(out,preset_id,int(case),k,display,result.get('processed_image',display),'none',None,reference,mask,
                                        processed_title=('Configured preprocessing + morphology' if 'processed_image' in result
                                                         else 'HU display; segmentation uses raw HU'))
            print(f'Evaluado caso {case}: {preset_id}',flush=True)
        frame=pd.DataFrame(rows);cases,summary=aggregate_metrics(frame)
        for name,table in [('metrics_per_slice',frame),('metrics_per_case',cases),('summary',summary)]:table.to_csv(out/f'{name}.csv',index=False)
        state.update(status='completed_with_candidate_failures' if failures else 'completed',failed_candidates=failures)
    except BaseException as exc:
        state.update(status='failed',error=str(exc));raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(out/'run.json',state)
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();run_experiment(args.config,args.run_id)
