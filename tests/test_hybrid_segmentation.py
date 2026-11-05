import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from psib.hybrid_segmentation import (combine_masks, prepare_components, detect_hybrid,
                                      configurations, validate_settings)
from psib.detection import configured_prediction
from psib.hybrid_experiment import (select_scores, choose_fold, process_slice, unpack,
                                   analyze, evaluate_mask, preflight, PROTOCOL, validate_run_metadata)
from psib.local_experiment import seal
from psib.lesion_eval import prepare_reference, match_lesions_one_to_one
from psib.detection_validation import make_folds

ROOT=Path(__file__).resolve().parents[1]
CONTROLS=json.loads((ROOT/'experiments/local_candidates/controls.json').read_text())


class HybridMaskTests(unittest.TestCase):
    def test_physical_diameter_and_exact_boundary(self):
        mask=np.zeros((12,12),bool);mask[3:7,4:6]=True
        liver=np.ones_like(mask);base=np.zeros_like(mask)
        diameter=prepare_components(mask,liver,(2.,.5))[1][1]
        self.assertGreater(diameter,prepare_components(mask,liver,(1.,.5))[1][1])
        self.assertFalse(combine_masks(base,mask,liver,(2.,.5),diameter)['mask'].any())
        self.assertTrue(np.array_equal(combine_masks(base,mask,liver,(2.,.5),diameter+1e-6)['mask'],mask))

    def test_diagonal_contact_and_complete_component(self):
        base=np.zeros((12,12),bool);base[2,2]=True
        candidate=np.zeros_like(base);candidate[3:5,3]=True;candidate[9,9]=True
        liver=np.ones_like(base)
        separate=combine_masks(base,candidate,liver,(1.,1.),10.,'disjoint')
        self.assertFalse(separate['mask'][3,3]);self.assertTrue(separate['mask'][9,9])
        full=combine_masks(base,candidate,liver,(1.,1.),10.,'complete')
        self.assertTrue(np.array_equal(full['mask'],base|candidate))
        self.assertEqual(separate['components'][0]['reason'],'touches_base')

    def test_fusion_can_reduce_detection_count(self):
        reference=np.zeros((14,14),bool);reference[3:6,2:5]=True;reference[3:6,8:11]=True
        bridge=np.zeros_like(reference);bridge[4,4:9]=True
        mask=combine_masks(reference,bridge,np.ones_like(reference),(1.,1.),10.,'complete')['mask']
        rows,lesions=evaluate_mask(mask,reference,(1.,1.),prepare_reference(reference,(1.,1.)))
        self.assertEqual(rows[0]['TP'],1);self.assertEqual(rows[0]['FN'],1)
        self.assertEqual(rows[0]['fusion_components'],1)
        public=match_lesions_one_to_one(mask,reference,(1.,1.),.1)
        self.assertEqual((public.status=='TP').sum(),rows[0]['TP'])

    def test_empty_liver_and_clipping(self):
        base=np.ones((8,8),bool);candidate=base.copy();liver=np.zeros_like(base)
        self.assertFalse(combine_masks(base,candidate,liver,(1.,1.))['mask'].any())
        liver[2:6,2:6]=True
        self.assertTrue(np.array_equal(combine_masks(base,candidate,liver,(1.,1.))['mask'],liver))
        empty=np.zeros_like(base)
        result=combine_masks(empty,empty,liver,(1.,1.))
        self.assertEqual(result['components'],[])

    def test_dispatch_determinism_and_branch_identity(self):
        rng=np.random.default_rng(17);hu=rng.normal(100,3,(32,32));hu[10:15,12:17]=40
        liver=np.ones_like(hu,bool);cfg=configurations(CONTROLS)[6]['settings']
        a=detect_hybrid(hu,liver,(.8,1.2,2.),cfg)
        b=configured_prediction(hu,liver,(.8,1.2,2.),{'detection':dict(enabled=True,**cfg)})
        for key in ('mask','base_mask','candidate_mask','added_mask'):
            np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue(np.all(a['mask'][a['base_mask']]))
        blank=detect_hybrid(hu,np.zeros_like(liver),(1.,1.),cfg)
        self.assertFalse(blank['mask'].any())

    def test_fixed_preset_batch_matches_shared_dispatch(self):
        import nibabel as nib
        import yaml
        from scripts.run_segmentation import run_experiment
        hu=np.full((16,16,1),100.,dtype=np.float32);hu[5:9,5:9,0]=40
        labels=np.ones_like(hu,dtype=np.uint8);labels[5:9,5:9,0]=2
        settings=configurations(CONTROLS)[0]['settings']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();(root/'results').mkdir()
            nib.save(nib.Nifti1Image(hu,np.eye(4)),root/'data/volume-1.nii.gz')
            nib.save(nib.Nifti1Image(labels,np.eye(4)),root/'data/segmentation-1.nii.gz')
            pd.DataFrame([dict(case_id=1,slice_index=0)]).to_csv(root/'results/selected_cases_slices.csv',index=False)
            cfg=dict(data_dir='data',output_dir='results',detection=dict(enabled=True,**settings))
            path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg))
            result=run_experiment(path,'hybrid')
            actual=np.load(result/'predictions/hybrid_preset/case_001_slice_000.npy')
            expected=configured_prediction(hu[:,:,0],labels[:,:,0]>0,(1.,1.,1.),cfg)['mask']
            np.testing.assert_array_equal(actual,expected)
            with self.assertRaises(ValueError):run_experiment(path,'unsupported',True)

    def test_settings_and_grid(self):
        records=configurations(CONTROLS)
        self.assertEqual(sum(r['stage']==1 for r in records),6)
        self.assertEqual(sum(r['stage']==2 for r in records),144)
        for r in records:validate_settings(r['settings'])
        for changes in ({'max_diameter_mm':0},{'addition_rule':'unknown'},{'max_diameter_mm':float('nan')},
                        {'base_config':{'detection':{'enabled':True}}}):
            with self.assertRaises(ValueError):validate_settings(records[0]['settings']|changes)
        with self.assertRaises(ValueError):combine_masks(np.zeros((2,2)),np.zeros((3,3)),np.ones((2,2)),(1,1))


class HybridSelectionTests(unittest.TestCase):
    def test_floor_ranking_and_explicit_fallback(self):
        common=dict(small_recall=.5,false_area_mm2=20.,fp_per_slice=2.)
        scores={'otsu_059':dict(common,dice_mean=.39,recall=.7),
                'a':dict(common,dice_mean=.399,recall=1.),
                'b':dict(common,dice_mean=.4,recall=.8),
                'c':dict(common,dice_mean=.6,recall=.75)}
        self.assertEqual(select_scores(scores,set(scores)),('b','selected'))
        self.assertEqual(select_scores(scores,{'a','otsu_059'}),('otsu_059','no_variant_reaches_dice_floor'))
        scores['d']=dict(scores['b'],small_recall=.6)
        self.assertEqual(select_scores(scores,set(scores))[0],'d')

    def test_held_out_outcomes_never_choose_parameters(self):
        records=configurations(CONTROLS)[:1];cid=records[0]['configuration_id']
        rows=[]
        for case in range(10):
            for config in ('default','otsu_059',cid):
                rows.append(dict(case_id=case,configuration_id=config,TP=2 if config==cid else 1,FP=0,FN=0 if config==cid else 1,
                    small_TP=1,small_FN=0,intersection=8,pred_area=10,reference_area=10,union=12,
                    pixel_area_mm2=1.,failed_candidates=0))
        frame=pd.DataFrame(rows);fold=make_folds(range(10))[0]
        before=choose_fold(frame,fold,records,'stage1')[:2]
        frame.loc[frame.case_id.isin(fold['test']),['TP','FP','FN','intersection']]=999
        self.assertEqual(before,choose_fold(frame,fold,records,'stage1')[:2])
        self.assertEqual(before[0],cid)

    def test_checkpoint_resume_and_report_from_masks(self):
        controls=CONTROLS
        records=[configurations(controls)[0],configurations(controls)[6]]
        hu=np.full((24,24),100.);hu[7:12,9:14]=40.
        liver=np.ones_like(hu,bool);ref=np.zeros_like(liver);ref[7:12,9:14]=True
        spacing=np.array([1.,1.,2.]);bilateral_cfg=controls['otsu_059']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';bilateral=root/'bilateral';out=root/'out';out.mkdir()
            pred_dir=bilateral/'predictions/cfg_000';pred_dir.mkdir(parents=True)
            row=SimpleNamespace(case_id=0,slice_index=0)
            folder=source/'case_000/slice_0000';folder.mkdir(parents=True)
            np.savez_compressed(folder/'inputs.npz',hu=hu,liver=liver,reference=ref,spacing=spacing)
            masks={cid:configured_prediction(hu,liver,spacing,cfg)['mask'] for cid,cfg in controls.items()}
            np.savez_compressed(folder/'masks.npz',ids=list(masks),shape=hu.shape,predictions=np.stack([np.packbits(m) for m in masks.values()]))
            np.save(pred_dir/'case_000_slice_000.npy',masks['otsu_059'])
            self.assertEqual(process_slice(source,bilateral,out,row,controls,bilateral_cfg,records),'completed')
            first=(out/'case_000/slice_0000/masks.npz').read_bytes()
            with patch('psib.hybrid_experiment.configured_prediction',side_effect=AssertionError('must reuse')):
                self.assertEqual(process_slice(source,bilateral,out,row,controls,bilateral_cfg,records),'cached')
            self.assertEqual(first,(out/'case_000/slice_0000/masks.npz').read_bytes())
            # 10 disjoint patients, five synthetic slices each; analysis has no dependency on source images.
            import shutil
            selection=[]
            template=out/'case_000/slice_0000'
            for c in range(10):
                for k in range(5):
                    selection.append(dict(case_id=c,slice_index=k))
                    target=out/f'case_{c:03d}'/f'slice_{k:04d}'
                    if c==0 and k==0:continue
                    shutil.copytree(template,target)
                    frame=pd.read_csv(target/'metrics.csv');frame['case_id']=c;frame['slice_index']=k
                    frame.to_csv(target/'metrics.csv',index=False);seal(target)
            pd.DataFrame(selection).to_csv(out/'selection.csv',index=False)
            for name,value in [('folds',make_folds(range(10))),('controls',controls),('configurations',records),('protocol',PROTOCOL),
                               ('manifest',dict(selection=selection,folds=make_folds(range(10)),controls=controls,configurations=records,protocol=PROTOCOL))]:
                (out/f'{name}.json').write_text(json.dumps(value))
            (out/'base_config.yaml').write_text('data_dir: data\noutput_dir: results\n')
            result=analyze(out,figures=False)
            self.assertEqual(result['masks_recomputed'],350)
            summary=pd.read_csv(out/'analysis/summary.csv')
            again=analyze(out,figures=False)
            pd.testing.assert_frame_equal(summary,pd.read_csv(out/'analysis/summary.csv'))
            self.assertEqual(result,again)
            (out/'protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'metadata differs'):validate_run_metadata(out)
            # A corrupt sealed checkpoint must not silently resume.
            (template/'metrics.csv').write_text('corrupt')
            with self.assertRaises(ValueError):process_slice(source,bilateral,out,row,controls,bilateral_cfg,records)


if __name__=='__main__':unittest.main()
