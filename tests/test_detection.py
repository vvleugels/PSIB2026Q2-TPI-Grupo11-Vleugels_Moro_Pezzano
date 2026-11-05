import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from psib.detection import detect_slice, prepare_detection, detect_prepared, configurations, configured_prediction
from psib.detection_validation import make_folds, nested_evaluation, select_variant
from psib.lesion_eval import match_lesions_one_to_one
from scripts.evaluate_lesions import _case_detection_summary
from scripts.analyze_agreement import agreement_statistics


class DetectionTests(unittest.TestCase):
    def test_constant_empty_and_absent_seeds(self):
        hu = np.full((30,30),100.)
        for method in ('hysteresis','watershed'):
            for liver in (np.ones_like(hu,bool),np.zeros_like(hu,bool)):
                self.assertFalse(detect_slice(hu,liver,(1,1),{'method':method})['mask'].any())
        hu[12:16,12:16]=80
        self.assertFalse(detect_slice(hu,np.ones_like(hu,bool),(1,1),{'high':100})['mask'].any())

    def test_small_multiple_peripheral_and_outside(self):
        hu=np.full((40,40),100.)
        liver=np.zeros_like(hu,bool); liver[5:35,5:35]=True
        hu[5:8,10:13]=40; hu[20:23,20:23]=40; hu[0:3,0:3]=-1000
        for method in ('hysteresis','watershed'):
            result=detect_slice(hu,liver,(1,1),{'method':method,'sigma_mm':0})
            self.assertTrue(result['mask'][6,11]);self.assertTrue(result['mask'][21,21])
            self.assertFalse(result['mask'][~liver].any())
            self.assertEqual(len(result['components']),2)

    def test_physical_area_and_display_independence(self):
        hu=np.full((20,20),100.);hu[10,10]=0
        liver=np.ones_like(hu,bool)
        result=detect_slice(hu,liver,(2,1),{'sigma_mm':0,'min_area_mm2':3})
        self.assertFalse(result['mask'].any())
        self.assertEqual(result['components'][0]['area_mm2'],2)
        self.assertEqual(result['components'][0]['discard_reason'],'area_below_minimum')
        cfg={'detection':{'enabled':True},'preprocessing':{'hu_window':{'center':0,'width':1}}}
        a=configured_prediction(hu,liver,(2,1),cfg)['mask']
        cfg['preprocessing']={'hu_window':{'center':500,'width':4000},'enhancement':{'method':'clahe'}}
        np.testing.assert_array_equal(a,configured_prediction(hu,liver,(2,1),cfg)['mask'])

    def test_watershed_fallback_and_cache(self):
        hu=np.full((20,20),100.);hu[8:11,8:11]=40
        liver=np.ones_like(hu,bool)
        p=prepare_detection(hu,liver,(1,1),{})
        p['score'][:]=3
        result=detect_prepared(p,{'method':'watershed'})
        self.assertTrue(result['mask'].all())
        self.assertEqual(result['fallback'][0]['reason'],'missing_background')
        with self.assertRaises(ValueError):detect_prepared(p,{'sigma_mm':1})
        self.assertEqual(len(configurations()),48)
        for settings in ({'low':2,'high':1},{'sigma_mm':-1},{'high':float('nan')}):
            with self.assertRaises(ValueError):detect_slice(hu,liver,(1,1),settings)

    def test_anisotropic_axes_and_gui_preview(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from scripts.gui import TpiLauncher
        hu=np.full((30,40),100.);hu[10:15,10:18]=40
        liver=np.ones_like(hu,bool)
        cfg={'detection':{'enabled':True,'method':'watershed'}}
        expected=configured_prediction(hu,liver,(.5,1.2),cfg)['mask']
        transposed=configured_prediction(hu.T,liver.T,(1.2,.5),cfg)['mask']
        np.testing.assert_array_equal(expected,transposed.T)
        preview=SimpleNamespace(case_volume=hu[:,:,None],case_labels=liver[:,:,None],
            case_spacing=(.5,1.2,2),slice_var=Mock(),figure_title=Mock(),view_mode=Mock(),
            current_settings=lambda:(cfg,{}),selected_case_id=lambda:1,
            draw_case_image=Mock(),update_slice_metrics=Mock(),refresh_steps=Mock(),append_log=Mock())
        preview.slice_var.get.return_value=0
        preview.view_mode.get.return_value='Después'
        TpiLauncher.show_case_slice(preview)
        np.testing.assert_array_equal(expected,preview.current_prediction)
        preview.append_log.assert_not_called()

    def test_compression_detector_receives_restored_hu(self):
        from scripts.run_compression import process_slice
        hu=np.full((24,24),100.);hu[8:14,8:14]=40
        labels=np.ones_like(hu,np.uint8);labels[8:14,8:14]=2
        cfg={'detection':{'enabled':True},'preprocessing':{'hu_window':{'center':50,'width':200}}}
        rows, images, reference, masks=process_slice(1,0,hu,labels,(1,1,1),cfg,[75])
        restored=images['control'].astype(np.float32)/255*200-50
        expected=configured_prediction(restored,labels>0,(1,1),cfg)['mask']
        np.testing.assert_array_equal(expected,masks['uncompressed_8bit'])
        np.testing.assert_array_equal(expected,masks['png_lossless'])

    def test_matching_cardinality_precedes_iou(self):
        from psib.lesion_eval import _optimal_pairs
        ious=np.array([[.8,.2],[.2,0]])
        self.assertEqual(set(_optimal_pairs(ious,ious >= .1)),{(0,1),(1,0)})

    def test_merge_counts_once(self):
        ref=np.zeros((20,20),bool);ref[5:9,3:6]=1;ref[5:9,10:13]=1
        pred=np.zeros_like(ref);pred[5:9,3:13]=1
        matches=match_lesions_one_to_one(pred,ref,(1,1),.1)
        self.assertEqual((matches.status=='TP').sum(),1)
        self.assertEqual((matches.status=='FN').sum(),1)
        self.assertEqual(len(match_lesions_one_to_one(np.zeros_like(ref),np.zeros_like(ref),(1,1))),0)

    def test_metrics_regressions(self):
        matches=pd.DataFrame({'case_id':[1,1],'status':['FP','FN']})
        self.assertEqual(_case_detection_summary(matches,[1]).iloc[0].F1,0)
        data=pd.DataFrame({'status':['TP','TP'],'size_class':['<10','<10'],
            'area_ref_mm2':[10,20],'area_pred_mm2':[12,22],'max_feret_ref_mm':[5,7],'max_feret_pred_mm':[6,8]})
        self.assertEqual(agreement_statistics(data).iloc[0].mean_of_pair_means,16)

    def test_nested_no_outer_selection_leakage(self):
        folds=make_folds(range(10));rows=[]
        for c in range(10):
            for cid in ('det_000','det_001','historical','otsu'):
                for iou in (.1,.25):
                    rows.append(dict(case_id=c,slice_index=0,configuration_id=cid,min_iou=iou,
                        TP=1,FP=0 if cid=='det_000' else 1,FN=1,small_TP=1,small_FN=1,
                        pred_area=10,reference_area=20,intersection=10,union=20))
        metrics=pd.DataFrame(rows)
        chosen=nested_evaluation(metrics,folds)[0]
        changed=metrics.copy(); idx=changed.case_id.isin(folds[0]['test']) & (changed.configuration_id=='det_001')
        changed.loc[idx,['TP','small_TP']]=10000
        revised=nested_evaluation(changed,folds)[0]
        self.assertEqual(list(chosen[chosen.fold==0].configuration_id),list(revised[revised.fold==0].configuration_id))
        self.assertEqual(chosen.iloc[0].configuration_id,'det_000')
        too_many=metrics[metrics.min_iou==.1].copy();too_many.FP=99
        self.assertIsNone(select_variant(too_many,8))
        # Unlimited selection must actually evaluate a detector, not abstain.
        unlimited = nested_evaluation(metrics.assign(FP=99), folds, budgets=(8, float('inf')))
        choices, _, summary = unlimited[:3]
        self.assertTrue(choices[choices.budget == 8].configuration_id.isna().all())
        self.assertTrue(choices[choices.budget == float('inf')].configuration_id.notna().all())
        scored = summary[(summary.family == 'detector') & (summary.budget == float('inf'))]
        self.assertTrue((scored.fp_per_slice == 99).all())
        self.assertTrue((scored.recall == .5).all())
        self.assertFalse(unlimited[-1]['eligible_for_promotion'])
        # An unlimited-only analysis does not silently revise the original gate.
        self.assertFalse(nested_evaluation(metrics, folds, budgets=(float('inf'),))[-1]['eligible_for_promotion'])

    def test_runner_and_3d_use_same_masks(self):
        from scripts.run_segmentation import run_experiment
        from scripts.evaluate_morphology_3d import resolve_configuration, _measure_case
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();out=root/'out';out.mkdir()
            hu=np.full((16,16,2),100.,dtype=np.float32);hu[5:9,5:9,:]=20
            labels=np.ones_like(hu,dtype=np.uint8);labels[5:9,5:9,:]=2
            nib.save(nib.Nifti1Image(hu,np.eye(4)),data/'volume-1.nii.gz')
            nib.save(nib.Nifti1Image(labels,np.eye(4)),data/'segmentation-1.nii.gz')
            cfg={'data_dir':str(data),'output_dir':str(out),'detection':{'enabled':True,'method':'watershed'}}
            path=root/'cfg.yaml';path.write_text(yaml.safe_dump(cfg))
            pd.DataFrame({'case_id':[1,1],'slice_index':[0,1]}).to_csv(out/'selected_cases_slices.csv',index=False)
            with patch('scripts.run_detection.export_slice_comparison'):
                run=run_experiment(path,'tiny')
            cid,settings=resolve_configuration(run,'det_000')
            dest=root/'3d';dest.mkdir()
            _measure_case(1,data,settings,cfg,8,dest,cid)
            volume=np.load(dest/'case_001_masks_3d.npz')['prediction_raw']
            for k in range(2):
                expected=configured_prediction(hu[:,:,k],labels[:,:,k]>0,(1,1),cfg)['mask']
                np.testing.assert_array_equal(expected,volume[:,:,k])
                np.testing.assert_array_equal(expected,np.load(run/'predictions'/cid/f'case_001_slice_{k:03d}.npy'))
            self.assertEqual(json.loads((run/'run.json').read_text())['status'],'completed')

if __name__=='__main__':unittest.main()
