"""Selection, cache, GUI lifecycle and opt-in real-worker regression tests."""
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
from psib.local_experiment import select,seal,verify_seal,fingerprint,process_patient,lesion_diagnostics,choose_fold
from psib.lesion_eval import prepare_reference


class LocalExperimentTests(unittest.TestCase):
    def score(self,dice=.4,recall=.7,small=.5,fp=100,failed=0):
        return dict(dice_mean=dice,recall=recall,small_recall=small,pixel_FP=fp,fp_per_slice=5.,failed_candidates=failed)

    def test_guardrails_fallback_and_unconstrained(self):
        scores={'default':self.score(),'local_000':self.score(.8,.6,.5),'local_001':self.score(.9,.8,.4)}
        self.assertEqual(select(scores)[0],'default')
        self.assertEqual(select(scores,False)[0],'local_001')
        scores['local_002']=self.score(.6,.7,.5)
        self.assertEqual(select(scores)[0],'local_002')
        scores['local_003']=self.score(.9,.9,.9,failed=1)
        self.assertEqual(select(scores)[0],'local_002')
        scores['local_004']=self.score(.6,.7,.5,fp=50)
        self.assertEqual(select(scores)[0],'local_004')

    def test_seal_detects_changed_missing_extra_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            self.assertFalse(verify_seal(path))
            (path/'data').write_text('first');seal(path);self.assertTrue(verify_seal(path))
            (path/'data').write_text('second')
            with self.assertRaises(ValueError):verify_seal(path)
            seal(path);(path/'extra').write_text('unexpected')
            with self.assertRaises(ValueError):verify_seal(path)
            (path/'extra').unlink();(path/'data').unlink()
            with self.assertRaises(ValueError):verify_seal(path)

    def test_fingerprint_sensitive_to_source_data_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir();(root/'scripts').mkdir()
            code=root/'src/model.py';code.write_text('# first')
            volume=root/'volume';volume.write_text('data')
            args=({1:(volume,volume)},pd.DataFrame([dict(case_id=1,slice_index=0)]),[],{})
            first,_=fingerprint(root,*args)
            code.write_text('# changed');second,_=fingerprint(root,*args)
            self.assertNotEqual(first,second)
            volume.write_text('changed');third,_=fingerprint(root,*args)
            self.assertNotEqual(second,third)
            args[1].loc[0,'slice_index']=1
            fourth,_=fingerprint(root,*args);self.assertNotEqual(third,fourth)

    def test_stage_gains_and_losses(self):
        ref=np.zeros((15,15),bool);ref[3:6,3:6]=True;ref[9:12,9:12]=True
        support=np.zeros_like(ref);support[3:6,3:6]=True
        result=dict(candidates=support,mask=np.zeros_like(ref))
        rows=lesion_diagnostics(np.full(ref.shape,100.),np.ones_like(ref),ref,(1,1),prepare_reference(ref,(1,1)),result)
        self.assertEqual([r['stage'] for r in rows],['lost_during_refinement','no_candidate'])

    def test_fold_selection_never_reads_test_scores(self):
        frame=pd.DataFrame([dict(case_id=case,configuration_id=cid,quality=quality)
            for case in (1,2) for cid,quality in [('default',.4),('local_000',.6)]])
        observed=[]
        def summarize_training(group):
            observed.extend(group.case_id.tolist())
            return self.score(dice=float(group.quality.mean()))
        fold=dict(train=[1],test=[2])
        with patch('psib.local_experiment.summarize',side_effect=summarize_training):
            first=choose_fold(frame,fold)
            frame.loc[frame.case_id==2,'quality']=-999
            second=choose_fold(frame,fold)
        self.assertEqual(first,second);self.assertEqual(set(observed),{1})

    def test_stale_gui_completion_never_applies(self):
        from scripts.gui import TpiLauncher
        app=Mock();app.local_preview_token=2;future=Mock()
        TpiLauncher.finish_local_preview(app,1,future)
        future.result.assert_not_called();future.done.assert_not_called()

    def test_context_unsupported_dispatch_is_explicit(self):
        from psib.detection import configured_prediction
        with self.assertRaisesRegex(ValueError,'vecinos'):
            configured_prediction(np.zeros((4,4)),np.ones((4,4),bool),(1,1),
                                  {'detection':dict(enabled=True,method='local_graphcut',context=True)})

    @unittest.skipUnless(os.environ.get('PSIB_RUN_LOCAL_INTEGRATION')=='1','Run explicitly on the destination PC')
    def test_serial_parallel_and_resume_identical_masks(self):
        import nibabel as nib
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir()
            for case in (1,2):
                hu=np.full((20,20,3),100.,np.float32);hu[7:12,7:12,:]=40
                labels=np.ones(hu.shape,np.uint8);labels[7:12,7:12,:]=2
                nib.save(nib.Nifti1Image(hu,np.diag([1.,1.,2.,1.])),data/f'volume-{case}.nii')
                nib.save(nib.Nifti1Image(labels,np.diag([1.,1.,2.,1.])),data/f'segmentation-{case}.nii')
            serial=root/'serial';parallel=root/'parallel';serial.mkdir();parallel.mkdir()
            controls={'default':{'detection':{'enabled':False},'segmentation':{'method':'otsu'}}}
            for case in (1,2):process_patient((case,[1],str(data),str(serial),controls))
            with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
                tasks=[(c,[1],str(data),str(parallel),controls) for c in (1,2)]
                list(pool.map(process_patient,tasks))
            for case in (1,2):
                relative=Path(f'case_{case:03d}/slice_0001/masks.npz')
                with np.load(serial/relative) as a,np.load(parallel/relative) as b:
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                resumed=process_patient((case,[1],str(data),str(parallel),controls))
                self.assertEqual(resumed['status'],'cached')


if __name__=='__main__':unittest.main()
