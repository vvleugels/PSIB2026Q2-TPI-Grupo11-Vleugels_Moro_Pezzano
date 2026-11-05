import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from psib.hybrid_segmentation import (combine_masks, prepare_components, detect_hybrid,
                                      configurations, validate_settings)
from psib.detection import configured_prediction
from psib.lesion_eval import match_lesions_one_to_one

ROOT=Path(__file__).resolve().parents[1]
CONTROLS=json.loads((ROOT/'tests/fixtures/hybrid_controls.json').read_text())


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
        public=match_lesions_one_to_one(mask,reference,(1.,1.),.1)
        self.assertEqual((public.status=='TP').sum(),1)
        self.assertEqual((public.status=='FN').sum(),1)

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


if __name__=='__main__':unittest.main()
