import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from psib.classical_search import configurations,score_mask,summarize_search,choose,greedy_union,validate_folds
from psib.lesion_eval import prepare_reference,match_lesions_one_to_one
from psib.detection_validation import make_folds
from psib.detection import configured_prediction
from scripts.run_segmentation import iter_predictions
from psib.preprocessing import prepare_denoised_slice,enhance


class ClassicalSearchTests(unittest.TestCase):
    def test_grid_and_existing_pipeline_equivalence(self):
        records=configurations({})
        self.assertEqual(len(records),216)
        self.assertEqual(len({r['configuration_id'] for r in records}),216)
        hu=np.full((25,30),100.);hu[5:10,6:12]=30.;hu[15:17,20:22]=20.
        liver=np.zeros_like(hu,dtype=bool);liver[2:-2,2:-2]=True
        for start in (0,48,210):
            chunk=records[start:start+6];c=chunk[0]['config']
            image=enhance(prepare_denoised_slice(hu,c),c['preprocessing']['enhancement'])
            variants=[dict(configuration_id=r['configuration_id'],**r['config']['segmentation']) for r in chunk]
            for (record,pred),r in zip(iter_predictions(image,liver,variants),chunk):
                expected=configured_prediction(hu,liver,(.7,.9),r['config'])['mask']
                np.testing.assert_array_equal(pred,expected)
                self.assertFalse((pred & ~liver).any())

    def test_fast_matches_equal_public_evaluator(self):
        rng=np.random.default_rng(2026)
        for shape in ((8,9),(17,12)):
            for density in (0,.1,.6,1):
                ref=rng.random(shape)<density;pred=rng.random(shape)<.15
                prepared=prepare_reference(ref,(.7,1.1))
                for row in score_mask(pred,prepared):
                    m=match_lesions_one_to_one(pred,ref,(.7,1.1),row['min_iou'],prepared_reference=prepared)
                    for key in ('TP','FP','FN'):self.assertEqual(row[key],int((m.status==key).sum()))
                    self.assertEqual(row['small_TP'],int(((m.status=='TP')&(m.size_class=='<10')).sum()))
                    if not ref.any():self.assertTrue(np.isnan(row['reference_coverage']))

    def test_union_recomputed_not_added_and_empty(self):
        ref=np.zeros((9,9),bool);ref[2:7,2:7]=True
        a=np.zeros_like(ref);a[2:7,2:4]=True
        b=np.zeros_like(ref);b[2:7,4:7]=True
        prepared=prepare_reference(ref,(1,1))
        self.assertEqual(score_mask(a|b,prepared)[0]['TP'],1)
        self.assertEqual(score_mask(a|b,prepared)[0]['FP'],0)
        self.assertEqual(score_mask(np.zeros_like(a),prepared)[0]['FN'],1)
        self.assertEqual(summarize_search(pd.DataFrame([dict(case_id=1,**score_mask(a|b,prepared)[0])]))['F2'],1.)

    def test_selection_only_requests_training_cases(self):
        seen=[]
        def evaluate(members,cases):
            seen.extend(cases)
            recovered=2 if 'b' in members and 'a' in members else 1
            return pd.DataFrame([dict(case_id=c,TP=recovered,FP=0,FN=2-recovered,
                small_TP=1,small_FN=0,reference_area=20,pred_area=10*recovered,
                intersection=10*recovered,union=20) for c in cases])
        start=choose(['a','b','c'],[1,2],evaluate)
        union,score,trace=greedy_union(['a','b','c'],[1,2],evaluate,start)
        self.assertEqual(start[0],('a',));self.assertEqual(union,('a','b'))
        self.assertEqual(score['F2'],1);self.assertEqual(len(trace),2)
        self.assertEqual(set(seen),{1,2})

    def test_frozen_folds_reject_overlap(self):
        folds=make_folds(range(10));validate_folds(folds,list(range(10)))
        folds[0]['test'][0]=folds[0]['train'][0]
        with self.assertRaises(ValueError):validate_folds(folds,list(range(10)))

    def test_packed_union_cache_and_patient_isolation(self):
        from scripts.run_classical_search import Evaluator
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'inputs').mkdir();(out/'predictions').mkdir()
            ref=np.zeros((9,9),bool);ref[2:7,2:7]=True
            a=np.zeros_like(ref);a[2:7,2:4]=True
            b=np.zeros_like(ref);b[2:7,4:7]=True
            rows=[]
            for case in (1,2):
                stem=f'case_{case:03d}_slice_000'
                np.savez(out/'inputs'/f'{stem}.npz',reference=ref,spacing=(1,1),liver=np.ones_like(ref))
                np.save(out/'predictions'/f'{stem}.npy',np.stack([np.packbits(m) for m in (a,b,a)]))
                for cid,mask in [('a',a),('b',b),('default',a)]:
                    rows.extend(dict(configuration_id=cid,case_id=case,slice_index=0,**r)
                        for r in score_mask(mask,prepare_reference(ref,(1,1))))
            pd.DataFrame(rows).to_csv(out/'individual_per_slice.csv',index=False)
            selection=pd.DataFrame(dict(case_id=[1,2],slice_index=[0,0]))
            records=[dict(configuration_id=c) for c in ('a','b')]
            evaluator=Evaluator(out,records,selection)
            np.testing.assert_array_equal(evaluator.mask(('a','b'),(1,0)),a|b)
            actual=evaluator.evaluate(('a','b'),[1],criteria=(.1,.25))
            self.assertEqual(set(actual.case_id),{1})
            self.assertTrue((actual.TP==1).all());self.assertTrue((actual.FP==0).all())
            self.assertNotIn((('a','b'),2,0),evaluator.cache)
            evaluator.save_cache()
            resumed=Evaluator(out,records,selection)
            pd.testing.assert_frame_equal(actual,resumed.evaluate(('a','b'),[1],criteria=(.1,.25)))

    def test_small_component_removed_by_area(self):
        records=configurations({});hu=np.full((20,20),100.);hu[5:7,5:7]=0
        liver=np.ones_like(hu,dtype=bool)
        keep=configured_prediction(hu,liver,(1,1),records[0]['config'])['mask']
        drop=configured_prediction(hu,liver,(1,1),records[1]['config'])['mask']
        self.assertEqual(keep.sum(),4);self.assertEqual(drop.sum(),0)

if __name__=='__main__':unittest.main()
