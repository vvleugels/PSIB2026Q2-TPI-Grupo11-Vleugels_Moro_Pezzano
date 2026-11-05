"""Prepared tests: execute on the destination PC, not during implementation."""
import itertools
import unittest
import numpy as np
from psib.local_segmentation import (configurations,validate_settings,prepare_maps,build_candidates,
    refine_candidates,detect_local,context_indices,context_from_volume,binary_graph_cut,
    local_contour,physical_disk)
from psib.detection import configured_prediction


class LocalSegmentationTests(unittest.TestCase):
    def phantom(self):
        hu=np.full((35,43),100.)
        x,y=np.ogrid[:35,:43]
        hu[(x-10)**2+(y-10)**2<16]=45
        hu[(x-23)**2+(y-29)**2<16]=155
        liver=np.ones(hu.shape,bool);liver[:2]=False;liver[-2:]=False
        return hu,liver,(.7,1.1,2.)

    def test_frozen_grid(self):
        grid=configurations()
        self.assertEqual(len(grid),108)
        self.assertEqual(len({str(sorted(r['settings'].items())) for r in grid}),108)
        for record in grid:self.assertEqual(record['settings'],validate_settings(record['settings']))

    def test_dark_bright_seeds_and_determinism(self):
        hu,liver,spacing=self.phantom()
        cfg=validate_settings(dict(polarity='both',high=1.5))
        maps=prepare_maps(hu,liver,spacing,cfg)
        first=build_candidates(maps,cfg);second=build_candidates(maps,cfg)
        self.assertEqual({c.polarity for c in first['candidates']},{'dark','bright'})
        self.assertEqual([c.origins for c in first['candidates']],[c.origins for c in second['candidates']])
        for method,refinement in [('local_growth',2.),('local_contour',.3),('local_graphcut',.3)]:
            result=refine_candidates(maps,first,{**cfg,'method':method,'refinement':refinement})
            self.assertFalse((result['mask'] & ~liver).any())
            self.assertEqual(len(result['components']),len(first['candidates']))
            self.assertTrue(np.isfinite(result['mask']).all())

    def test_constant_negative_empty_and_anisotropic(self):
        for liver in (np.zeros((15,17),bool),np.ones((15,17),bool)):
            for method in ('local_growth','local_contour','local_graphcut'):
                result=detect_local(np.full(liver.shape,100.),liver,(.5,1.2),dict(method=method,refinement=.3))
                self.assertFalse(result['mask'].any());self.assertEqual(result['failed_candidates'],0)
        disk=physical_disk(2.,(.5,1.))
        self.assertEqual(disk.shape,(9,5))

    def test_context_boundaries_and_required_argument(self):
        hu,liver,spacing=self.phantom()
        self.assertEqual(context_indices(0,5,2.),[1,2])
        self.assertEqual(context_indices(2,5,6.),[])
        with self.assertRaisesRegex(ValueError,'vecinos'):
            detect_local(hu,liver,spacing,dict(context=True))
        first=detect_local(hu,liver,spacing,{})
        second=detect_local(hu,liver,spacing,dict(context=True),context=[])
        np.testing.assert_array_equal(first['mask'],second['mask'])
        with self.assertRaises(ValueError):
            detect_local(hu,liver,spacing,dict(context=True),context=[(6.,hu,liver)])

    def test_reference_independence_and_shared_dispatch(self):
        hu,liver,spacing=self.phantom()
        volume=np.repeat(hu[:,:,None],3,axis=2)
        labels=np.repeat(liver[:,:,None].astype(np.uint8),3,axis=2)
        changed=labels.copy();changed[10:12,10:12,:]=2
        cfg={'detection':dict(enabled=True,method='local_growth',context=True,polarity='both')}
        results=[]
        for seg in (labels,changed):
            context=context_from_volume(volume,seg>0,1,spacing)
            result=configured_prediction(hu,seg[:,:,1]>0,spacing,cfg,context=context)
            results.append(result)
        np.testing.assert_array_equal(results[0]['seeds'],results[1]['seeds'])
        np.testing.assert_array_equal(results[0]['mask'],results[1]['mask'])
        direct=detect_local(hu,liver,spacing,{k:v for k,v in cfg['detection'].items() if k!='enabled'},
                            context=context_from_volume(volume,labels>0,1,spacing))
        np.testing.assert_array_equal(results[0]['mask'],direct['mask'])

    def test_small_peripheral_neighboring_and_heterogeneous(self):
        hu,liver,spacing=self.phantom()
        hu[2:6,3:7]=50.;hu[9:12,16:19]=60.;hu[10:12,10:12]=75.
        for method in ('local_growth','local_contour','local_graphcut'):
            result=detect_local(hu,liver,spacing,dict(method=method,refinement=.3 if method!='local_growth' else 2.,polarity='both',high=1.5))
            self.assertFalse((result['mask'] & ~liver).any())
            self.assertTrue(result['seeds'].any())
            self.assertEqual(len(result['candidate_masks']),len(result['components']))

    def test_cut_matches_enumeration(self):
        cf=np.array([[0.,2.],[1.,3.]])
        cb=np.array([[4.,1.],[2.,0.]])
        a=np.array([0,0,1,2]);b=np.array([1,2,3,3]);w=np.array([.5,1.,.5,1.])
        fg=np.array([[True,False],[False,False]]);bg=np.array([[False,False],[False,True]])
        valid=np.ones((2,2),bool)
        mask,diagnostics=binary_graph_cut(cf,cb,(a,b,w),fg,bg,valid)
        def energy(m):return float(np.where(m,cf,cb).sum()+np.sum(w*(m.ravel()[a]!=m.ravel()[b])))
        possible=[]
        for bits in itertools.product((False,True),repeat=2):
            m=np.array([True,*bits,False]).reshape(2,2);possible.append(energy(m))
        self.assertAlmostEqual(energy(mask),min(possible))
        self.assertTrue(mask[fg].all());self.assertFalse(mask[bg].any())
        self.assertLess(diagnostics['hard_capacity'],2**31)
        with self.assertRaises(ValueError):binary_graph_cut(cf,cb,(a,b,w),fg,fg,valid)

    def test_cut_large_costs_and_one_pixel_axis(self):
        cf=np.array([[0.,1e12,1e12]]);cb=np.array([[1e12,1e12,0.]])
        fg=np.array([[True,False,False]]);bg=np.array([[False,False,True]])
        mask,d=binary_graph_cut(cf,cb,([0,1],[1,2],[.2,.2]),fg,bg,np.ones_like(fg))
        self.assertTrue(mask[0,0]);self.assertFalse(mask[0,2]);self.assertGreater(d['capacity_scale'],0)

    def test_local_contour_diagnostics_and_missing_background(self):
        hu,liver,spacing=self.phantom()
        support=hu<80;seeds=np.zeros_like(liver);seeds[10,10]=True
        cfg=validate_settings(dict(method='local_contour',refinement=.3))
        mask,diagnostics=local_contour(hu,liver,seeds,support,spacing[:2],cfg)
        self.assertLessEqual(diagnostics['iterations'],150)
        self.assertEqual(len(diagnostics['fitting_energy']),diagnostics['iterations'])
        self.assertTrue(np.isfinite(diagnostics['fitting_energy']).all())
        self.assertTrue(mask[10,10])
        with self.assertRaisesRegex(ValueError,'background'):
            local_contour(hu,liver,seeds,liver,spacing[:2],cfg)

    def test_invalid_configuration(self):
        for values in (dict(high=.1),dict(context='yes'),dict(refinement=-1),dict(background_mm=[1,2]),dict(max_iterations=0)):
            with self.assertRaises(ValueError):validate_settings(values)


if __name__=='__main__':unittest.main()
