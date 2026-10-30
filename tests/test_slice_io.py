"""Small synthetic NIfTI fixtures for the bounded-memory 2D reader."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from psib.data_io import iter_case_slices
from scripts.select_slices_2d import select_2d


class SliceIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, ct, seg, affine):
        nib.save(nib.Nifti1Image(ct, affine), self.root / 'volume-1.nii.gz')
        nib.save(nib.Nifti1Image(seg, affine), self.root / 'segmentation-1.nii.gz')

    def test_flipped_axes_without_full_volume_loading(self):
        ct = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
        seg = (ct % 3).astype(np.uint8)
        self.save(ct, seg, np.diag([-2., 3., -4., 1.]))
        with patch.object(nib.Nifti1Image, 'get_fdata', side_effect=AssertionError('full volume read')):
            slices = list(iter_case_slices(1, [0, 3], self.root))
        for k, image, labels, spacing in slices:
            np.testing.assert_array_equal(image, ct[::-1, :, 4-k])
            np.testing.assert_array_equal(labels, seg[::-1, :, 4-k])
            self.assertEqual(spacing, (2., 3., 4.))

    def test_permuted_axes_and_mask_only(self):
        ct = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
        seg = (ct % 3).astype(np.uint8)
        affine = np.array([[0., 0., 2., 0.], [3., 0., 0., 0.], [0., 4., 0., 0.], [0., 0., 0., 1.]])
        self.save(ct, seg, affine)
        k, image, labels, spacing = next(iter_case_slices(1, [2], self.root))
        np.testing.assert_array_equal(image, ct[:, 2, :].T)
        np.testing.assert_array_equal(labels, seg[:, 2, :].T)
        self.assertEqual(spacing, (2., 3., 4.))
        _, image, labels, _ = next(iter_case_slices(1, [2], self.root, mask_only=True))
        self.assertIsNone(image)
        np.testing.assert_array_equal(labels, seg[:, 2, :].T)

    def test_range_and_label_validation(self):
        ct = np.zeros((3, 3, 3), dtype=np.int16)
        seg = ct.copy()
        seg[0, 0, 1] = 256
        self.save(ct, seg, np.eye(4))
        with self.assertRaises(IndexError):
            list(iter_case_slices(1, [3], self.root))
        with self.assertRaisesRegex(ValueError, 'slice 1'):
            list(iter_case_slices(1, [1], self.root))

    def test_selection_2d_matches_area_ranking(self):
        ct = np.zeros((9, 9, 6), dtype=np.int16)
        seg = np.ones_like(ct, dtype=np.uint8)
        seg[2:7, 2:7, 1] = 2
        seg[2:6, 2:6, 3] = 2
        seg[2:5, 2:5, 5] = 2
        self.save(ct, seg, np.eye(4))
        config = self.root / 'config.yaml'
        config.write_text('data_dir: .\noutput_dir: results\nselection:\n  n_cases: 1\n  slices_per_case: 3\n  minimum_slice_gap: 2\n')
        import pandas as pd
        selected = pd.read_csv(select_2d(config))
        self.assertEqual(selected.slice_index.tolist(), [1, 3, 5])
        self.assertNotIn('tumor_lesions_3d', selected.columns)


if __name__ == '__main__':
    unittest.main()
