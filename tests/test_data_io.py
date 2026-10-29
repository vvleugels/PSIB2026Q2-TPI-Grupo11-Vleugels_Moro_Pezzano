import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from psib.data_io import get_slice, list_cases, load_case, load_segmentation


class DataIOTests(unittest.TestCase):
    def test_pair_discovery_canonical_orientation_and_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.diag([-2.0, 3.0, 4.0, 1.0])
            affine[0, 3] = 2.0
            volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            seg = np.zeros_like(volume, dtype=np.uint8)
            seg[0, 1, 2] = 2
            nib.save(nib.Nifti1Image(volume, affine), root / "volume-7.nii.gz")
            nib.save(nib.Nifti1Image(seg, affine), root / "segmentation-7.nii.gz")
            self.assertEqual(list_cases(root), [7])
            ct, mask, spacing = load_case(7, root)
            np.testing.assert_array_equal(ct, volume[::-1, :, :])
            np.testing.assert_array_equal(mask, seg[::-1, :, :])
            np.testing.assert_array_equal(load_segmentation(7, root), mask)
            self.assertEqual(spacing, (2.0, 3.0, 4.0))
            np.testing.assert_array_equal(get_slice(ct, 2), ct[:, :, 2])

    def test_missing_pair_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "paired CT volume"):
                load_case(1, tmp)


if __name__ == "__main__":
    unittest.main()
