import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from psib.dataset import build_inventory, select_cases, select_slices


class DatasetTests(unittest.TestCase):
    def test_inventory_samples_only_seeded_paired_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
            for case_id in range(6):
                ct = np.full((4, 4, 4), case_id, dtype=np.int16)
                seg = np.zeros_like(ct, dtype=np.uint8)
                seg[1, 1, 1:1 + (case_id % 3 + 1)] = 2
                nib.save(nib.Nifti1Image(ct, affine), root / f"volume-{case_id}.nii.gz")
                nib.save(nib.Nifti1Image(seg, affine), root / f"segmentation-{case_id}.nii.gz")
            pool_a = build_inventory(root, candidate_pool_size=3, seed=44)
            pool_b = build_inventory(root, candidate_pool_size=3, seed=44)
            self.assertEqual(pool_a.case_id.tolist(), pool_b.case_id.tolist())
            self.assertEqual(len(pool_a), 3)
            self.assertEqual(pool_a.candidate_pool_rank.tolist(), [1, 2, 3])

    def test_inventory_volume_and_3d_lesions_use_real_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.diag([2.0, 3.0, 4.0, 1.0])
            ct = np.full((6, 7, 8), 50, dtype=np.int16)
            seg = np.zeros_like(ct, dtype=np.uint8)
            seg[1:4, 1:4, 1:3] = 1
            seg[1:3, 1:3, 2:4] = 2
            seg[5, 6, 6] = 2
            nib.save(nib.Nifti1Image(ct, affine), root / "volume-3.nii.gz")
            nib.save(nib.Nifti1Image(seg, affine), root / "segmentation-3.nii.gz")
            row = build_inventory(root).iloc[0]
            self.assertEqual(row.tumor_voxels, 9)
            self.assertAlmostEqual(row.tumor_volume_cm3, 9 * 2 * 3 * 4 / 1000)
            self.assertEqual(row.tumor_lesions_3d, 2)

    def test_slice_selection_respects_minimum_gap(self):
        seg = np.zeros((10, 10, 20), dtype=np.uint8)
        seg[1:8, 1:8, 2] = 2
        seg[1:7, 1:7, 4] = 2
        seg[1:6, 1:6, 9] = 2
        seg[1:5, 1:5, 16] = 2
        chosen = select_slices(seg, 3, minimum_gap=4)
        self.assertEqual(chosen, [2, 9, 16])
        self.assertTrue(all(b - a >= 4 for a, b in zip(chosen, chosen[1:])))

    def test_case_selection_is_reproducible_and_prefers_multiple_lesions(self):
        frame = pd.DataFrame({"case_id": range(12), "tumor_volume_cm3": np.arange(12),
                              "tumor_lesions_3d": [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        self.assertEqual(select_cases(frame, 6, 4), select_cases(frame, 6, 4))
        self.assertEqual(len(select_cases(frame, 6, 4)), 6)
        self.assertGreaterEqual(sum(frame.set_index("case_id").loc[i].tumor_lesions_3d > 1
                                    for i in select_cases(frame, 6, 4)), 2)

    def test_case_selection_reports_missing_terciles(self):
        frame = pd.DataFrame({"case_id": range(5), "tumor_volume_cm3": [1] * 5,
                              "tumor_lesions_3d": [2, 1, 1, 1, 1]})
        with self.assertRaisesRegex(ValueError, "does not cover all three tumor-volume terciles"):
            select_cases(frame, 3, 2026)


if __name__ == "__main__":
    unittest.main()
