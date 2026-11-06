import json
import tempfile
import unittest
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psib.morphology3d import remove_small_components_3d
from scripts.evaluate_morphology_3d import evaluate


class Morphology3DPredictionTests(unittest.TestCase):
    def test_filter_removes_small_6_connected_components(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 0, 0] = True
        mask[2:4, 2:4, 2:4] = True

        filtered = remove_small_components_3d(mask, min_voxels=8)

        self.assertEqual(int(filtered.sum()), 8)
        self.assertFalse(filtered[0, 0, 0])

    def test_filter_requires_a_positive_integer_and_binary_volume(self):
        with self.assertRaises(ValueError):
            remove_small_components_3d(np.zeros((2, 2, 2), dtype=bool), 0)
        with self.assertRaises(ValueError):
            remove_small_components_3d(np.full((2, 2, 2), 2), 1)

    def test_end_to_end_stacks_every_axial_slice_and_compares_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            labels = np.zeros((8, 8, 3), dtype=np.uint8)
            labels[1:7, 1:7, :] = 1
            labels[3:5, 3:5, 1:3] = 2
            ct = np.zeros(labels.shape, dtype=np.float32)
            ct[labels == 1] = 20.0
            ct[labels == 2] = -50.0
            affine = np.diag((1.0, 1.0, 2.0, 1.0))
            nib.save(nib.Nifti1Image(ct, affine), data / "volume-1.nii.gz")
            nib.save(nib.Nifti1Image(labels, affine), data / "segmentation-1.nii.gz")

            run = root / "results" / "segmentation" / "tiny"
            run.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (run / "effective_config.yaml").write_text(yaml.safe_dump({
                "data_dir": str(data), "output_dir": str(root / "results"),
                "preprocessing": {"hu_window": {"center": 0, "width": 100},
                                  "denoising": {"method": "median", "median_size": 1}},
            }), encoding="utf-8")
            pd.DataFrame([{
                "configuration_id": "cfg_000", "method": "global", "threshold": 0.4,
                "k": 1.0, "direction": "lower", "opening": False, "closing": False,
                "fill_holes": False, "footprint": "disk", "radius": 1, "min_area": 0,
            }]).to_csv(run / "configurations.csv", index=False)
            pd.DataFrame([{"configuration_id": "cfg_000", "case_id": 1, "dice": 1.0,
                           "iou": 1.0}]).to_csv(run / "metrics_per_case.csv", index=False)
            pd.DataFrame([{"case_id": 1, "slice_index": 1}]).to_csv(
                run / "selected_cases_slices.csv", index=False)

            output = evaluate(run, data, min_component_voxels=8)

            result = pd.read_csv(output / "morphology_3d_prediction_per_case.csv").iloc[0]
            self.assertEqual(result.n_slices, 3)
            self.assertEqual(result.reference_tumor_voxels, 8)
            self.assertEqual(result.prediction_clean_voxels, 8)
            self.assertAlmostEqual(result.dice_3d_clean, 1.0)
            self.assertAlmostEqual(result.clean_volume_relative_error, 0.0)
            self.assertTrue((output / "comparison_3d.png").is_file())
            self.assertTrue((output / "case_001_masks_3d.npz").is_file())
            self.assertEqual(result.reference_lesions, 1)
            self.assertEqual(result.detected_lesions, 1)
            self.assertEqual(result.false_positive_components, 0)
            self.assertEqual(result.liver_roi, "reference")
            lesions = pd.read_csv(output / "morphology_3d_prediction_lesions.csv")
            self.assertEqual(len(lesions), 1)
            self.assertAlmostEqual(lesions.lesion_dice.iloc[0], 1.0)
            by_size = pd.read_csv(output / "morphology_3d_prediction_by_size.csv").set_index("size_group")
            self.assertAlmostEqual(by_size.loc["all", "detection_rate"], 1.0)

            subset = evaluate(run, data, min_component_voxels=8, output_name="subset",
                              case_ids=[1], save_masks=False)
            self.assertFalse((subset / "case_001_masks_3d.npz").exists())
            with self.assertRaises(ValueError):
                evaluate(run, data, output_name="missing", case_ids=[99])


if __name__ == "__main__":
    unittest.main()
