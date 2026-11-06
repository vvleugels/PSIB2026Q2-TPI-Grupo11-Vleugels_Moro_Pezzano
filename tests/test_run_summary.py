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
from psib.run_summary import MEAN_LABEL, case_table, compute_lesion_sizes_2d, list_runs


def _completed(path: Path, **extra) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(json.dumps({"status": "completed", **extra}), encoding="utf-8")


class RunSummaryTests(unittest.TestCase):
    def test_3d_table_groups_medium_and_large_lesions_and_appends_mean(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "segmentation" / "run1" / "morphology3d_prediction_x"
            _completed(output.parent)
            pd.DataFrame({"configuration_id": ["cfg_000"], "case_id": [1]}).to_csv(
                output.parent / "metrics_per_case.csv", index=False)
            _completed(output, n_cases=2, min_component_mm3=50, min_slices=2)
            pd.DataFrame({"case_id": [1, 2], "dice_3d_clean": [0.8, 0.4],
                          "reference_volume_cm3": [10.0, 4.0],
                          "prediction_clean_volume_cm3": [10.0, 8.0]}).to_csv(
                output / "morphology_3d_prediction_per_case.csv", index=False)
            pd.DataFrame({"case_id": [1, 1, 2, 2, 2],
                          "max_axial_feret_mm": [12.0, 25.0, 5.0, 15.0, 30.0],
                          "coverage": [0.9, 0.6, 1.0, 0.2, 0.7],
                          "lesion_dice": [0.8, 0.5, 0.9, 0.1, 0.6]}).to_csv(
                output / "morphology_3d_prediction_lesions.csv", index=False)

            table, info = case_table(output)

            self.assertEqual(info["kind"], "3D")
            self.assertEqual(list(table.case_id), [1, 2, MEAN_LABEL])
            mean = table.iloc[-1]
            self.assertEqual((mean.medium_covered, mean.medium_total), (1, 2))  # 12 mm yes, 15 mm no
            self.assertEqual((mean.large_covered, mean.large_total), (2, 2))
            self.assertAlmostEqual(mean.medium_dice, 0.45)
            self.assertAlmostEqual(mean.case_dice, 0.6)
            self.assertAlmostEqual(table.precision.iloc[1], 0.4 * 12 / 2 / 8)
            labels = [run["label"] for run in list_runs(Path(temporary))]
            self.assertTrue(any("filtro ≥50 mm³, ≥2 cortes" in label for label in labels))
            self.assertTrue(any(label.startswith("2D · run1") for label in labels))

    def test_2d_table_without_sizes_has_empty_groups_and_computes_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            labels = np.zeros((40, 40, 2), dtype=np.uint8)
            labels[2:38, 2:38, :] = 1
            labels[5:20, 5:20, 0] = 2   # 14 px -> 19.8 mm diagonal at 1 mm: medium
            labels[25:27, 25:27, 0] = 2  # tiny lesion, excluded from both groups
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(np.zeros(labels.shape, np.float32), affine), data / "volume-1.nii.gz")
            nib.save(nib.Nifti1Image(labels, affine), data / "segmentation-1.nii.gz")
            run = root / "segmentation" / "run2"
            _completed(run)
            (run / "effective_config.yaml").write_text(yaml.safe_dump({"data_dir": str(data)}), encoding="utf-8")
            pd.DataFrame({"configuration_id": ["cfg_000"], "case_id": [1], "dice": [0.5],
                          "pred_area": [100], "intersection": [50]}).to_csv(run / "metrics_per_case.csv", index=False)
            pd.DataFrame({"configuration_id": ["cfg_000"], "case_id": [1], "slice_index": [0]}).to_csv(
                run / "metrics_per_slice.csv", index=False)
            prediction = np.zeros((40, 40), dtype=bool)
            prediction[5:20, 5:12] = True
            (run / "predictions" / "cfg_000").mkdir(parents=True)
            np.save(run / "predictions" / "cfg_000" / "case_001_slice_000.npy", prediction)

            table, info = case_table(run)
            self.assertFalse(info["has_sizes"])
            self.assertTrue(np.isnan(table.medium_total.iloc[-1]))

            compute_lesion_sizes_2d(run, progress=lambda _message: None)
            table, info = case_table(run)
            self.assertTrue(info["has_sizes"])
            self.assertEqual((table.medium_covered.iloc[0], table.medium_total.iloc[0]), (0, 1))
            self.assertAlmostEqual(table.medium_dice.iloc[0], 2 * 105 / (225 + 105))
            self.assertEqual(table.large_total.iloc[-1], 0)
            self.assertAlmostEqual(table.precision.iloc[0], 0.5)


if __name__ == "__main__":
    unittest.main()
