import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psib.lesion_eval import classify_diameter, match_lesions


class LesionEvaluationTests(unittest.TestCase):
    def test_perfect_match(self):
        ref = np.zeros((20, 20), dtype=bool)
        ref[3:8, 4:10] = True
        result = match_lesions(ref, ref, (0.8, 1.2))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].status, "TP")
        self.assertEqual(result.iloc[0].group_dice, 1.0)
        self.assertAlmostEqual(result.iloc[0].area_ref_mm2, 28.8)

    def test_prediction_split_is_grouped_with_reference_lesion(self):
        ref = np.zeros((15, 20), dtype=bool)
        pred = np.zeros_like(ref)
        ref[4:9, 3:15] = True
        pred[4:9, 3:8] = True
        pred[4:9, 10:15] = True
        result = match_lesions(pred, ref, (1, 1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].status, "TP")
        self.assertEqual(result.iloc[0].predicted_components, "1;2")

    def test_prediction_merge_is_grouped_with_reference_lesions(self):
        ref = np.zeros((15, 20), dtype=bool)
        pred = np.zeros_like(ref)
        ref[4:8, 3:7] = True
        ref[4:8, 10:14] = True
        pred[4:8, 3:14] = True
        result = match_lesions(pred, ref, (1, 1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].status, "TP")
        self.assertEqual(result.iloc[0].reference_components, "1;2")

    def test_missed_lesion_and_false_positive(self):
        ref = np.zeros((20, 20), dtype=bool)
        pred = np.zeros_like(ref)
        ref[2:5, 2:5] = True
        pred[12:16, 12:16] = True
        result = match_lesions(pred, ref, (1, 1))
        self.assertCountEqual(result.status, ["FN", "FP"])

    def test_minimum_iou_and_size_classes_with_anisotropic_spacing(self):
        ref = np.zeros((30, 30), dtype=bool)
        pred = np.zeros_like(ref)
        ref[5:8, 5:8] = True
        pred[5:8, 7:10] = True
        self.assertEqual(match_lesions(pred, ref, (2, 3), min_iou=0.5).status.tolist(), ["FN", "FP"])
        self.assertEqual(classify_diameter(9.99), "<10")
        self.assertEqual(classify_diameter(10.0), "10-20")
        self.assertEqual(classify_diameter(20.0), ">=20")


if __name__ == "__main__":
    unittest.main()
