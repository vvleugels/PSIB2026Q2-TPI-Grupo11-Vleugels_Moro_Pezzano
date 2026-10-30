import unittest

import numpy as np

from psib.metrics import dice_score, iou_score


class MetricsTests(unittest.TestCase):
    metrics = (dice_score, iou_score)

    def test_synthetic_scores(self):
        pred = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
        overlap = np.array([[0, 1, 1], [0, 1, 1]], dtype=bool)
        empty = np.zeros_like(pred)
        scenarios = (
            (pred, pred, 1.0, 1.0),
            (pred, ~pred, 0.0, 0.0),
            (empty, empty, 1.0, 1.0),
            (pred, empty, 0.0, 0.0),
            (empty, pred, 0.0, 0.0),
            (pred, overlap, 0.5, 1 / 3),
        )
        for index, (prediction, reference, dice, iou) in enumerate(scenarios):
            for metric, expected in zip(self.metrics, (dice, iou)):
                with self.subTest(scenario=index, metric=metric.__name__):
                    actual = metric(prediction, reference)
                    self.assertAlmostEqual(actual, expected)
                    self.assertIsInstance(actual, float)
                    self.assertTrue(np.isfinite(actual))
                    self.assertGreaterEqual(actual, 0)
                    self.assertLessEqual(actual, 1)
                    self.assertEqual(actual, metric(reference, prediction))

    def test_numeric_binary_types_and_inputs_are_preserved(self):
        for dtype in (bool, np.uint8, np.int64, np.float32, np.float64, np.complex128):
            pred = np.array([[1, 0], [1, 0]], dtype=dtype)
            reference = np.array([[1, 1], [0, 0]], dtype=dtype)
            pred_before, ref_before = pred.copy(), reference.copy()
            pred.flags.writeable = reference.flags.writeable = False
            for metric in self.metrics:
                with self.subTest(dtype=dtype, metric=metric.__name__):
                    self.assertEqual(metric(pred, reference), metric(pred.astype(bool), reference.astype(bool)))
                    np.testing.assert_array_equal(pred, pred_before)
                    np.testing.assert_array_equal(reference, ref_before)

    def test_3d_masks_count_all_voxels(self):
        pred = np.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]])
        reference = np.array([[[1, 0], [1, 0]], [[1, 0], [1, 0]]])
        self.assertEqual(dice_score(pred, reference), 0.5)
        self.assertAlmostEqual(iou_score(pred, reference), 1 / 3)

    def test_shape_alert_includes_both_shapes(self):
        for pred_shape, ref_shape in (((4, 4), (2, 2)), ((2, 3), (3, 2)), ((2, 2), (2, 2, 1))):
            for metric in self.metrics:
                with self.subTest(shapes=(pred_shape, ref_shape), metric=metric.__name__):
                    with self.assertRaises(ValueError) as raised:
                        metric(np.zeros(pred_shape), np.zeros(ref_shape))
                    message = str(raised.exception)
                    self.assertIn("misma forma", message)
                    self.assertIn(f"predicción {pred_shape}", message)
                    self.assertIn(f"referencia {ref_shape}", message)

    def test_invalid_values_in_either_mask(self):
        for value in (2, -1, 0.5, np.nan, np.inf, -np.inf, 1 + 1j, "1", None):
            invalid = np.array([[value]])
            valid = np.zeros((1, 1), dtype=bool)
            for metric in self.metrics:
                for pred, reference in ((invalid, valid), (valid, invalid)):
                    with self.subTest(value=value, metric=metric.__name__, pred_invalid=pred is invalid):
                        with self.assertRaises(ValueError):
                            metric(pred, reference)

    def test_invalid_dimensions(self):
        for shape in ((), (4,), (1, 1, 1, 1)):
            for metric in self.metrics:
                with self.subTest(shape=shape, metric=metric.__name__):
                    with self.assertRaises(ValueError):
                        metric(np.zeros(shape), np.zeros(shape))


if __name__ == "__main__":
    unittest.main()
