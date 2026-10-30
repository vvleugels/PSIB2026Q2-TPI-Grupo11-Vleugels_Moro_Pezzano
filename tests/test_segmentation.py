import unittest

import numpy as np

from psib.segmentation import segment_slice


class SegmentationTests(unittest.TestCase):
    def segment_mask(self, mask, settings, liver=None):
        if liver is None:
            liver = np.ones_like(mask, dtype=bool)
        return segment_slice(np.where(mask, 0.1, 0.9), liver, settings)

    def test_global_threshold_and_liver_exclusion(self):
        image = np.array([[0.1, 0.45, 0.9], [0.2, 0.3, 0.1]])
        liver = np.array([[1, 1, 1], [1, 0, 0]])
        expected = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(segment_slice(image, liver, {}), expected)

    def test_otsu_uses_only_liver(self):
        liver = np.zeros((8, 8), dtype=bool)
        liver[2:6, 2:6] = True
        image = np.full((8, 8), 0.5)
        image[2:4, 2:6] = 0.2
        image[4:6, 2:6] = 0.8
        expected = np.zeros_like(liver)
        expected[2:4, 2:6] = True
        for outside in (0.0, 1.0):
            image[~liver] = outside
            np.testing.assert_array_equal(segment_slice(image, liver, {"method": "otsu"}), expected)

    def test_statistical_population_std_and_directions(self):
        # Liver mean=0.5 and population std=sqrt(0.08); k=1 selects endpoints.
        image = np.array([[0.1, 0.3, 0.5], [0.7, 0.9, 0.0]])
        liver = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
        for direction, coordinate in (("lower", (0, 0)), ("upper", (1, 1))):
            expected = np.zeros_like(liver)
            expected[coordinate] = True
            for outside in (0.0, 1.0):
                image[~liver] = outside
                np.testing.assert_array_equal(segment_slice(image, liver, {
                    "method": "statistical", "direction": direction, "k": 1.0,
                }), expected)
        # k=0 puts the threshold exactly on 0.5, which neither direction selects.
        for direction in ("lower", "upper"):
            result = segment_slice(image, liver, {"method": "statistical", "k": 0, "direction": direction})
            self.assertFalse(result[0, 2])
            self.assertEqual(np.count_nonzero(result), 2)

    def test_empty_liver_and_constant_intensity(self):
        for value in (0.0, 0.3, 1.0):
            image = np.full((4, 4), value)
            for method in ("global", "otsu", "statistical"):
                self.assertFalse(segment_slice(image, np.zeros_like(image), {"method": method}).any())
            for method in ("otsu", "statistical"):
                for direction in ("lower", "upper"):
                    self.assertFalse(segment_slice(image, np.ones_like(image), {
                        "method": method, "direction": direction,
                    }).any())

    def test_opening_removes_noise_and_footprints_differ(self):
        mask = np.zeros((13, 13), dtype=bool)
        mask[3:10, 3:10] = True
        mask[1, 1] = True
        square = self.segment_mask(mask, {"opening": True, "footprint": "square"})
        expected = mask.copy()
        expected[1, 1] = False
        np.testing.assert_array_equal(square, expected)
        circle = self.segment_mask(mask, {"opening": True, "footprint": "disk"})
        self.assertFalse(circle[1, 1])
        self.assertFalse(circle[3, 3])
        self.assertTrue(circle[6, 6])
        # A larger footprint removes a small object that a 3x3 square preserves.
        small = np.zeros((9, 9), dtype=bool)
        small[3:6, 3:6] = True
        self.assertTrue(self.segment_mask(small, {"opening": True, "footprint": "square"}).any())
        self.assertFalse(self.segment_mask(small, {"opening": True, "footprint": "square", "radius": 2}).any())

    def test_closing_bridges_small_gap(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[3:8, 2:5] = True
        mask[3:8, 6:9] = True
        expected = np.zeros_like(mask)
        expected[3:8, 2:9] = True
        np.testing.assert_array_equal(self.segment_mask(mask, {"closing": True, "footprint": "square"}), expected)

    def test_hole_filling_uses_eight_neighbors(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:7, 2:7] = True
        mask[4, 4] = False
        expected = mask.copy()
        expected[4, 4] = True
        np.testing.assert_array_equal(self.segment_mask(mask, {"fill_holes": True}), expected)
        # A diagonal path to the background is not a hole with 8 neighbors.
        mask[2, 2] = mask[3, 3] = False
        np.testing.assert_array_equal(self.segment_mask(mask, {"fill_holes": True}), mask)

    def test_area_filter_boundary_and_diagonal_connectivity(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1, 1:3] = True  # area 2: removed
        mask[4, 4] = mask[5, 5] = mask[6, 6] = True  # area 3: kept
        mask[9:11, 9:11] = True  # area 4: kept
        expected = mask.copy()
        expected[1, 1:3] = False
        np.testing.assert_array_equal(self.segment_mask(mask, {"min_area": 3}), expected)
        np.testing.assert_array_equal(self.segment_mask(mask, {"min_area": 0}), mask)

    def test_operations_stay_inside_liver(self):
        liver = np.zeros((13, 13), dtype=bool)
        liver[2:11, 2:11] = True
        liver[6, 6] = False
        mask = liver.copy()
        mask[2:7, 6] = False
        for settings in ({"opening": True}, {"closing": True}, {"fill_holes": True},
                         {"opening": True, "closing": True, "fill_holes": True, "min_area": 2}):
            result = self.segment_mask(mask, settings, liver)
            self.assertFalse(result[~liver].any())

    def test_integrated_pipeline_and_input_preservation(self):
        mask = np.zeros((17, 17), dtype=bool)
        mask[4:13, 4:13] = True
        mask[8, 8] = False
        mask[1, 1] = True
        image = np.where(mask, 0.1, 0.9)
        liver = np.ones_like(mask, dtype=np.uint8)
        settings = {"opening": True, "closing": True, "fill_holes": True,
                    "footprint": "square", "min_area": 20}
        image_before, liver_before, settings_before = image.copy(), liver.copy(), settings.copy()
        image.flags.writeable = liver.flags.writeable = False
        result = segment_slice(image, liver, settings)
        expected = np.zeros_like(mask)
        expected[4:13, 4:13] = True
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.bool_)
        self.assertEqual(result.shape, image.shape)
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(liver, liver_before)
        self.assertEqual(settings, settings_before)

    def test_shape_alert_and_dimensions(self):
        for image_shape, liver_shape in (((2, 3), (3, 2)), ((2, 2), (3, 3))):
            with self.assertRaises(ValueError) as raised:
                segment_slice(np.zeros(image_shape), np.zeros(liver_shape), {})
            self.assertIn(str(image_shape), str(raised.exception))
            self.assertIn(str(liver_shape), str(raised.exception))
        for shape in ((), (4,), (2, 2, 2)):
            with self.assertRaises(ValueError):
                segment_slice(np.zeros(shape), np.zeros(shape), {})

    def test_invalid_image_and_mask_values(self):
        valid = np.ones((2, 2))
        for value in (-0.1, 1.1, np.nan, np.inf, -np.inf, 0.5 + 0j, "0.5", None):
            with self.subTest(image_value=value), self.assertRaises(ValueError):
                segment_slice(np.full((2, 2), value), valid, {})
        for value in (2, -1, 0.5, np.nan, np.inf, "1", None):
            with self.subTest(mask_value=value), self.assertRaises(ValueError):
                segment_slice(valid, np.full((2, 2), value), {})

    def test_invalid_settings_are_checked_even_for_empty_liver(self):
        invalid = (
            {"method": "unknown"}, {"direction": "unknown"}, {"footprint": "unknown"},
            {"threshold": -1}, {"threshold": 1.1}, {"threshold": np.nan},
            {"threshold": "0.4"}, {"threshold": True}, {"k": -1}, {"k": np.inf},
            {"radius": 0}, {"radius": 1.5}, {"radius": True},
            {"min_area": -1}, {"min_area": 1.5}, {"min_area": True},
            {"opening": "false"}, {"closing": 1}, {"fill_holes": None},
            {"typo": True}, None,
        )
        image = np.zeros((3, 3))
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                segment_slice(image, np.zeros_like(image), settings)
        # Image validation also precedes the empty-liver shortcut.
        with self.assertRaises(ValueError):
            segment_slice(np.full((3, 3), np.nan), np.zeros_like(image), {})


if __name__ == "__main__":
    unittest.main()
