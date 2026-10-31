import unittest

import numpy as np

from psib.features import extract_geometric_features, extract_texture_features


class FeatureTests(unittest.TestCase):
    def test_geometry_reports_one_row_per_eight_connected_component(self):
        mask = np.zeros((8, 9), dtype=bool)
        mask[1, 1] = mask[2, 2] = True  # Diagonal pixels are connected.
        mask[5:7, 6:8] = True

        features = extract_geometric_features(mask, (2.0, 3.0))

        self.assertEqual(len(features), 2)
        self.assertEqual(features.area_pixels.tolist(), [2, 4])
        self.assertEqual(features.area_mm2.tolist(), [12.0, 24.0])
        self.assertTrue((features.max_diameter_mm > 0).all())
        self.assertTrue(features.eccentricity.between(0, 1, inclusive="both").all())
        self.assertTrue(features.solidity.between(0, 1, inclusive="right").all())

    def test_geometry_empty_mask_keeps_stable_columns(self):
        features = extract_geometric_features(np.zeros((4, 5), dtype=bool), (0.8, 1.2))
        self.assertEqual(len(features), 0)
        self.assertEqual(features.columns.tolist(), [
            "component_id", "area_pixels", "area_mm2", "eccentricity", "solidity",
            "max_diameter_mm", "centroid_x_mm", "centroid_y_mm",
        ])

    def test_geometry_validates_mask_and_spacing(self):
        valid = np.zeros((3, 3), dtype=bool)
        for mask, spacing in ((np.zeros((3, 3, 1)), (1.0, 1.0)),
                              (np.full((3, 3), 2), (1.0, 1.0)),
                              (valid, (1.0,)), (valid, (1.0, 0.0)),
                              (valid, (1.0, np.nan))):
            with self.subTest(mask_shape=np.shape(mask), spacing=spacing), self.assertRaises(ValueError):
                extract_geometric_features(mask, spacing)

    def test_texture_constant_region_has_zero_contrast_and_one_energy(self):
        image = np.full((3, 3), 0.6, dtype=float)
        mask = np.ones((3, 3), dtype=bool)

        features = extract_texture_features(image, mask, levels=8)

        self.assertEqual(features["contrast"], 0.0)
        self.assertEqual(features["homogeneity"], 1.0)
        self.assertEqual(features["energy"], 1.0)
        self.assertEqual(features["correlation"], 1.0)
        self.assertEqual(features["pair_count"], 20)
        self.assertEqual(features["n_offsets"], 4)

    def test_texture_counts_only_pairs_inside_roi(self):
        image = np.array([[0.0, 1.0, 0.5]])
        mask = np.array([[1, 1, 0]], dtype=bool)

        features = extract_texture_features(image, mask, levels=2)

        self.assertEqual(features["pair_count"], 1)
        self.assertEqual(features["n_offsets"], 1)
        self.assertEqual(features["contrast"], 1.0)
        self.assertEqual(features["homogeneity"], 0.5)
        self.assertEqual(features["energy"], 2 ** -0.5)
        self.assertEqual(features["correlation"], -1.0)

    def test_texture_without_neighbor_pairs_returns_nan_properties(self):
        features = extract_texture_features(np.array([[0.4]]), np.array([[1]]))
        for name in ("contrast", "homogeneity", "energy", "correlation"):
            self.assertTrue(np.isnan(features[name]))
        self.assertEqual(features["pair_count"], 0)
        self.assertEqual(features["n_offsets"], 0)

    def test_texture_validates_image_mask_and_parameters(self):
        image = np.zeros((3, 3), dtype=float)
        mask = np.ones((3, 3), dtype=bool)
        invalid = (
            (np.zeros((3, 3, 1)), mask, {}),
            (image, np.ones((2, 2)), {}),
            (np.full((3, 3), np.nan), mask, {}),
            (np.full((3, 3), 1.1), mask, {}),
            (image, np.full((3, 3), 2), {}),
            (image, mask, {"levels": 1}),
            (image, mask, {"levels": 257}),
            (image, mask, {"distances": (0,)}),
            (image, mask, {"angles": ()}),
        )
        for bad_image, bad_mask, options in invalid:
            with self.subTest(options=options, image_shape=np.shape(bad_image)), self.assertRaises(ValueError):
                extract_texture_features(bad_image, bad_mask, **options)


if __name__ == "__main__":
    unittest.main()
