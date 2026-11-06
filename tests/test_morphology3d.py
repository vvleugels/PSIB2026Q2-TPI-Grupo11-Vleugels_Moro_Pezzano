import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psib.morphology3d import (filter_components_3d, match_lesions_3d,
                               summarize_lesions_by_size, tumor_components_3d)


class Morphology3DTests(unittest.TestCase):
    def test_spherical_component_reports_volume_and_physical_diameters(self):
        grid = np.indices((21, 21, 21), dtype=float)
        sphere = sum((axis - 10) ** 2 for axis in grid) <= 25
        segmentation = np.zeros(sphere.shape, dtype=np.uint8)
        segmentation[sphere] = 2

        components, _ = tumor_components_3d(segmentation, (1.0, 1.0, 1.0))

        self.assertEqual(len(components), 1)
        item = components[0]
        self.assertAlmostEqual(item["max_feret_3d_mm"], 10.0, delta=1.5)
        self.assertAlmostEqual(item["max_axial_feret_mm"], 10.0, delta=1.5)
        self.assertAlmostEqual(item["equivalent_diameter_mm"],
                               (6 * item["voxel_count"] / np.pi) ** (1 / 3))
        self.assertGreaterEqual(item["bbox_diagonal_mm"], item["max_feret_3d_mm"])

    def test_elongated_ellipsoid_uses_anisotropic_spacing(self):
        x, y, z = np.indices((31, 15, 9), dtype=float)
        ellipsoid = ((x - 15) / 10) ** 2 + ((y - 7) / 4) ** 2 + ((z - 4) / 2) ** 2 <= 1
        segmentation = np.zeros(ellipsoid.shape, dtype=np.uint8)
        segmentation[ellipsoid] = 2

        components, _ = tumor_components_3d(segmentation, (0.8, 1.2, 3.0))

        item = components[0]
        self.assertAlmostEqual(item["max_feret_3d_mm"], 16.0, delta=1.2)
        self.assertAlmostEqual(item["max_axial_feret_mm"], 16.0, delta=1.2)
        self.assertAlmostEqual(item["bbox_diagonal_mm"], np.linalg.norm((
            item["extent_x_mm"], item["extent_y_mm"], item["extent_z_mm"])))

    def test_flat_component_falls_back_when_3d_hull_is_degenerate(self):
        segmentation = np.zeros((8, 8, 1), dtype=np.uint8)
        segmentation[1:6, 2:5, 0] = 2

        components, _ = tumor_components_3d(segmentation, (2.0, 1.0, 4.0))

        self.assertEqual(len(components), 1)
        self.assertGreater(components[0]["max_feret_3d_mm"], 0)
        self.assertEqual(components[0]["extent_z_mm"], 4.0)

    def test_empty_mask_has_no_components(self):
        components, render_data = tumor_components_3d(np.zeros((3, 4, 5), dtype=np.uint8), (1, 2, 3))
        self.assertEqual(components, [])
        self.assertIsNone(render_data)

    def test_axial_feret_never_exceeds_3d_feret(self):
        rng = np.random.default_rng(0)
        segmentation = np.zeros((20, 20, 6), dtype=np.uint8)
        segmentation[ndimage_blob(rng, segmentation.shape)] = 2
        for spacing in ((0.7, 0.7, 2.5), (1.0, 1.0, 1.0), (0.8, 1.2, 5.0)):
            components, _ = tumor_components_3d(segmentation, spacing)
            for item in components:
                self.assertLessEqual(item["max_axial_feret_mm"], item["max_feret_3d_mm"] + 1e-9)

    def test_collinear_and_single_voxel_components(self):
        segmentation = np.zeros((10, 10, 3), dtype=np.uint8)
        segmentation[2, 2:8, 1] = 2      # straight line of 6 voxels along Y
        segmentation[8, 8, 0] = 2        # isolated voxel
        components, _ = tumor_components_3d(segmentation, (1.0, 0.5, 2.0))
        by_size = sorted(components, key=lambda item: item["voxel_count"])
        self.assertEqual(by_size[0]["max_feret_3d_mm"], 0.0)
        self.assertAlmostEqual(by_size[1]["max_feret_3d_mm"], 2.5)
        self.assertAlmostEqual(by_size[1]["max_axial_feret_mm"], 2.5)

    def test_lesion_matching_detects_one_of_two_spheres_and_counts_false_positives(self):
        grid = np.indices((40, 40, 20), dtype=float)
        big = ((grid[0] - 10) ** 2 + (grid[1] - 10) ** 2 + (grid[2] - 10) ** 2) <= 64   # 8 mm radius
        small = ((grid[0] - 30) ** 2 + (grid[1] - 30) ** 2 + (grid[2] - 10) ** 2) <= 9  # 3 mm radius
        reference = big | small
        prediction = big.copy()
        prediction[35:38, 2:5, 2:5] = True  # false positive far from both lesions

        lesions, false_positives = match_lesions_3d(reference, prediction, (1.0, 1.0, 1.0))

        self.assertEqual(len(lesions), 2)
        detected = {round(item["max_axial_feret_mm"]): item["detected"] for item in lesions}
        self.assertEqual(sorted(detected.values()), [False, True])
        big_row = max(lesions, key=lambda item: item["voxel_count"])
        self.assertAlmostEqual(big_row["lesion_dice"], 1.0)
        self.assertAlmostEqual(big_row["coverage"], 1.0)
        self.assertEqual(len(false_positives), 1)
        summary = {row["size_group"]: row for row in summarize_lesions_by_size(lesions)}
        self.assertEqual(summary["all"]["n_lesions"], 2)
        self.assertAlmostEqual(summary["all"]["detection_rate"], 0.5)
        self.assertEqual(summary[">=10 mm"]["n_detected"], 1)
        self.assertEqual(summary["<10 mm"]["n_detected"], 0)
        self.assertAlmostEqual(summary["all"]["covered_rate"], 0.5)

    def test_diagonal_fragment_joins_its_lesion_and_touching_is_not_covering(self):
        reference = np.zeros((30, 30, 6), dtype=bool)
        reference[5:10, 5:10, 2:4] = True
        reference[10, 10, 4] = True          # touches the lesion only by a corner
        reference[20:23, 20:23, 2] = True    # small separate lesion
        prediction = np.zeros_like(reference)
        prediction[5:10, 5:10, 2:4] = True
        prediction[18:29, 18:29, 0:6] = True  # large region that swallows the small lesion
        prediction[20:23, 20:23, 2] = False   # ...except the lesion itself, apart from one voxel
        prediction[20, 20, 2] = True
        lesions, _ = match_lesions_3d(reference, prediction, (1, 1, 1))
        self.assertEqual(len(lesions), 2)
        small = min(lesions, key=lambda item: item["voxel_count"])
        self.assertTrue(small["detected"])
        self.assertFalse(small["covered"])
        self.assertEqual(len(match_lesions_3d(reference, prediction, (1, 1, 1), connectivity=6)[0]), 3)

    def test_lesion_matching_min_iou_rejects_a_much_larger_prediction(self):
        reference = np.zeros((30, 30, 5), dtype=bool)
        reference[10:13, 10:13, 2] = True
        prediction = np.zeros_like(reference)
        prediction[2:28, 2:28, 0:5] = True
        loose, _ = match_lesions_3d(reference, prediction, (1, 1, 1), min_iou=0.0)
        strict, _ = match_lesions_3d(reference, prediction, (1, 1, 1), min_iou=0.1)
        self.assertTrue(loose[0]["detected"])
        self.assertFalse(strict[0]["detected"])

    def test_filter_uses_physical_volume_and_axial_span(self):
        mask = np.zeros((10, 10, 6), dtype=bool)
        mask[1:3, 1:3, 1] = True        # 4 voxels, one slice
        mask[6:8, 6:8, 1:4] = True      # 12 voxels, three slices
        spans = filter_components_3d(mask, (1.0, 1.0, 1.0), min_slices=2)
        self.assertEqual(int(spans.sum()), 12)
        coarse = filter_components_3d(mask, (1.0, 1.0, 5.0), min_volume_mm3=30)
        fine = filter_components_3d(mask, (0.5, 0.5, 1.0), min_volume_mm3=30)
        self.assertEqual(int(coarse.sum()), 12)  # 4 x 5 mm³ = 20 < 30 removed; 12 x 5 = 60 kept
        self.assertEqual(int(fine.sum()), 0)


def ndimage_blob(rng, shape):
    from scipy import ndimage
    noise = ndimage.gaussian_filter(rng.random(shape), 1.5)
    return noise > np.quantile(noise, 0.8)


if __name__ == "__main__":
    unittest.main()
