import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psib.morphology3d import tumor_components_3d


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


if __name__ == "__main__":
    unittest.main()
