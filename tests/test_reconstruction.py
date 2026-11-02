"""Small synthetic validation of the simulated reconstruction utilities."""
import unittest
import warnings

import numpy as np
from skimage.transform import radon

from psib.reconstruction import hu_to_mu, mu_to_hu, psnr, simulate_reconstruction


class ReconstructionTests(unittest.TestCase):
    @staticmethod
    def disc(size=64):
        y, x = np.ogrid[:size, :size]
        image = np.full((size, size), -1000.0)
        image[(x - size / 2) ** 2 + (y - size / 2) ** 2 < (size * .28) ** 2] = 0.0
        return image

    def test_hu_mu_round_trip(self):
        hu = np.array([-1000.0, -650.0, -120.0, 0.0, 70.0])
        np.testing.assert_allclose(mu_to_hu(hu_to_mu(hu)), hu)

    def test_disc_sinogram_uses_requested_angle_count_and_projection_support(self):
        image = hu_to_mu(self.disc())
        sinogram = radon(image, theta=np.linspace(0, 180, 20, endpoint=False),
                         circle=True, preserve_range=True)
        self.assertEqual(sinogram.shape[1], 20)
        self.assertGreater(float(sinogram.max()), 0)
        self.assertTrue(np.all(np.isfinite(sinogram)))

    def test_fully_nonzero_rectangular_roi_fits_inside_radon_circle(self):
        image = np.zeros((31, 46), dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulate_reconstruction(image, angle_count=12)
        self.assertFalse(any("outside the reconstruction circle" in str(item.message) for item in caught))

    def test_fbp_ramp_reconstructs_a_disc_with_bounded_error(self):
        hu = self.disc()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reconstructions, sinogram, theta = simulate_reconstruction(hu, angle_count=180)
        self.assertFalse(any("outside the reconstruction circle" in str(item.message) for item in caught))
        error = np.sqrt(np.mean((hu_to_mu(reconstructions["fbp_ramp"]) - hu_to_mu(hu)) ** 2))
        self.assertLess(error, 0.06)
        self.assertEqual(sinogram.shape[1], len(theta))

    def test_sart_runs_and_error_improves_with_iterations(self):
        hu = self.disc(48)
        truth = hu_to_mu(hu)
        errors = []
        for iterations in (1, 2, 4):
            reconstructions, _, _ = simulate_reconstruction(
                hu, angle_count=60, sart_iterations=iterations)
            errors.append(float(np.sqrt(np.mean((hu_to_mu(reconstructions["sart"]) - truth) ** 2))))
        self.assertTrue(errors[2] < errors[1] < errors[0], errors)

    def test_psnr_identical_arrays_is_infinite(self):
        image = np.array([[0.0, 0.4], [0.7, 1.0]])
        self.assertEqual(psnr(image, image), float("inf"))


if __name__ == "__main__":
    unittest.main()
