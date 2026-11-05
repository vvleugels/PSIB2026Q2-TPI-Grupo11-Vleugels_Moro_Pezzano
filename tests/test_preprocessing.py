import unittest

import numpy as np

from psib.preprocessing import (apply_hu_window, crop_to_liver, denoise_bilateral,
                                denoise_gaussian, denoise_mean, enhance, liver_bounding_box,
                                prepare_denoised_slice, prepare_enhanced_slice, preprocess_slice)


class PreprocessingTests(unittest.TestCase):
    def test_contrast_scales_around_midgray_after_equalization(self):
        image = np.array([[0.1, 0.25, 0.5, 0.75, 0.9]], dtype=np.float32)
        np.testing.assert_array_equal(enhance(image, {"method": "none"}), image)
        np.testing.assert_allclose(enhance(image, {"method": "none", "contrast": 0.5}),
                                   [[0.3, 0.375, 0.5, 0.625, 0.7]])
        np.testing.assert_allclose(enhance(image, {"method": "none", "contrast": 2}), [[0, 0, 0.5, 1, 1]])
        method = {"method": "histogram_equalization"}
        np.testing.assert_allclose(enhance(image, {**method, "contrast": 0.5}),
                                   (enhance(image, method) - 0.5) * 0.5 + 0.5)
        with self.assertRaisesRegex(ValueError, "contraste"):
            enhance(image, {"contrast": float("nan")})

    def test_hu_window_clips_and_scales(self):
        result = apply_hu_window(np.array([[-300, -140, 60, 260, 1000]], dtype=np.int16), 60, 400)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[0, 0, 0.5, 1, 1]])
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

    def test_disabled_hu_window_uses_the_full_slice_range(self):
        image = np.array([[-100, -20, 300]], dtype=np.float32)
        settings = {"hu_window": {"enabled": False, "width": 0},
                    "denoising": {"method": "none"}, "enhancement": {"method": "none"}}
        expected = [[0, 0.2, 1]]
        np.testing.assert_allclose(prepare_denoised_slice(image, settings), expected)
        result = preprocess_slice(image, np.ones_like(image, dtype=bool), settings)
        np.testing.assert_allclose(result["windowed"], expected)
        np.testing.assert_array_equal(prepare_denoised_slice(np.full_like(image, 300), settings),
                                      np.zeros_like(image))

    def test_liver_preprocessing_is_independent_of_exterior_pixels(self):
        image = np.arange(80, dtype=np.float32).reshape(8, 10)
        liver = np.zeros(image.shape, dtype=bool)
        liver[1:7, 2:8] = True
        liver[2:5, 4:6] = False
        modified = np.where(liver, image, np.nan)
        for denoising in ('none', 'gaussian', 'mean', 'median', 'bilateral'):
            for enhancement in ('none', 'histogram_equalization', 'clahe'):
                with self.subTest(denoising=denoising, enhancement=enhancement):
                    settings = {'hu_window': {'enabled': False}, 'denoising': {'method': denoising},
                                'enhancement': {'method': enhancement, 'contrast': 1.25, 'clahe_kernel_size': 3}}
                    expected = prepare_enhanced_slice(image, settings, liver_mask=liver)
                    result = prepare_enhanced_slice(modified, settings, liver_mask=liver)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertFalse(result[~liver].any())
                    np.testing.assert_array_equal(result, expected)

    def test_masked_gaussian_preserves_constant_liver_at_the_boundary(self):
        liver = np.zeros((7, 7), dtype=bool)
        liver[2:5, 2:5] = True
        image = np.where(liver, 60, -1000)
        denoised = prepare_denoised_slice(image, {'hu_window': {'center': 60}}, liver_mask=liver)
        np.testing.assert_allclose(denoised[liver], 0.5)
        self.assertFalse(denoised[~liver].any())

    def test_crop_coordinates_and_bounds(self):
        image = np.arange(100).reshape(10, 10)
        liver = np.zeros_like(image, dtype=bool)
        liver[2:5, 3:7] = True
        crop, box = crop_to_liver(image, liver, margin=2)
        self.assertEqual(box, (0, 1, 7, 9))
        np.testing.assert_array_equal(crop, image[0:7, 1:9])
        self.assertEqual(liver_bounding_box(liver, 0), (2, 3, 5, 7))

    def test_filters_preserve_shape_and_float_dtype(self):
        image = np.random.default_rng(2).random((12, 13), dtype=np.float32)
        for result in (denoise_bilateral(image), denoise_gaussian(image), denoise_mean(image)):
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
