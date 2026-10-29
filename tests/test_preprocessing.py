import unittest

import numpy as np

from psib.preprocessing import (apply_hu_window, crop_to_liver, denoise_bilateral,
                                denoise_gaussian, denoise_median, liver_bounding_box)


class PreprocessingTests(unittest.TestCase):
    def test_hu_window_clips_and_scales(self):
        result = apply_hu_window(np.array([[-300, -140, 60, 260, 1000]], dtype=np.int16), 60, 400)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[0, 0, 0.5, 1, 1]])
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 1)

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
        for result in (denoise_median(image), denoise_gaussian(image), denoise_bilateral(image)):
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
