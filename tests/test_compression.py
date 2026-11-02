"""Synthetic checks for the educational compression experiment."""
import unittest

import cv2
import numpy as np

from psib.compression import huffman_encode_decode, jpeg_like_dct
from scripts.run_compression import _encode_png, quantize_8bit


class CompressionTests(unittest.TestCase):
    def test_huffman_round_trip(self):
        symbols = [0, 1, 1, 2, 0, 3, 3, 3, 2, 1]
        decoded, table, bit_count = huffman_encode_decode(symbols)
        self.assertEqual(decoded, symbols)
        self.assertEqual(set(table), set(symbols))
        self.assertGreater(bit_count, 0)

    def test_png_lossless_round_trip_and_8bit_quantization(self):
        image = np.arange(63, dtype=np.uint8).reshape(7, 9)
        decoded, size = _encode_png(image)
        np.testing.assert_array_equal(decoded, image)
        self.assertGreater(size, 0)
        np.testing.assert_array_equal(quantize_8bit(image.astype(float) / 255), image)

    def test_dct_quality_100_is_close_to_control(self):
        y, x = np.mgrid[:24, :24]
        image = ((x * 7 + y * 5 + ((x // 4 + y // 4) % 2) * 3) % 256).astype(np.uint8)
        restored, estimate = jpeg_like_dct(image, 100)
        self.assertEqual(restored.shape, image.shape)
        self.assertLessEqual(float(np.mean(np.abs(restored.astype(float) - image))), 1.0)
        self.assertGreater(estimate, 0)

    def test_real_jpeg_size_decreases_with_quality_on_synthetic_image(self):
        y, x = np.mgrid[:128, :128]
        image = ((x * 3 + y * 5 + (x * y) % 37) % 256).astype(np.uint8)
        sizes = []
        for quality in (95, 75, 50, 25, 10):
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            self.assertTrue(ok)
            sizes.append(encoded.nbytes)
        self.assertTrue(all(a > b for a, b in zip(sizes, sizes[1:])), sizes)


if __name__ == "__main__":
    unittest.main()
