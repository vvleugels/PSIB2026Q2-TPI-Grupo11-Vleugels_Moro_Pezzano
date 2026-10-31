"""Synthetic checks for single-slice filter comparison and selection errors."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from scripts import compare_filters as comparison
from psib.preprocessing import (apply_hu_window, denoise_bilateral,
                                denoise_gaussian, denoise_median)


class CompareFiltersTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / "config.yaml"
        # A sibling local config must override these deliberately unusable paths.
        self.config.write_text("data_dir: missing\noutput_dir: missing\n")
        self.output = self.root / "results"
        self.output.mkdir()
        settings = {
            "data_dir": ".", "output_dir": "results",
            "preprocessing": {
                "hu_window": {"center": 40, "width": 200},
                "denoising": {"method": "gaussian", "median_size": 5,
                              "gaussian_sigma": 2.0, "bilateral_sigma_color": 0.15,
                              "bilateral_sigma_spatial": 2.0},
            },
        }
        (self.root / "config.local.yaml").write_text(yaml.safe_dump(settings))
        self.ct = np.random.default_rng(5).integers(-140, 260, (32, 40, 3), dtype=np.int16)
        self.ct[3:12, 8:25, 1] = 100
        affine = np.diag([-1., 1., 1., 1.])
        nib.save(nib.Nifti1Image(self.ct, affine), self.root / "volume-1.nii.gz")
        nib.save(nib.Nifti1Image(np.ones_like(self.ct, dtype=np.uint8), affine),
                 self.root / "segmentation-1.nii.gz")
        self.selection = self.output / "selected_cases_slices.csv"
        # The invalid second entry ensures only the first selected slice is read.
        pd.DataFrame({"case_id": [1, 999], "slice_index": [1, 999]}).to_csv(
            self.selection, index=False)

    def test_configured_filters_orientation_and_bounded_read(self):
        windowed = apply_hu_window(self.ct[::-1, :, 1], 40, 200)
        expected = [windowed, denoise_median(windowed, 5),
                    denoise_gaussian(windowed, 2.0), denoise_bilateral(windowed, 0.15, 2.0)]
        close = comparison.plt.close
        try:
            with patch.object(nib.Nifti1Image, "get_fdata", side_effect=AssertionError("Full volume read")), \
                    patch.object(comparison.plt, "close"):
                output = comparison.compare_filters(self.config)
                figure = comparison.plt.gcf()
                self.assertEqual(len(figure.axes), 4)
                for ax, image in zip(figure.axes, expected):
                    np.testing.assert_allclose(ax.images[0].get_array(), image.T)
                    self.assertEqual(ax.images[0].get_clim(), (0, 1))
                    self.assertEqual(ax.images[0].origin, "lower")
                self.assertEqual(output.name, "filter_comparison_case_001_slice_001.png")
                self.assertTrue(output.is_file())
                self.assertGreater(comparison.plt.imread(output).size, 0)
        finally:
            close("all")

    def test_missing_and_empty_selection(self):
        self.selection.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Run scripts/select_slices_2d.py"):
            comparison.compare_filters(self.config)
        for text in ("", "case_id,slice_index\n"):
            self.selection.write_text(text)
            with self.assertRaisesRegex(ValueError, "Selection CSV is empty"):
                comparison.compare_filters(self.config)


if __name__ == "__main__":
    unittest.main()
