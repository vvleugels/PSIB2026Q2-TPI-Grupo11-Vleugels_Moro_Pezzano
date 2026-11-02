"""Tiny synthetic NIfTI integrations for both report experiments."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from scripts.run_compression import run_experiment as run_compression
from scripts.run_reconstruction import run_experiment as run_reconstruction


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        ct = np.full((24, 24, 3), 60, dtype=np.int16)
        seg = np.zeros((24, 24, 3), dtype=np.uint8)
        seg[3:21, 3:21, :] = 1
        seg[8:12, 8:12, 0] = 2
        seg[9:11, 9:11, 1] = 2
        ct[seg == 2] = -50
        nib.save(nib.Nifti1Image(ct, np.eye(4)), self.data / "volume-1.nii.gz")
        nib.save(nib.Nifti1Image(seg, np.eye(4)), self.data / "segmentation-1.nii.gz")
        self.output = self.root / "results"
        self.output.mkdir()
        cfg = {"data_dir": str(self.data), "output_dir": str(self.output),
               "preprocessing": {"hu_window": {"center": 60, "width": 400},
                                 "denoising": {"method": "median", "median_size": 1}},
               "segmentation": {"method": "global", "threshold": .4},
               "compression": {"qualities": [75, 25, 10]},
               "reconstruction": {"angles": [20], "downsample": 1, "n_cases": 1,
                                  "slices_per_case": 3, "sart_iterations": 1, "margin": 0}}
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        selection = pd.DataFrame({"selection_rank": [1, 1, 1], "case_id": [1, 1, 1],
                                  "slice_index": [0, 1, 2],
                                  "slice_role": ["largest", "small", "tumor_free"]})
        selection.to_csv(self.output / "selected_cases_slices.csv", index=False)

    def test_compression_runner_writes_per_slice_and_figures(self):
        out = run_compression(self.config, "tiny")
        frame = pd.read_csv(out / "per_slice.csv")
        self.assertEqual(frame.case_id.nunique(), 1)
        self.assertIn("texture_tumor_contrast_relative_change", frame.columns)
        self.assertTrue((out / "summary.csv").is_file())
        self.assertTrue((out / "figures" / "compression_example_grid.png").is_file())

    def test_reconstruction_runner_writes_metrics_and_figures(self):
        out = run_reconstruction(self.config, "tiny")
        frame = pd.read_csv(out / "per_slice.csv")
        self.assertEqual(set(frame.method), {"sbp", "fbp_ramp", "fbp_hann", "sart"})
        self.assertEqual(set(frame.slice_role), {"largest", "small", "tumor_free"})
        self.assertTrue((out / "summary.csv").is_file())
        self.assertTrue((out / "figures" / "sinogram.png").is_file())
        self.assertTrue((out / "figures" / "reconstruction_grid.png").is_file())


if __name__ == "__main__":
    unittest.main()
