"""Checks for the shared workflow behind the Windows launcher."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import yaml

from scripts import run_tpi as launcher


class RunTPITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch.object(launcher, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        clock = patch.object(launcher, "datetime")
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        self.clock.now.return_value = datetime(2026, 10, 31, 18, 4, 9)
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump({
            "data_dir": "configured", "output_dir": "custom-results",
            "selection": {"n_cases": 1, "slices_per_case": 3, "minimum_slice_gap": 1},
        }))
        self.data = self.root / "configured"
        self.fallback = self.root / "data" / "Selected_database"

    def save_data(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        ct = np.full((9, 9, 3), 60, dtype=np.int16)
        labels = np.zeros_like(ct, dtype=np.uint8)
        labels[1:8, 1:8, :] = 1
        labels[3:5, 3:5, :] = 2
        ct[labels == 2] = 0
        nib.save(nib.Nifti1Image(ct, np.eye(4)), directory / "volume-1.nii.gz")
        nib.save(nib.Nifti1Image(labels, np.eye(4)), directory / "segmentation-1.nii.gz")

    def check_data_choice(self, expected):
        configs = []

        def inspect(path, *args):
            configs.append(yaml.safe_load(path.read_text()))
            return self.root / "custom-results" / "segmentation" / "20261031-180409"

        with patch.object(launcher, "select_2d", side_effect=inspect), \
                patch.object(launcher, "run_experiment", side_effect=inspect) as experiment:
            launcher.run_tpi(self.config)
            self.assertEqual(experiment.call_args.args[1], "20261031-180409")
        self.assertEqual(len(configs), 2)
        for cfg in configs:
            self.assertEqual(Path(cfg["data_dir"]), expected)
            self.assertEqual(Path(cfg["output_dir"]), self.root / "custom-results")

    def test_prefers_configured_directory(self):
        self.save_data(self.data)
        self.save_data(self.fallback)
        self.check_data_choice(self.data)

    def test_missing_or_empty_configured_directory_falls_back(self):
        self.save_data(self.fallback)
        self.check_data_choice(self.fallback)
        self.data.mkdir()
        self.check_data_choice(self.fallback)

    def test_errors_when_neither_directory_has_pairs(self):
        with patch.object(launcher, "select_2d") as select:
            with self.assertRaisesRegex(FileNotFoundError, "No paired CT/mask cases"):
                launcher.run_tpi(self.config)
            select.assert_not_called()

    def test_existing_timestamp_does_not_overwrite_selection(self):
        self.save_data(self.data)
        (self.root / "custom-results/segmentation/20261031-180409").mkdir(parents=True)
        with patch.object(launcher, "select_2d") as select:
            with self.assertRaisesRegex(FileExistsError, "next second"):
                launcher.run_tpi(self.config)
            select.assert_not_called()

    def test_local_config_fallback_end_to_end(self):
        self.save_data(self.fallback)
        local = yaml.safe_load(self.config.read_text())
        local["output_dir"] = "local-results"
        (self.root / "config.local.yaml").write_text(yaml.safe_dump(local))
        original = self.config.read_bytes()
        result = launcher.run_tpi(self.config)
        self.assertEqual(result, self.root / "local-results/segmentation/20261031-180409")
        self.assertTrue((result / "metrics_per_slice.csv").is_file())
        effective = yaml.safe_load((result / "effective_config.yaml").read_text())
        self.assertEqual(Path(effective["data_dir"]), self.fallback)
        self.assertEqual(self.config.read_bytes(), original)
        self.assertEqual(yaml.safe_load((self.root / "config.local.yaml").read_text()), local)


if __name__ == "__main__":
    unittest.main()
