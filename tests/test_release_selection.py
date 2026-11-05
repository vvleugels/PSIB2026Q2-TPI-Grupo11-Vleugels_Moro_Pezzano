"""La selección final debe cubrir todos los pacientes disponibles del lote."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from scripts.select_slices_2d import select_2d


class ReleaseSelectionTests(unittest.TestCase):
    def test_final_config_selects_all_131_patients_and_five_slices(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config.yaml').read_text())
        self.assertNotIn('case_ids', cfg['selection'])
        self.assertEqual(cfg['selection']['n_cases'], 131)
        self.assertEqual(cfg['selection']['slices_per_case'], 5)
        # IDs deliberately non-consecutive: membership, not range(131), matters.
        available = list(range(10, 272, 2))
        def slices(*args, **kwargs):
            for k, area in enumerate([0, 25, 40, 60, 100]):
                labels = np.ones((12, 12), dtype=np.uint8)
                labels.flat[:area] = 2
                yield k, None, labels, (1., 1., 15.)
        with tempfile.TemporaryDirectory() as temp:
            cfg['output_dir'] = str(Path(temp) / 'results')
            config = Path(temp) / 'config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            with patch('scripts.select_slices_2d.list_cases', return_value=available), \
                 patch('scripts.select_slices_2d.iter_case_slices', side_effect=slices), \
                 redirect_stdout(StringIO()):
                output = select_2d(config)
            rows = pd.read_csv(output)
            self.assertEqual(set(rows.case_id), set(available))
            self.assertEqual(len(rows), 655)
            self.assertTrue((rows.groupby('case_id').size() == 5).all())
