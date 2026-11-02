import unittest

import numpy as np
import pandas as pd

from scripts.analyze_agreement import agreement_statistics
from scripts.compare_methods import compare_pair, holm_adjust


class AgreementTests(unittest.TestCase):
    def test_bland_altman_reports_known_bias_and_relative_error(self):
        frame = pd.DataFrame({
            "status": ["TP", "TP", "FN"],
            "size_class": ["<10", "10-20", "none"],
            "area_ref_mm2": [10.0, 20.0, 0.0], "area_pred_mm2": [12.0, 24.0, 0.0],
            "max_feret_ref_mm": [5.0, 10.0, np.nan], "max_feret_pred_mm": [6.0, 12.0, np.nan],
        })

        summary = agreement_statistics(frame).set_index("measure")

        self.assertAlmostEqual(summary.loc["area", "mean_difference"], 3.0)
        self.assertAlmostEqual(summary.loc["area", "mean_percent_difference"], 20.0)
        self.assertAlmostEqual(summary.loc["max_feret", "mean_difference"], 1.5)
        self.assertEqual(summary.loc["area", "n_matched_groups"], 2)

    def test_compare_pair_uses_case_level_differences_and_exact_wilcoxon(self):
        frame = pd.DataFrame([
            {"configuration_id": cfg, "case_id": case, "dice": score}
            for case, (a, b) in enumerate(zip((0.8, 0.7, 0.9, 0.6, 0.75, 0.85),
                                               (0.7, 0.6, 0.8, 0.5, 0.65, 0.75)), start=1)
            for cfg, score in (("a", a), ("b", b))
        ])

        result = compare_pair(frame, "a", "b")

        self.assertEqual(result["n_cases"], 6)
        self.assertAlmostEqual(result["median_difference_a_minus_b"], 0.1)
        self.assertAlmostEqual(result["p_value"], 0.03125)
        self.assertEqual(result["direction"], "A>B")
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
