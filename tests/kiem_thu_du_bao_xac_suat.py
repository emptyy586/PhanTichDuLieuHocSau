import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.du_bao_xac_suat import evaluate_validation_summary, rolling_origin_eval, sample_total_window


class ForecastProbEvaluationTest(unittest.TestCase):
    def test_evaluate_validation_summary_matches_by_date_and_id(self):
        data = pd.DataFrame(
            {
                "id": ["A", "A"],
                "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "y": [2.0, 100.0],
                "split": ["val", "val"],
            }
        )
        forecast = pd.DataFrame(
            {
                "id": ["A", "A"],
                "origin_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "q": [0.5, 0.5],
                "quantile_value": [1.0, 100.0],
            }
        )

        summary = evaluate_validation_summary(data, forecast)
        wape_row = summary[(summary["metric"] == "wape") & (summary["q"] == 0.5)].iloc[0]

        self.assertAlmostEqual(wape_row["value"], 1.0 / 102.0)

    def test_sample_total_window_does_not_blow_up_for_zero_mean(self):
        draws = sample_total_window(mean=0.0, variance=1.0, n_samples=500)

        self.assertTrue(np.all(draws >= 0))
        self.assertLess(float(draws.max()), 20.0)

    def test_poisson_distribution_scaling(self):
        daily = sample_total_window(mean=3.0, variance=3.0, n_samples=30000)
        window = sample_total_window(mean=9.0, variance=9.0, n_samples=30000)

        self.assertTrue(np.quantile(daily, 0.5) > 2.0)
        self.assertTrue(np.quantile(daily, 0.5) < 5.0)
        self.assertTrue(np.quantile(window, 0.5) > 6.0)
        self.assertTrue(np.quantile(window, 0.5) < 13.0)

    def test_pinball_known_answer(self):
        actual = np.array([3.0, 7.0])
        pred = np.array([2.0, 8.0])
        q = 0.5

        pinball = np.maximum(q * (actual - pred), (q - 1.0) * (actual - pred))
        self.assertAlmostEqual(float(np.mean(pinball)), 0.5)

    def test_join_rows_do_not_duplicate(self):
        data = pd.DataFrame(
            {"id": ["A", "A"], "date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "y": [1.0, 2.0], "split": ["val", "val"]}
        )
        forecast = pd.DataFrame(
            {"id": ["A", "A"], "origin_date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "q": [0.5, 0.5], "quantile_value": [1.0, 2.0]}
        )

        merged = data[["id", "date", "y"]].merge(forecast[["id", "target_date", "q", "quantile_value"]], left_on=["id", "date"], right_on=["id", "target_date"], how="inner")
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged["id"].nunique(), 1)

    def test_n_samples_changes_quantile_output(self):
        q_small = sample_total_window(mean=3.0, variance=100.0, n_samples=500)
        q_large = sample_total_window(mean=3.0, variance=100.0, n_samples=5000)

        self.assertNotEqual(float(np.quantile(q_small, 0.999)), float(np.quantile(q_large, 0.999)))

    def test_daily_target_date_tracks_horizon_offset(self):
        dates = pd.date_range("2020-01-01", periods=40, freq="D")
        rows = []
        for idx, d in enumerate(dates):
            rows.append({"id": "A", "date": d, "y": 1.0 + (idx % 5), "split": "train" if idx < 30 else "val"})
        data = pd.DataFrame(rows)

        forecast = rolling_origin_eval(data, split="val")
        daily = forecast[(forecast["level"] == "daily") & (forecast["q"] == 0.5)].copy()
        offset_days = (daily["target_date"] - daily["origin_date"]).dt.days
        self.assertTrue((offset_days == 3).all())
        self.assertEqual(daily["target_date"].nunique(), daily.shape[0])


if __name__ == "__main__":
    unittest.main()
