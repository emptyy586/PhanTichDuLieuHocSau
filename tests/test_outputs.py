"""Kiểm tra tích hợp trên file kết quả thật (cần đã chạy `python -m src.run_forecasts --split val`)."""
import numpy as np
import pandas as pd

from src.prob_models import QGRID, load_panel, window_targets
from src.report_utils import md_table
from src.utils import ROOT

MODELS = ["WindowAverage", "SeasonalNaive", "CrostonSBA", "TSB", "LGBM-Q", "LGBM-PoisNB", "MLP-NB"]


def test_val_npz_shapes_cover_all_series_and_origins_and_monotone():
    arr = np.load(ROOT / "results" / "forecasts" / "val.npz", allow_pickle=True)
    assert len(arr["ids"]) == 823
    for H in (3, 4):
        n = 28 - H + 1  # số gốc có cửa sổ nằm trọn trong 28 ngày val
        assert arr[f"origins_H{H}"].shape == (n,)
        assert arr[f"actual_H{H}"].shape == (823, n)
        for m in MODELS:
            Q = arr[f"{m}|H{H}|Q"]
            assert Q.shape == (823, n, len(QGRID)), (m, H)
            assert np.isfinite(Q).all() and (Q >= 0).all()
            assert (np.diff(Q, axis=-1) >= -1e-9).all(), f"quantile giảm theo q: {m} H={H}"  # monotone
            assert arr[f"{m}|H{H}|mean"].shape == (823, n)


def test_val_eval_csv_schema_and_coverage_matches_npz():
    ev = pd.read_csv(ROOT / "results" / "forecast_eval_val_all.csv")
    assert list(ev.columns) == ["model", "H", "group_type", "group", "metric", "q", "value"]
    assert set(ev.model) == set(MODELS) and set(ev.H) == {3, 4}
    assert {"overall", "speed", "demand_type", "demand_type_x_speed"} == set(ev.group_type)
    arr = np.load(ROOT / "results" / "forecasts" / "val.npz", allow_pickle=True)
    Q = arr["TSB|H3|Q"]
    W = arr["actual_H3"]
    cov = (W[..., None] <= Q).mean(axis=(0, 1))
    got = ev[(ev.model == "TSB") & (ev.H == 3) & (ev.group == "overall") & (ev.metric == "coverage")].sort_values("q").value.to_numpy()
    assert np.allclose(cov, got)


def test_actual_in_npz_equals_window_sum_of_val_demand():
    panel = load_panel()
    arr = np.load(ROOT / "results" / "forecasts" / "val.npz", allow_pickle=True)
    W = window_targets(panel.Y, 4)[:, arr["origins_H4"]]
    assert np.array_equal(W, arr["actual_H4"])
    assert arr["origins_H4"].min() == panel.val_start - 1


def test_md_table_rounds_to_four_decimals():
    df = pd.DataFrame({"a": [0.123456789]}, index=["x"])
    assert "0.1235" in md_table(df)
