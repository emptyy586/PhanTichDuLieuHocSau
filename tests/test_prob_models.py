import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from src import ml_models as ml
from src.forecast_prob import sample_total_window
from src.prob_models import (QGRID, Panel, croston_sba_level, estimate_alpha, evaluate_forecast, group_masks, nb_quantiles,
                             origins_for_split, pinball_loss, seasonal_naive_window_mean, stat_forecast, tsb_level,
                             window_average_level, window_targets)


def make_panel(S=6, T=260, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    Y = rng.poisson(rng.uniform(0.3, 4.0, size=(S, 1)), size=(S, T)).astype(float)
    dates = pd.date_range("2020-01-01", periods=T)
    ev = np.array([None] * T, dtype=object)
    ev[::17] = "Easter"
    return Panel(
        ids=np.array([f"s{i}" for i in range(S)]), dates=dates, Y=Y,
        price=rng.uniform(1, 3, size=(S, T)), snap=(np.arange(T) % 10 < 3).astype(float),
        event_name=ev, event_type=ev, weekday=dates.weekday.to_numpy(), month=dates.month.to_numpy(),
        demand_type=np.array(["smooth", "lumpy"] * (S // 2)), speed=np.array(["fast", "slow"] * (S // 2)),
        val_start=200, test_start=230,
    )


def test_pinball_known_answer():
    assert pinball_loss(np.array([10.0]), np.array([4.0]), 0.9)[0] == pytest.approx(5.4)  # q*(y-pred)
    assert pinball_loss(np.array([4.0]), np.array([10.0]), 0.9)[0] == pytest.approx(0.6)  # (1-q)*(pred-y)


def test_nb_quantiles_match_scipy_and_cdf_definition():
    mu, alpha = np.array([0.5, 3.0, 20.0]), np.array([4.0, 0.3, 0.05])
    Q = nb_quantiles(mu, alpha, QGRID)
    n, p = 1 / alpha, 1 / (1 + alpha * mu)
    for k, q in enumerate(QGRID):
        assert np.allclose(Q[:, k], nbinom.ppf(q, n, p))
        assert (nbinom.cdf(Q[:, k], n, p) >= q - 1e-12).all()
        assert (nbinom.cdf(Q[:, k] - 1, n, p) < q + 1e-12).all()


def test_nb_quantiles_monotone_in_q_and_zero_mean():
    Q = nb_quantiles(np.array([[0.0, 1.0, 7.0]]), np.array([[0.5, 2.0, 0.1]]), QGRID)
    assert (np.diff(Q, axis=-1) >= 0).all()
    assert (Q[0, 0] == 0).all()


def test_legacy_sampler_mean_not_inflated_when_n_below_one():
    d = sample_total_window(1.0, 25.0, 200000, np.random.default_rng(0))  # n = 1/24 < 1: trước đây mean_out ≈ 24
    assert abs(d.mean() - 1.0) < 0.15


def test_window_targets_values_and_nan_tail():
    Y = np.arange(1.0, 11.0)[None, :]
    W = window_targets(Y, 3)
    assert W[0, 0] == 2 + 3 + 4 and W[0, 6] == 8 + 9 + 10 and np.isnan(W[0, 7:]).all()


def test_origins_window_inside_split():
    P = make_panel()
    for H in (3, 4):
        o = origins_for_split(P, "val", H)
        assert len(o) == 30 - H + 1 and o.min() == P.val_start - 1 and o.max() + H == P.test_start - 1


def test_croston_tsb_constant_demand():
    Y = np.full((1, 100), 3.0)
    assert croston_sba_level(Y, 0.1)[0, -1] == pytest.approx(3.0 * (1 - 0.05))
    assert tsb_level(Y, 0.1, 0.1)[0, -1] == pytest.approx(3.0)
    assert window_average_level(Y, 28)[0, -1] == pytest.approx(3.0)
    assert seasonal_naive_window_mean(Y, 3, 7)[0, -1] == pytest.approx(9.0)


def test_no_leak_stat_forecast_and_alpha():
    P = make_panel()
    t0 = P.val_start + 5
    origins = np.array([t0])
    for model, params in [("WindowAverage", {}), ("SeasonalNaive", {}), ("CrostonSBA", {"alpha": 0.1}), ("TSB", {"alpha_d": 0.1, "alpha_p": 0.1})]:
        a = stat_forecast(model, P, 3, origins, params, 1.0, 0.5, n_hist=60)
        Y2 = P.Y.copy()
        Y2[:, t0 + 1:] = 999.0  # đổi toàn bộ doanh số tương lai
        P2 = Panel(**{**P.__dict__, "Y": Y2})
        b = stat_forecast(model, P2, 3, origins, params, 1.0, 0.5, n_hist=60)
        assert np.allclose(a["Q"], b["Q"]) and np.allclose(a["alpha"], b["alpha"]) and np.allclose(a["mean"], b["mean"]), model


def test_no_leak_features():
    P = make_panel()
    t0 = P.val_start + 3
    X1, names = ml.build_features(P, 3, np.array([t0]), effect_end=P.val_start - 1)
    Y2 = P.Y.copy()
    Y2[:, t0 + 1:] = 777.0
    X2, _ = ml.build_features(Panel(**{**P.__dict__, "Y": Y2}), 3, np.array([t0]), effect_end=P.val_start - 1)
    assert np.array_equal(np.nan_to_num(X1, nan=-1), np.nan_to_num(X2, nan=-1))
    assert not any(n.startswith("fut_") and "sales" in n for n in names)


def test_estimate_alpha_known_poisson_is_small():
    rng = np.random.default_rng(1)
    Y = rng.poisson(3.0, size=(50, 600)).astype(float)
    M = 3.0 * window_average_level(Y, 28) * 0 + 9.0  # dự báo trung bình đúng của cửa sổ H=3
    o = np.array([500])
    a = estimate_alpha(Y, np.full(Y.shape, 9.0), 3, o, np.array(["g"] * 50), 180, 0.0)
    assert a.shape == (50, 1) and a.mean() < 0.05 and M[0, 0] == 9.0


def test_evaluate_forecast_coverage_and_shapes():
    P = make_panel()
    H = 3
    o = origins_for_split(P, "val", H)
    W = window_targets(P.Y, H)[:, o]
    fc = {"Q": np.repeat(W[..., None], len(QGRID), axis=-1), "mean": W}
    ev = evaluate_forecast("oracle", H, P, o, fc)
    ov = ev[(ev.group_type == "overall")]
    assert ov[ov.metric == "coverage"].value.eq(1.0).all()
    assert ov[ov.metric == "coverage_lt"].value.eq(0.0).all()
    assert ov[ov.metric == "pinball_avg"].value.iloc[0] == 0.0
    assert ev[ev.metric == "n_series"].groupby("group_type").value.sum().to_dict() == {
        "demand_type": 6.0, "demand_type_x_speed": 6.0, "overall": 6.0, "speed": 6.0}
    assert len(group_masks(P)) == 1 + 2 + 2 + 2


def test_lgbm_quantiles_monotone_and_shape():
    P = make_panel()
    X, y, org, names = ml.make_training_set(P, 3, P.val_start - 1, P.val_start - 1)
    assert X.shape[0] == len(y) == len(org) and X.shape[1] == len(names)
    assert org.max() + 3 <= P.val_start - 1  # nhãn không vượt quá ngày cuối train
    models = ml.fit_lgbm(X, y)
    o = origins_for_split(P, "val", 3)
    Xe, _ = ml.build_features(P, 3, o, P.val_start - 1)
    out = ml.predict_lgbm(models, Xe, Xe.shape[:2], Xe[..., names.index("age_days")] < 0)
    assert out["Q"].shape == (6, len(o), len(QGRID)) and (np.diff(out["Q"], axis=-1) >= 0).all() and (out["Q"] >= 0).all()
