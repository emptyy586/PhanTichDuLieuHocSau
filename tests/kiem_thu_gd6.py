import numpy as np
import pandas as pd

from src import mo_hinh_ml as ml
from src.loai_bo_dac_trung import exog_mask
from src.mo_hinh_xac_suat import (EXT_QGRID, QGRID, mase_scale, nb_quantiles, origins_for_split, window_targets)
from src.diem_an_toan import k_grid, series_cost
from src.mo_phong_ton_kho import simulate_order_up_to
from tests.kiem_thu_mo_hinh_xac_suat import make_panel


def test_k_grid_is_0p5_to_3p0_step_0p1():
    k = k_grid()
    assert k[0] == 0.5 and k[-1] == 3.0 and len(k) == 26 and np.isclose(np.diff(k), 0.1).all() and 1.0 in k


def test_series_cost_matches_hold_plus_ratio_times_lost():
    demand = np.array([[2.0, 0.0, 5.0, 1.0, 3.0, 2.0]])
    levels = np.full((1, 1, 3), 4.0)
    res = simulate_order_up_to(demand, levels, 1)
    assert np.allclose(series_cost(res, 10.0), res["hold_units"] + 10.0 * res["lost"])


def test_mase_scale_uses_train_only_and_equals_hand_calculation():
    P = make_panel(S=2)
    P.Y[:] = 0.0
    P.Y[0, :] = np.arange(P.n_days) % 2  # 0,1,0,1,...; chuỗi 1 toàn 0
    H = 3
    scale = mase_scale(P, H)
    W = window_targets(P.Y, H)
    first = 1
    s = np.arange(first + H, P.val_start - 1 - H + 1)
    assert np.isclose(scale[0], np.abs(W[0, s] - W[0, s - H]).mean())
    P2 = type(P)(**{**P.__dict__, "Y": np.where(np.arange(P.n_days)[None, :] >= P.val_start, 999.0, P.Y)})
    assert np.isclose(mase_scale(P2, H)[0], scale[0])  # đổi dữ liệu val/test không làm đổi scale


def test_ext_quantiles_below_main_grid_and_monotone():
    assert max(EXT_QGRID) < min(QGRID)
    mu, a = np.array([[0.5, 3.0, 9.0]]), np.array([[1.0, 0.3, 0.1]])
    both = nb_quantiles(mu, a, EXT_QGRID + QGRID)
    assert (np.diff(both, axis=-1) >= 0).all()


def test_lgbm_ext_quantiles_not_above_first_main_quantile():
    P = make_panel()
    X, y, org, names = ml.make_training_set(P, 3, P.val_start - 1, P.val_start - 1)
    models = ml.fit_lgbm(X, y, extra_q=EXT_QGRID)
    o = origins_for_split(P, "val", 3)
    Xe, _ = ml.build_features(P, 3, o, P.val_start - 1)
    out = ml.predict_lgbm(models, Xe, Xe.shape[:2], Xe[..., names.index("age_days")] < 0)
    assert out["Qext"].shape == (6, len(o), len(EXT_QGRID))
    assert (np.diff(np.concatenate([out["Qext"], out["Q"]], axis=-1), axis=-1) >= -1e-9).all()


def test_exog_mask_drops_only_price_snap_event_columns():
    P = make_panel()
    _, names = ml.build_features(P, 3, np.array([P.val_start]), P.val_start - 1)
    keep = exog_mask(names)
    dropped = {n for n, k in zip(names, keep) if not k}
    assert dropped == {"price_fut", "price_ratio_fut_past28", "price_ratio_next_today", "fut_snap_days", "fut_event_days", "fut_event_index"}
    assert "mean_28" in names and keep[names.index("mean_28")] and keep[names.index("fut_wd_next")]


def test_mlp_seed_changes_weights_but_same_seed_reproduces():
    import torch
    P = make_panel()
    X, y, org, names = ml.make_training_set(P, 3, P.val_start - 1, P.val_start - 1)
    a = ml.fit_mlp_nb(X, y, org, names, 3, fixed_epochs=1, seed=1)
    b = ml.fit_mlp_nb(X, y, org, names, 3, fixed_epochs=1, seed=1)
    c = ml.fit_mlp_nb(X, y, org, names, 3, fixed_epochs=1, seed=2)
    w = lambda m: m["net"][0].weight.detach().numpy()  # noqa: E731
    assert np.allclose(w(a), w(b)) and not np.allclose(w(a), w(c))
