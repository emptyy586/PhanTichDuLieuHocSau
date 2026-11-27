import numpy as np

from src.loai_bo_dac_trung import EFFECT_COLS, effect_mask
from src.mo_phong_ton_kho import MODELS, quantile_to_levels
from src.mo_hinh_xac_suat import QGRID, nb_quantiles
from src.tien_ich import ROOT


def test_lgbm_q_levels_are_nonnegative_integers_and_round_to_nearest():
    q = np.array([-0.2, 0.0, 0.3, 0.49, 0.5, 0.6, 1.49, 2.51])
    lv = quantile_to_levels("LGBM-Q", q)
    assert (lv >= 0).all() and np.array_equal(lv, np.round(lv))
    assert lv.tolist() == [0, 0, 0, 0, 1, 1, 1, 3]  # 0.3→0, 0.6→1


def test_nb_models_are_not_changed_by_rounding():
    q = nb_quantiles(np.array([0.5, 3.0, 12.0]), np.array([1.0, 0.3, 0.1]), QGRID)  # số nguyên
    for m in MODELS:
        if m != "LGBM-Q":
            assert np.array_equal(quantile_to_levels(m, q), q)


def test_real_forecasts_levels_used_in_simulation_are_integers():
    for split in ("val", "test"):
        arr = np.load(ROOT / "results" / "forecasts" / f"{split}.npz", allow_pickle=True)
        for H in (3, 4):
            for key in ("Q", "Qext"):
                for m in MODELS:
                    lv = quantile_to_levels(m, arr[f"{m}|H{H}|{key}"])
                    assert (lv >= 0).all() and np.array_equal(lv, np.round(lv)), (split, H, key, m)


def test_smaller_kappa_gives_narrower_upper_tail_and_spread():
    """Kappa nhỏ hơn (alpha nhỏ hơn): phân vị đuôi trên (q>=0.95) không lớn hơn và khoảng Q95−Q50 không rộng hơn.
    LƯU Ý: không đúng ở MỌI q: với trung bình cố định, alpha nhỏ làm phân vị thấp/giữa (q<=0.9 khi alpha lớn) có thể TĂNG (ít khối lượng ở 0)."""
    ks = [0.25, 0.35, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    qs = [0.5, 0.95, 0.99]
    for mu in (0.3, 1.0, 3.0, 10.0, 30.0):
        for a in (0.2, 1.0, 3.0):
            Q = np.stack([nb_quantiles(np.array(mu), np.array(a * k), qs) for k in ks])  # [K, nq]
            assert (np.diff(Q[:, 1:], axis=0) >= 0).all(), (mu, a)  # q >= 0.95 không giảm theo kappa
            assert (np.diff(Q[:, 1] - Q[:, 0]) >= 0).all(), (mu, a)  # Q95 - Q50 không giảm theo kappa


def test_effect_mask_drops_only_the_two_effect_columns():
    names = ["mean_28", "fut_wd_index", "fut_event_index", "fut_snap_days", "price_fut"]
    assert effect_mask(names).tolist() == [True, False, False, True, True] and set(EFFECT_COLS) == {"fut_wd_index", "fut_event_index"}
