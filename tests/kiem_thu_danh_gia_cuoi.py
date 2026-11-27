import numpy as np
import pandas as pd

from src.danh_gia_cuoi import case_levels, group_policy_levels
from src.mo_hinh_xac_suat import QGRID
from tests.kiem_thu_mo_hinh_xac_suat import make_panel


def _arr(S, n, H=3):
    base = np.arange(len(QGRID), dtype=float)[None, None, :] * np.ones((S, n, 1))
    return {f"origins_H{H}": np.arange(n), f"A|H{H}|Q": base + 2.0, f"B|H{H}|Q": base + 100.0, f"A|H{H}|mean": np.full((S, n), 2.2)}


def test_group_policy_levels_follow_group_choice():
    P = make_panel()
    arr = _arr(P.n_series, 5)
    pol = pd.DataFrame({"group": ["smooth_fast", "lumpy_slow"], "model": ["A", "B"], "q_star": [0.7, 0.9]})
    lv = group_policy_levels(arr, P, pol, 3)
    k7, k9 = QGRID.index(0.7), QGRID.index(0.9)
    sf = P.group_ds == "smooth_fast"
    assert (lv[0, sf] == (k7 + 2.0)).all() and (lv[0, ~sf] == (k9 + 100.0)).all()


def test_case_levels_point_and_quantile():
    arr = _arr(4, 3)
    assert (case_levels(arr, "A", 3, "point", None) == 3.0).all()
    assert (case_levels(arr, "A", 3, "quantile", 0.5) == 2.0).all()
