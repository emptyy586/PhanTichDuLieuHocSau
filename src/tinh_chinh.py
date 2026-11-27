"""Chọn siêu tham số trên VALIDATION bằng pinball trung bình (mọi q, mọi H)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.mo_hinh_xac_suat import (CONFIG, Panel, avg_pinball, estimate_alpha, nb_quantiles, origins_for_split,
                             stat_forecast, stat_window_mean, QGRID)


def horizons() -> list[int]:
    """Các H = lead_time + review_period từ config."""
    R = int(CONFIG["review_period"])
    return [int(L) + R for L in CONFIG["lead_times"]]


def stat_param_grid(model: str) -> list[dict]:
    """Lưới tham số mức dự báo của từng mô hình thống kê (đọc từ config)."""
    if model == "CrostonSBA":
        return [{"alpha": a} for a in CONFIG["croston_alpha_grid"]]
    if model == "TSB":
        return [{"alpha_d": d, "alpha_p": p} for d, p in itertools.product(CONFIG["tsb_alpha_d_grid"], CONFIG["tsb_alpha_p_grid"])]
    return [{}]


def tune_stat_model(model: str, panel: Panel, split: str = "val") -> tuple[dict, pd.DataFrame]:
    """Hai bước: (1) chọn tham số mức dự báo với kappa=1, w=0.5; (2) chọn (kappa, w_series). Trả về (hparams, log)."""
    Hs = horizons()
    origins = {H: origins_for_split(panel, split, H) for H in Hs}
    log = []
    best_params, best_score = None, np.inf
    for params in stat_param_grid(model):
        fc = {H: stat_forecast(model, panel, H, origins[H], params, kappa=1.0, w_series=0.5) for H in Hs}
        score = avg_pinball(panel, Hs, origins, fc)
        log.append({"model": model, "stage": 1, "params": str(params), "kappa": 1.0, "w_series": 0.5, "pinball_avg": score})
        if score < best_score:
            best_params, best_score = params, score
    # bước 2: tính M một lần, đổi (kappa, w)
    M = {H: stat_window_mean(model, panel.Y, H, best_params) for H in Hs}
    best = (1.0, 0.5, np.inf)
    for kappa, w in itertools.product(CONFIG["tune_kappa"], CONFIG["tune_w_series"]):
        fc = {}
        for H in Hs:
            alpha = estimate_alpha(panel.Y, M[H], H, origins[H], panel.group_ds, int(CONFIG["alpha_hist_window"]), w, kappa)
            mean = np.nan_to_num(M[H][:, origins[H]])
            fc[H] = {"Q": nb_quantiles(mean, alpha, QGRID)}
        score = avg_pinball(panel, Hs, origins, fc)
        log.append({"model": model, "stage": 2, "params": str(best_params), "kappa": kappa, "w_series": w, "pinball_avg": score})
        if score < best[2]:
            best = (kappa, w, score)
    return {"params": best_params, "kappa": best[0], "w_series": best[1], "pinball_avg": best[2]}, pd.DataFrame(log)
