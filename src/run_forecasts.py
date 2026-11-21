"""Chạy toàn bộ mô hình dự báo xác suất cho một split (val hoặc test) và xuất kết quả.

    python -m src.run_forecasts --split val     # tinh chỉnh siêu tham số trên val, ghi hparams.json
    python -m src.run_forecasts --split test    # dùng hparams.json đã chốt, huấn luyện lại trên train+val

Đầu ra: results/forecasts/{split}.npz, results/forecast_eval_{split}_all.csv, results/hparams.json (chỉ khi val).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import ml_models as ml
from src.prob_models import (CONFIG, QGRID, Panel, avg_pinball, estimate_alpha, evaluate_forecast, load_panel, nb_quantiles,
                             origins_for_split, stat_forecast, window_targets)
from src.tuning import horizons, tune_stat_model
from src.utils import ROOT

STAT_MODELS = ["WindowAverage", "SeasonalNaive", "CrostonSBA", "TSB"]
HPARAMS_PATH = ROOT / "results" / "hparams.json"


def _poisson_nb_forecast(panel: Panel, H: int, origins: np.ndarray, models: dict, effect_end: int, kappa: float,
                         w_series: float) -> dict:
    """LGBM-Poisson + NB: mean = LightGBM Poisson; alpha ước lượng cuốn chiếu từ phần dư của chính mô hình trên các gốc
    lịch sử (trong mẫu đối với phần train, ngoài mẫu đối với phần val/test đã quan sát)."""
    n_hist = int(CONFIG["alpha_hist_window"])
    hist_org = np.arange(max(origins.min() - H - n_hist, 28), origins.max() + 1)
    X, names = ml.build_features(panel, H, hist_org, effect_end)
    S, n = X.shape[:2]
    pred = np.maximum(models["poisson"].predict(X.reshape(S * n, -1)), 0.0).reshape(S, n)
    pred[X[..., names.index("age_days")] < 0] = 0.0
    M = np.full((panel.n_series, panel.n_days), np.nan)
    M[:, hist_org] = pred
    alpha = estimate_alpha(panel.Y, M, H, origins, panel.group_ds, n_hist, w_series, kappa)
    mean = pred[:, np.searchsorted(hist_org, origins)]
    return {"mean": mean, "alpha": alpha, "Q": nb_quantiles(mean, alpha, QGRID)}


def run(split: str) -> None:
    """Chạy toàn bộ pipeline dự báo cho `split` và ghi kết quả."""
    t_start = time.time()
    panel = load_panel()
    Hs = horizons()
    origins = {H: origins_for_split(panel, split, H) for H in Hs}
    last_label_day = panel.val_start - 1 if split == "val" else panel.test_start - 1
    effect_end = last_label_day
    hp: dict = {} if split == "val" else json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))
    tuning_logs, forecasts, timing = [], {}, {}

    # ---- mô hình thống kê + NB
    for m in STAT_MODELS:
        t0 = time.time()
        if split == "val":
            hp[m], log = tune_stat_model(m, panel, "val")
            tuning_logs.append(log)
        forecasts[m] = {H: stat_forecast(m, panel, H, origins[H], hp[m]["params"], hp[m]["kappa"], hp[m]["w_series"]) for H in Hs}
        timing[m] = time.time() - t0
        print(f"[{split}] {m}: {hp[m]}  ({timing[m]:.1f}s)", flush=True)

    # ---- LightGBM (phân vị trực tiếp + Poisson), MLP-NB
    lgb_models, mlp_models, feats = {}, {}, {}
    for H in Hs:
        t0 = time.time()
        X, y, org_row, names = ml.make_training_set(panel, H, last_label_day, effect_end)
        print(f"[{split}] H={H}: train rows={len(y)}, features={len(names)}", flush=True)
        lgb_models[H] = ml.fit_lgbm(X, y)
        timing[f"LGBM_fit_H{H}"] = time.time() - t0
        print(f"[{split}] LightGBM H={H} xong ({timing[f'LGBM_fit_H{H}']:.1f}s)", flush=True)
        t0 = time.time()
        mlp_models[H] = ml.fit_mlp_nb(X, y, org_row, names, H, fixed_epochs=None if split == "val" else hp["MLP-NB"]["epochs"][str(H)])
        timing[f"MLP_fit_H{H}"] = time.time() - t0
        print(f"[{split}] MLP-NB H={H}: epoch={mlp_models[H]['best_epoch']} ({timing[f'MLP_fit_H{H}']:.1f}s)", flush=True)
        Xe, _ = ml.build_features(panel, H, origins[H], effect_end)
        feats[H] = Xe

    shape = {H: feats[H].shape[:2] for H in Hs}
    no_sale = {H: feats[H][..., names.index("age_days")] < 0 for H in Hs}
    forecasts["LGBM-Q"] = {H: ml.predict_lgbm(lgb_models[H], feats[H], shape[H], no_sale[H]) for H in Hs}

    # LGBM-Poisson + NB: tinh chỉnh (kappa, w) trên val
    if split == "val":
        best = (np.inf, 1.0, 0.5)
        for kappa in CONFIG["tune_kappa"]:
            for w in CONFIG["tune_w_series"]:
                fc = {H: _poisson_nb_forecast(panel, H, origins[H], lgb_models[H], effect_end, kappa, w) for H in Hs}
                sc = avg_pinball(panel, Hs, origins, fc)
                tuning_logs.append(pd.DataFrame([{"model": "LGBM-PoisNB", "stage": 2, "params": "{}", "kappa": kappa, "w_series": w, "pinball_avg": sc}]))
                if sc < best[0]:
                    best = (sc, kappa, w)
        hp["LGBM-PoisNB"] = {"kappa": best[1], "w_series": best[2], "pinball_avg": best[0]}
        # MLP-NB: chọn kappa
        bestm = (np.inf, 1.0)
        for kappa in CONFIG["tune_kappa"]:
            fc = {H: ml.predict_mlp_nb(mlp_models[H], feats[H], shape[H], no_sale[H], kappa) for H in Hs}
            sc = avg_pinball(panel, Hs, origins, fc)
            tuning_logs.append(pd.DataFrame([{"model": "MLP-NB", "stage": 2, "params": "{}", "kappa": kappa, "w_series": np.nan, "pinball_avg": sc}]))
            if sc < bestm[0]:
                bestm = (sc, kappa)
        hp["MLP-NB"] = {"kappa": bestm[1], "pinball_avg": bestm[0], "epochs": {str(H): mlp_models[H]["best_epoch"] for H in Hs}}
    forecasts["LGBM-PoisNB"] = {H: _poisson_nb_forecast(panel, H, origins[H], lgb_models[H], effect_end, hp["LGBM-PoisNB"]["kappa"],
                                                       hp["LGBM-PoisNB"]["w_series"]) for H in Hs}
    forecasts["MLP-NB"] = {H: ml.predict_mlp_nb(mlp_models[H], feats[H], shape[H], no_sale[H], hp["MLP-NB"]["kappa"]) for H in Hs}

    # ---- đánh giá + lưu
    evals = [evaluate_forecast(m, H, panel, origins[H], forecasts[m][H]) for m in forecasts for H in Hs]
    ev = pd.concat(evals, ignore_index=True)
    ev.to_csv(ROOT / "results" / f"forecast_eval_{split}_all.csv", index=False)
    (ROOT / "results" / "forecasts").mkdir(parents=True, exist_ok=True)
    arrays = {"ids": panel.ids, "demand_type": panel.demand_type, "speed": panel.speed}
    for H in Hs:
        arrays[f"origins_H{H}"] = origins[H]
        arrays[f"actual_H{H}"] = window_targets(panel.Y, H)[:, origins[H]]
        # ngày cuối của từng chuỗi nhu cầu để mô phỏng tồn kho: lưu thêm Y của split
    for m in forecasts:
        for H in Hs:
            arrays[f"{m}|H{H}|Q"] = forecasts[m][H]["Q"]
            arrays[f"{m}|H{H}|mean"] = forecasts[m][H]["mean"]
    arrays["quantile_grid"] = np.array(QGRID)
    np.savez_compressed(ROOT / "results" / "forecasts" / f"{split}.npz", **arrays)
    if split == "val":
        HPARAMS_PATH.write_text(json.dumps({**hp, "_meta": {"seed": CONFIG["seed"], "horizons": Hs, "quantile_grid": QGRID,
                                                          "tuning_criterion": "pinball trung bình mọi q và mọi H trên val"}},
                                           indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        pd.concat(tuning_logs, ignore_index=True).to_csv(ROOT / "results" / "tuning_log.csv", index=False)
    timing["total"] = time.time() - t_start
    (ROOT / "results" / f"timing_{split}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"[{split}] xong trong {timing['total']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    run(parser.parse_args().split)


if __name__ == "__main__":
    main()
