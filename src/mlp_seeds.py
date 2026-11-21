"""GĐ6: độ ổn định của MLP-NB theo seed (42, 43, 44) và kiểm tra hội tụ (đường NLL holdout theo epoch).

Mỗi seed: huấn luyện trên train (chọn epoch bằng holdout nội bộ), chọn kappa trên val (chung H=3,4), đánh giá val;
sau đó huấn luyện lại trên train+val với đúng số epoch/kappa đó và đánh giá test.
    python -m src.mlp_seeds
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import ml_models as ml
from src.prob_models import CONFIG, avg_pinball, evaluate_forecast, load_panel, origins_for_split
from src.tuning import horizons
from src.utils import ROOT


def run() -> None:
    """Chạy 3 seed × H ∈ {3,4} cho val và test; ghi results/mlp_seeds.csv, results/mlp_nll_curves.csv."""
    panel = load_panel()
    Hs = horizons()
    seeds = [int(s) for s in CONFIG["mlp_seeds"]]
    data = {}
    for split, last in [("val", panel.val_start - 1), ("test", panel.test_start - 1)]:
        for H in Hs:
            X, y, org_row, names = ml.make_training_set(panel, H, last, last)
            org = origins_for_split(panel, split, H)
            Xe, _ = ml.build_features(panel, H, org, last)
            data[(split, H)] = {"train": (X, y, org_row), "names": names, "org": org, "Xe": Xe, "no_sale": Xe[..., names.index("age_days")] < 0}
    rows, curves = [], []
    for seed in seeds:
        t0 = time.time()
        models = {H: ml.fit_mlp_nb(*data[("val", H)]["train"], data[("val", H)]["names"], H, seed=seed) for H in Hs}
        for H in Hs:
            curves += [{"seed": seed, "H": H, "max_epochs": int(CONFIG["mlp_max_epochs"]), "epoch": e + 1, "holdout_nll": v}
                       for e, v in enumerate(models[H]["holdout_nll"])]
        d = {H: data[("val", H)] for H in Hs}
        best = (np.inf, 1.0)
        for kappa in CONFIG["tune_kappa"]:
            fc = {H: ml.predict_mlp_nb(models[H], d[H]["Xe"], d[H]["Xe"].shape[:2], d[H]["no_sale"], kappa) for H in Hs}
            sc = avg_pinball(panel, Hs, {H: d[H]["org"] for H in Hs}, fc)
            if sc < best[1 - 1]:
                best = (sc, kappa)
        kappa = best[1]
        for H in Hs:
            fc = ml.predict_mlp_nb(models[H], d[H]["Xe"], d[H]["Xe"].shape[:2], d[H]["no_sale"], kappa)
            ev = evaluate_forecast("MLP-NB", H, panel, d[H]["org"], fc)
            ov = ev[(ev.group_type == "overall") & ev.metric.isin(["pinball_avg", "wape", "mase"])].set_index("metric").value
            rows.append({"seed": seed, "H": H, "split": "val", "kappa": kappa, "epochs": models[H]["best_epoch"], **ov.to_dict()})
        for H in Hs:  # test: huấn luyện lại trên train+val với epoch đã chốt
            dt = data[("test", H)]
            m = ml.fit_mlp_nb(*dt["train"], dt["names"], H, fixed_epochs=models[H]["best_epoch"], seed=seed)
            fc = ml.predict_mlp_nb(m, dt["Xe"], dt["Xe"].shape[:2], dt["no_sale"], kappa)
            ev = evaluate_forecast("MLP-NB", H, panel, dt["org"], fc)
            ov = ev[(ev.group_type == "overall") & ev.metric.isin(["pinball_avg", "wape", "mase"])].set_index("metric").value
            rows.append({"seed": seed, "H": H, "split": "test", "kappa": kappa, "epochs": models[H]["best_epoch"], **ov.to_dict()})
        print(f"seed {seed} xong {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "mlp_seeds.csv", index=False)
    # hội tụ: seed 42, tối đa `mlp_convergence_epochs` epoch trên tập train (val)
    long_ep = int(CONFIG["mlp_convergence_epochs"])
    for H in Hs:
        m = ml.fit_mlp_nb(*data[("val", H)]["train"], data[("val", H)]["names"], H, seed=seeds[0], max_epochs=long_ep)
        curves += [{"seed": seeds[0], "H": H, "max_epochs": long_ep, "epoch": e + 1, "holdout_nll": v} for e, v in enumerate(m["holdout_nll"])]
        d = data[("val", H)]
        kappa = float(CONFIG["tune_kappa"][3])
        print(f"H={H}: best_epoch (max {long_ep}) = {m['best_epoch']}", flush=True)
    pd.DataFrame(curves).to_csv(ROOT / "results" / "mlp_nll_curves.csv", index=False)


if __name__ == "__main__":
    run()
