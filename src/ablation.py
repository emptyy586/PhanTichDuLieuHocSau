"""GĐ6: ablation biến ngoại sinh cho LightGBM-Q (bỏ giá, SNAP, sự kiện; giữ lịch sử doanh số + thứ/tháng + speed/demand_type).

    python -m src.ablation          # cần results/hparams.json (không dùng tham số mới), ghi results/ablation_exog.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import ml_models as ml
from src.prob_models import QGRID, evaluate_forecast, load_panel, origins_for_split
from src.tuning import horizons
from src.utils import ROOT

EXOG_PREFIXES = ("price", "fut_snap", "fut_event")  # giá, SNAP, sự kiện (thứ/tháng/hiệu ứng thứ giữ lại: là lịch)


def exog_mask(names: list[str]) -> np.ndarray:
    """True cho các cột KHÔNG phải ngoại sinh (giá/SNAP/sự kiện) – tức cột được giữ lại ở bản ablation."""
    return np.array([not n.startswith(EXOG_PREFIXES) for n in names])


def run() -> pd.DataFrame:
    """Huấn luyện LightGBM-Q đầy đủ (đối chứng, cùng cấu hình) và bản bỏ ngoại sinh cho val và test; trả về bảng chỉ số."""
    panel = load_panel()
    rows, timing = [], {}
    for split, last_label in [("val", panel.val_start - 1), ("test", panel.test_start - 1)]:
        for H in horizons():
            X, y, _, names = ml.make_training_set(panel, H, last_label, last_label)
            keep = exog_mask(names)
            org = origins_for_split(panel, split, H)
            Xe, _ = ml.build_features(panel, H, org, last_label)
            no_sale = Xe[..., names.index("age_days")] < 0
            for variant, cols in [("full", np.ones(len(names), bool)), ("no_exog", keep)]:
                t0 = time.time()
                models = ml.fit_lgbm(X[:, cols], y)
                fc = ml.predict_lgbm(models, Xe[..., cols], Xe.shape[:2], no_sale)
                ev = evaluate_forecast(f"LGBM-Q[{variant}]", H, panel, org, fc)
                ev["split"], ev["n_features"] = split, int(cols.sum())
                rows.append(ev)
                timing[f"{split}_H{H}_{variant}"] = time.time() - t0
                print(split, H, variant, int(cols.sum()), f"{timing[f'{split}_H{H}_{variant}']:.1f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(ROOT / "results" / "ablation_exog.csv", index=False)
    return out


if __name__ == "__main__":
    run()
