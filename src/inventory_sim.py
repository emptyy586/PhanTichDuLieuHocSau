"""Mô phỏng chính sách bổ sung order-up-to hằng ngày (lost sales) trên nhu cầu thực của split.

Quy ước thời gian (gốc t0 = ngày cuối của giai đoạn trước; ngày mô phỏng d = 1..D ứng với t0+d):
- Cuối ngày t0+d đặt hàng (d = 0..n-1, n = D-H+1 gốc có dự báo): S = ceil(phân vị q của D_H(t0+d)); đặt max(0, S - vị trí tồn kho),
  vị trí tồn kho = tồn tay + hàng đang về. Hàng về cuối ngày t0+d+L, dùng được từ ngày t0+d+L+1 = t0+d+H.
- Trong ngày: bán min(tồn, nhu cầu); phần thiếu bị MẤT (lost sales). Cuối ngày nhận hàng đến hạn.
- Tồn kho cuối ngày E_d (sau bán và sau nhận hàng) là tồn kho tính chi phí giữ hàng (đơn vị/ngày) và tồn kho trung bình.
- Tồn đầu (cuối ngày t0) = S của gốc đầu tiên (order_up_to_at_start), không có hàng đang về.
Chi phí (tương đối, KHÔNG phải chi phí thực): giữ = holding_cost × tồn cuối ngày; thiếu = holding_cost × cost_ratio × đơn vị thiếu.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.prob_models import CONFIG, QGRID, group_masks, load_panel, split_bounds
from src.utils import ROOT

MODELS = ["WindowAverage", "SeasonalNaive", "CrostonSBA", "TSB", "LGBM-Q", "LGBM-PoisNB", "MLP-NB"]


def simulate_order_up_to(demand: np.ndarray, levels: np.ndarray, lead_time: int, return_trace: bool = False) -> dict:
    """Mô phỏng order-up-to cho C kịch bản song song.

    demand: [S, D] nhu cầu ngày d=1..D; levels: [C, S, n] mức order-up-to S tại các gốc d=0..n-1 (đã làm tròn lên).
    Trả về các mảng [C, S]: sales, lost, hold_units (tổng tồn cuối ngày), stockout_days, orders_total, avg_inventory;
    kèm trace theo ngày nếu return_trace.
    """
    C, S, n = levels.shape
    D = demand.shape[1]
    L = int(lead_time)
    onhand = levels[:, :, 0].astype(float).copy()  # tồn đầu = S tại gốc đầu tiên
    arrivals = np.zeros((C, S, D + L + 2))  # arrivals[..., k]: hàng về cuối ngày k
    sales = np.zeros((C, S))
    lost = np.zeros((C, S))
    hold_units = np.zeros((C, S))
    stockout_days = np.zeros((C, S))
    orders_total = np.zeros((C, S))
    trace = {"start": [], "received": [], "sales": [], "end": [], "order": []} if return_trace else None
    for d in range(1, D + 1):
        y = demand[:, d - 1][None, :]
        start = onhand.copy()
        s = np.minimum(onhand, y)
        onhand = onhand - s
        rec = arrivals[:, :, d]
        onhand = onhand + rec
        sales += s
        lost += y - s
        stockout_days += (y - s > 1e-12)
        hold_units += onhand
        order = np.zeros((C, S))
        if d <= n - 1:
            pipe = arrivals[:, :, d + 1:].sum(axis=2)
            order = np.maximum(0.0, levels[:, :, d] - (onhand + pipe))
            arrivals[:, :, d + L] += order
            orders_total += order
        if return_trace:
            trace["start"].append(start), trace["received"].append(rec.copy()), trace["sales"].append(s)
            trace["end"].append(onhand.copy()), trace["order"].append(order)
    out = {"sales": sales, "lost": lost, "hold_units": hold_units, "stockout_days": stockout_days,
           "orders_total": orders_total, "avg_inventory": hold_units / D}
    if return_trace:
        out["trace"] = {k: np.stack(v) for k, v in trace.items()}
    return out


def aggregate(res: dict, demand: np.ndarray, masks: list, holding_cost: float, cost_ratios: list[float]) -> list[dict]:
    """Gộp chỉ số theo nhóm: fill rate (theo khối lượng: tổng bán / tổng cầu), số ngày hết hàng TB/chuỗi, tồn kho TB/chuỗi,
    chi phí TB/chuỗi (giữ, thiếu, tổng) cho từng cost_ratio. Trả về list dict cho từng (kịch bản, nhóm, cost_ratio)."""
    rows = []
    total_dem = demand.sum(axis=1)
    for gt, g, m in masks:
        dem = total_dem[m].sum()
        sold = res["sales"][:, m].sum(axis=1)
        fill = np.where(dem > 0, sold / max(dem, 1e-12), np.nan)
        stock_days = res["stockout_days"][:, m].mean(axis=1)
        avg_inv = res["avg_inventory"][:, m].mean(axis=1)
        hold_cost = holding_cost * res["hold_units"][:, m].mean(axis=1)
        lost_units = res["lost"][:, m].mean(axis=1)
        for r in cost_ratios:
            stockout_cost = holding_cost * r * lost_units
            for c in range(hold_cost.shape[0]):
                rows.append({"case": c, "group_type": gt, "group": g, "n_series": int(m.sum()), "cost_ratio": r,
                             "fill_rate": float(fill[c]), "stockout_days": float(stock_days[c]), "avg_inventory": float(avg_inv[c]),
                             "lost_units": float(lost_units[c]), "holding_cost": float(hold_cost[c]),
                             "stockout_cost": float(stockout_cost[c]), "total_cost": float(hold_cost[c] + stockout_cost[c])})
    return rows


def run_inventory_sim(split: str, forecast_path=None, out_path=None) -> pd.DataFrame:
    """Chạy mọi mô hình × q × L cộng chính sách điểm trên split; ghi results/inventory_sim_{split}.csv."""
    panel = load_panel()
    arr = np.load(forecast_path or ROOT / "results" / "forecasts" / f"{split}.npz", allow_pickle=True)
    start, end = split_bounds(panel, split)
    demand = panel.Y[:, start:end + 1]
    R = int(CONFIG["review_period"])
    holding = float(CONFIG["holding_cost"])
    ratios = [float(r) for r in CONFIG["cost_ratios"]]
    masks = group_masks(panel)
    rows = []
    for L in [int(x) for x in CONFIG["lead_times"]]:
        H = L + R
        cases, meta = [], []
        for model in MODELS:
            Q = arr[f"{model}|H{H}|Q"]
            for k, q in enumerate(QGRID):
                cases.append(np.ceil(Q[..., k])), meta.append((model, "quantile", q))
            cases.append(np.ceil(arr[f"{model}|H{H}|mean"])), meta.append((model, "point", np.nan))
        res = simulate_order_up_to(demand, np.stack(cases), L)
        for r in aggregate(res, demand, masks, holding, ratios):
            model, policy, q = meta[r.pop("case")]
            rows.append({"model": model, "policy": policy, "q": q, "L": L, "H": H, **r})
    df = pd.DataFrame(rows)
    df.to_csv(out_path or ROOT / "results" / f"inventory_sim_{split}.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    d = run_inventory_sim(ap.parse_args().split)
    print(d.shape)
