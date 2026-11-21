"""GĐ6: chính sách điểm có hệ số an toàn S = ceil(k × mean cửa sổ) vs chính sách phân vị (lưới mở rộng q ≥ 0.3).

k (lưới 0.5–3.0, bước 0.1) và q* đều chọn trên VAL cho từng (mô hình, cost_ratio, L); đánh giá trên TEST.
    python -m src.safety_point
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.inventory_sim import MODELS, quantile_to_levels, simulate_order_up_to
from src.policy_analysis import paired_bootstrap
from src.prob_models import CONFIG, QGRID, load_panel, split_bounds
from src.utils import ROOT

HOLD = float(CONFIG["holding_cost"])


def k_grid() -> np.ndarray:
    """Lưới hệ số an toàn k từ config (bắt đầu, kết thúc, bước), làm tròn 1 chữ số."""
    a, b, step = CONFIG["safety_k_grid"]
    return np.round(np.arange(a, b + step / 2, step), 1)


def series_cost(res: dict, ratio: float) -> np.ndarray:
    """Tổng chi phí từng chuỗi cho mọi kịch bản: [C, S] = holding × (tồn kho-ngày + ratio × đơn vị thiếu)."""
    return HOLD * (res["hold_units"] + ratio * res["lost"])


def quantile_levels(arr, model: str, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Mức order-up-to [C,S,n] của mọi q (lưới phụ + lưới chính, tăng dần) và mảng q tương ứng."""
    qs = np.concatenate([arr["quantile_grid_ext"], arr["quantile_grid"]]).astype(float)
    Q = np.concatenate([arr[f"{model}|H{H}|Qext"], arr[f"{model}|H{H}|Q"]], axis=-1)
    return np.moveaxis(quantile_to_levels(model, Q), -1, 0), qs


def evaluate(split_arrs: dict, panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (bảng so sánh k vs q* trên test, bảng q* mới trên val với lưới mở rộng)."""
    ks = k_grid()
    ratios = [float(r) for r in CONFIG["cost_ratios"]]
    rows, qrows = [], []
    dem = {s: panel.Y[:, slice(*(lambda b: (b[0], b[1] + 1))(split_bounds(panel, s)))] for s in ("val", "test")}
    for L in [int(x) for x in CONFIG["lead_times"]]:
        H = L + int(CONFIG["review_period"])
        for model in MODELS:
            sim = {}
            for s in ("val", "test"):
                arr = split_arrs[s]
                mean = arr[f"{model}|H{H}|mean"]
                lv_k = np.stack([np.ceil(k * mean) for k in ks])
                lv_q, qs = quantile_levels(arr, model, H)
                lv_main = np.ceil(mean)[None]
                sim[s] = {"k": simulate_order_up_to(dem[s], lv_k, L), "q": simulate_order_up_to(dem[s], lv_q, L),
                          "pt": simulate_order_up_to(dem[s], lv_main, L), "qs": qs}
            for r in ratios:
                ck = series_cost(sim["val"]["k"], r).mean(axis=1)
                cq = series_cost(sim["val"]["q"], r).mean(axis=1)
                ik, iq = int(np.argmin(ck)), int(np.argmin(cq))
                main_mask = np.isin(sim["val"]["qs"], QGRID)
                iq_old = int(np.flatnonzero(main_mask)[np.argmin(cq[main_mask])])
                t_k = series_cost(sim["test"]["k"], r)[ik]
                t_q = series_cost(sim["test"]["q"], r)[iq]
                t_qold = series_cost(sim["test"]["q"], r)[iq_old]
                t_pt = series_cost(sim["test"]["pt"], r)[0]
                row = {"model": model, "L": L, "cost_ratio": r, "k_val": float(ks[ik]), "q_star_val": float(sim["val"]["qs"][iq]),
                       "q_star_val_main_grid": float(sim["val"]["qs"][iq_old]),
                       "cost_val_k": float(ck[ik]), "cost_val_q": float(cq[iq]), "cost_val_point": float(series_cost(sim["val"]["pt"], r)[0].mean()),
                       "cost_test_point": float(t_pt.mean()), "cost_test_k": float(t_k.mean()), "cost_test_q": float(t_q.mean()),
                       "cost_test_q_main_grid": float(t_qold.mean())}
                for name, d in [("k_minus_q", t_k - t_q), ("q_minus_point", t_q - t_pt), ("k_minus_point", t_k - t_pt)]:
                    m, lo, hi = paired_bootstrap(d)
                    row[f"{name}"], row[f"{name}_lo"], row[f"{name}_hi"] = m, lo, hi
                rows.append(row)
                qrows.append({"model": model, "L": L, "cost_ratio": r, "q_star_main_grid": row["q_star_val_main_grid"], "q_star_ext": row["q_star_val"],
                              "cost_val_q_main": float(cq[iq_old]), "cost_val_q_ext": float(cq[iq]), "cost_test_q_main": row["cost_test_q_main_grid"],
                              "cost_test_q_ext": row["cost_test_q"]})
    return pd.DataFrame(rows), pd.DataFrame(qrows)


def _simulate_all(split_arrs: dict, panel, L: int, model: str, ks: np.ndarray) -> dict:
    """Mô phỏng val/test cho: mọi k (điểm an toàn), mọi q (lưới mở rộng + chính), điểm thuần (k=1)."""
    H = L + int(CONFIG["review_period"])
    out = {}
    for s in ("val", "test"):
        arr = split_arrs[s]
        start, end = split_bounds(panel, s)
        demand = panel.Y[:, start:end + 1]
        mean = arr[f"{model}|H{H}|mean"]
        lv_q, qs = quantile_levels(arr, model, H)
        out[s] = {"k": simulate_order_up_to(demand, np.stack([np.ceil(k * mean) for k in ks]), L),
                  "q": simulate_order_up_to(demand, lv_q, L), "pt": simulate_order_up_to(demand, np.ceil(mean)[None], L), "qs": qs}
    return out


def evaluate_by_speed(split_arrs: dict, panel, ratios=(10.0,)) -> pd.DataFrame:
    """So sánh (điểm an toàn k*) và (phân vị q*) TRONG TỪNG nhóm speed: k*, q* chọn trên val theo chi phí TB của chính nhóm đó;
    đánh giá trên test với hiệu chi phí từng chuỗi và KTC bootstrap (ghép cặp, theo chuỗi trong nhóm)."""
    ks = k_grid()
    rows = []
    for L in [int(x) for x in CONFIG["lead_times"]]:
        for model in MODELS:
            sim = _simulate_all(split_arrs, panel, L, model, ks)
            for r in ratios:
                cv_k, cv_q = series_cost(sim["val"]["k"], r), series_cost(sim["val"]["q"], r)
                ct_k, ct_q, ct_pt = series_cost(sim["test"]["k"], r), series_cost(sim["test"]["q"], r), series_cost(sim["test"]["pt"], r)[0]
                for g in ("fast", "medium", "slow"):
                    m = panel.speed == g
                    ik, iq = int(np.argmin(cv_k[:, m].mean(axis=1))), int(np.argmin(cv_q[:, m].mean(axis=1)))
                    tk, tq, tp = ct_k[ik][m], ct_q[iq][m], ct_pt[m]
                    row = {"model": model, "L": L, "cost_ratio": r, "group": g, "n": int(m.sum()), "k_val": float(ks[ik]),
                           "q_star_val": float(sim["val"]["qs"][iq]), "cost_test_point": float(tp.mean()), "cost_test_k": float(tk.mean()),
                           "cost_test_q": float(tq.mean())}
                    for name, d in [("k_minus_q", tk - tq), ("q_minus_point", tq - tp), ("k_minus_point", tk - tp)]:
                        mm, lo, hi = paired_bootstrap(d)
                        row[name], row[f"{name}_lo"], row[f"{name}_hi"] = mm, lo, hi
                    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    panel = load_panel()
    arrs = {s: np.load(ROOT / "results" / "forecasts" / f"{s}.npz", allow_pickle=True) for s in ("val", "test")}
    comp, qnew = evaluate(arrs, panel)
    comp.to_csv(ROOT / "results" / "safety_point_test.csv", index=False)
    qnew.to_csv(ROOT / "results" / "q_star_ext_val.csv", index=False)
    evaluate_by_speed(arrs, panel, ratios=[float(r) for r in CONFIG["cost_ratios"]]).to_csv(ROOT / "results" / "safety_point_by_speed_test.csv", index=False)
    print(comp[(comp.L == 2) & (comp.cost_ratio == 10)].round(3).to_string())


if __name__ == "__main__":
    main()
