"""GĐ6: bảng hiệu chuẩn rời rạc – P(y<=Q) và P(y<Q) cho mọi mô hình, q <= 0.95, H=3,4, val và test.

Mô hình hiệu chuẩn tốt (với phân phối rời rạc) khi q nằm trong [P(y<Q), P(y<=Q)]; báo cả dung sai ±0.05.
    python -m src.calibration
"""
from __future__ import annotations

import pandas as pd

from src.utils import ROOT

TOL = 0.05


def calibration_table(ev: pd.DataFrame, split: str, q_max: float = 0.95) -> pd.DataFrame:
    """Bảng (model,H,split,q,P_le,P_lt,in_bracket,in_bracket_tol,abs_dev_ok) tổng thể từ file đánh giá."""
    o = ev[(ev.group_type == "overall") & ev.metric.isin(["coverage", "coverage_lt"]) & (ev.q <= q_max + 1e-9)]
    t = o.pivot_table(index=["model", "H", "q"], columns="metric", values="value").reset_index()
    t = t.rename(columns={"coverage": "P_le", "coverage_lt": "P_lt"})
    t["split"] = split
    t["in_bracket"] = (t.P_lt <= t.q + 1e-12) & (t.q <= t.P_le + 1e-12)
    t["in_bracket_tol"] = (t.P_lt - TOL <= t.q) & (t.q <= t.P_le + TOL)
    t["abs_dev_ok"] = (t.P_le - t.q).abs() <= TOL
    return t


def summarize(tab: pd.DataFrame) -> pd.DataFrame:
    """Số mức q (trên tổng) thỏa từng tiêu chí, theo (split, H, model)."""
    g = tab.groupby(["split", "H", "model"])
    return g.agg(n_q=("q", "size"), in_bracket=("in_bracket", "sum"), in_bracket_tol=("in_bracket_tol", "sum"),
                 abs_dev_ok=("abs_dev_ok", "sum")).reset_index()


def main() -> None:
    tabs = [calibration_table(pd.read_csv(ROOT / "results" / f"forecast_eval_{s}_all.csv"), s) for s in ("val", "test")]
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(ROOT / "results" / "calibration_table.csv", index=False)
    summ = summarize(tab)
    summ.to_csv(ROOT / "results" / "calibration_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
