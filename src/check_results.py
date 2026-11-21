"""Kiểm tra nhanh results/inventory_sim_*.csv: định danh, gộp nhóm, tính đơn điệu và vài số tham chiếu.
Chạy:  python check_results.py [đường_dẫn_csv]   (mặc định results/inventory_sim_val.csv)"""
import sys

import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results/inventory_sim_val.csv"
df = pd.read_csv(path)
o = df[df.group_type == "overall"]
fails = 0


def check(name, cond, detail=""):
    global fails
    fails += 0 if cond else 1
    print(("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail else ""))


KEY = ["model", "policy", "q", "L", "cost_ratio"]
check("mọi dòng overall có n_series = 823", (o.n_series == 823).all())
check("tổng chi phí = giữ + thiếu", (o.total_cost - o.holding_cost - o.stockout_cost).abs().max() < 1e-9)
check("chi phí thiếu = tỷ lệ x đơn vị mất", (o.stockout_cost - o.cost_ratio * o.lost_units).abs().max() < 1e-9)
check("chi phí giữ = 28 x tồn kho TB", (o.holding_cost - 28 * o.avg_inventory).abs().max() < 1e-9)
check("fill rate trong [0,1], chi phí >= 0",
      o.fill_rate.between(0, 1).all() and (o[["holding_cost", "stockout_cost", "total_cost"]] >= 0).all().all())

ov = o.set_index(KEY).total_cost
for gt in ["speed", "demand_type", "demand_type_x_speed"]:
    a = df[df.group_type == gt]
    s = a.assign(c=a.total_cost * a.n_series).groupby(KEY, dropna=False)[["c", "n_series"]].sum()
    diff = (s.c / s.n_series - ov.reindex(s.index)).abs().max()
    check(f"gộp theo {gt} (có trọng số n_series) = overall", diff < 1e-9, f"lệch tối đa {diff:.2e}")

q = o[o.policy == "quantile"]
bad = [k for k, g in q.groupby(["model", "L", "cost_ratio"])
       if (g.sort_values("q").fill_rate.diff().dropna() < -1e-12).any()
       or (g.sort_values("q").avg_inventory.diff().dropna() < -1e-12).any()]
check("fill rate và tồn kho không giảm khi q tăng", not bad, f"vi phạm: {bad[:3]}" if bad else "")

# số tham chiếu (val, L=2, tỷ lệ 10) theo báo cáo sau GĐ7
REF = [("WindowAverage", "quantile", 0.8, 271.4933), ("WindowAverage", "point", np.nan, 291.1191),
       ("MLP-NB", "quantile", 0.8, 252.4508), ("MLP-NB", "point", np.nan, 267.0243),
       ("LGBM-Q", "quantile", 0.8, 254.4751)]
print("\nSố tham chiếu (chỉ có nghĩa với file val):")
for m, pol, qq, exp in REF:
    s = o[(o.model == m) & (o.policy == pol) & (o.L == 2) & (o.cost_ratio == 10)]
    s = s[np.isclose(s.q, qq)] if pol == "quantile" else s
    got = float(s.total_cost.iloc[0]) if len(s) else np.nan
    check(f"{m} {pol} q={qq}: kỳ vọng {exp}", abs(got - exp) < 5e-4, f"thực tế {got:.4f}")

r = q[(q.L == 2) & (q.cost_ratio == 10)]
print("\nq* (tối thiểu tổng chi phí) theo mô hình, L=2, tỷ lệ 10:")
print(r.loc[r.groupby("model").total_cost.idxmin(), ["model", "q", "fill_rate", "total_cost"]].round(4).to_string(index=False))
print(f"\nTổng kết: {'TẤT CẢ PASS' if not fails else str(fails) + ' mục FAIL'}")
