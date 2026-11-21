"""GĐ5: đánh giá cuối trên TEST với mô hình/q/chính sách ĐÃ CHỐT bằng validation (không chỉnh theo test).

    python -m src.final_eval        # cần results/forecasts/test.npz và results/inventory_sim_test.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.inventory_sim import MODELS, aggregate, simulate_order_up_to
from src.policy_analysis import BASE_RATIO, paired_bootstrap
from src.prob_models import CONFIG, QGRID, group_masks, load_panel, split_bounds
from src.utils import ROOT

HOLD = float(CONFIG["holding_cost"])


def group_policy_levels(arr, panel, pol_L: pd.DataFrame, H: int) -> np.ndarray:
    """Mức order-up-to [1,S,n] khi mỗi chuỗi dùng (mô hình, q) của nhóm speed×demand_type theo bảng chính sách."""
    n = arr[f"origins_H{H}"].shape[0]
    levels = np.zeros((1, panel.n_series, n))
    gds = panel.group_ds
    for _, r in pol_L.iterrows():
        m = gds == r.group
        levels[0, m] = np.ceil(arr[f"{r.model}|H{H}|Q"][m][..., QGRID.index(float(r.q_star))])
    return levels


def case_levels(arr, model: str, H: int, policy: str, q: float | None) -> np.ndarray:
    """Mức order-up-to [1,S,n] cho (mô hình, quantile q) hoặc chính sách điểm."""
    if policy == "point":
        return np.ceil(arr[f"{model}|H{H}|mean"])[None]
    return np.ceil(arr[f"{model}|H{H}|Q"][..., QGRID.index(float(q))])[None]


def policy_comparison(panel, arrs: dict, qs: pd.DataFrame, pol: pd.DataFrame, ratio: float = BASE_RATIO) -> tuple[pd.DataFrame, dict]:
    """So sánh trên val và test ba chính sách chốt từ val (L ∈ lead_times, cost_ratio=ratio):
    (A) chính sách theo nhóm; (B) mô hình+q tốt nhất toàn cục; (C) chính sách điểm tốt nhất. Trả về bảng và chi phí từng chuỗi."""
    rows, per_series = [], {}
    masks = group_masks(panel)
    for L in [int(x) for x in CONFIG["lead_times"]]:
        H = L + int(CONFIG["review_period"])
        ov = qs[(qs.L == L) & (qs.cost_ratio == ratio) & (qs.group_type == "overall")].sort_values("cost_qstar").iloc[0]
        pt = qs[(qs.L == L) & (qs.cost_ratio == ratio) & (qs.group_type == "overall")].sort_values("cost_point").iloc[0]
        chosen = {"A_theo_nhom": ("group", None), f"B_toan_cuc ({ov.model}, q={ov.q_star:g})": ("model_q", (ov.model, ov.q_star)),
                  f"C_diem ({pt.model})": ("point", pt.model)}
        for split, arr in arrs.items():
            start, end = split_bounds(panel, split)
            demand = panel.Y[:, start:end + 1]
            for name, (kind, spec) in chosen.items():
                if kind == "group":
                    lv = group_policy_levels(arr, panel, pol[pol.L == L], H)
                elif kind == "model_q":
                    lv = case_levels(arr, spec[0], H, "quantile", spec[1])
                else:
                    lv = case_levels(arr, spec, H, "point", None)
                res = simulate_order_up_to(demand, lv, L)
                per_series[(L, split, name)] = HOLD * (res["hold_units"][0] + ratio * res["lost"][0])
                for r in aggregate(res, demand, masks, HOLD, [ratio]):
                    r.pop("case")
                    rows.append({"L": L, "policy": name, "split": split, **r})
    return pd.DataFrame(rows), per_series


def transfer_table(qs: pd.DataFrame, sim_test: pd.DataFrame) -> pd.DataFrame:
    """Với q*(val) của từng (mô hình, L, cost_ratio) tổng thể: chi phí/fill rate trên test so với chính sách điểm trên test."""
    ov = qs[(qs.group_type == "overall")]
    t = sim_test[(sim_test.group_type == "overall")]
    rows = []
    for _, r in ov.iterrows():
        a = t[(t.model == r.model) & (t.L == r.L) & (t.cost_ratio == r.cost_ratio) & (t.policy == "quantile") & (np.isclose(t.q, r.q_star))].iloc[0]
        p = t[(t.model == r.model) & (t.L == r.L) & (t.cost_ratio == r.cost_ratio) & (t.policy == "point")].iloc[0]
        rows.append({"model": r.model, "L": r.L, "cost_ratio": r.cost_ratio, "q_star_val": r.q_star, "cost_val": r.cost_qstar,
                     "fill_val": r.fill_qstar, "cost_test": a.total_cost, "fill_test": a.fill_rate, "inv_test": a.avg_inventory,
                     "cost_point_test": p.total_cost, "fill_point_test": p.fill_rate,
                     "gain_vs_point_val": r.cost_qstar / r.cost_point - 1, "gain_vs_point_test": a.total_cost / p.total_cost - 1})
    return pd.DataFrame(rows)


def main() -> None:
    panel = load_panel()
    qs = pd.read_csv(ROOT / "results" / "q_star_val.csv")
    pol = pd.read_csv(ROOT / "results" / "policy_table_val.csv")
    sim_test = pd.read_csv(ROOT / "results" / "inventory_sim_test.csv")
    arrs = {s: np.load(ROOT / "results" / "forecasts" / f"{s}.npz", allow_pickle=True) for s in ("val", "test")}
    comp, per = policy_comparison(panel, arrs, qs, pol)
    comp.to_csv(ROOT / "results" / "policy_eval_val_test.csv", index=False)
    transfer_table(qs, sim_test).to_csv(ROOT / "results" / "test_qstar_transfer.csv", index=False)
    boot = []
    for L in [int(x) for x in CONFIG["lead_times"]]:
        names = sorted({k[2] for k in per if k[0] == L})
        a = [n for n in names if n.startswith("A_")][0]
        for other in [n for n in names if n != a]:
            for split in ("val", "test"):
                mean, lo, hi = paired_bootstrap(per[(L, split, a)] - per[(L, split, other)])
                boot.append({"L": L, "split": split, "A_minus": other, "mean_diff": mean, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(boot).to_csv(ROOT / "results" / "policy_eval_bootstrap.csv", index=False)
    print(comp[(comp.group_type == "overall")].round(4).to_string())


if __name__ == "__main__":
    main()
