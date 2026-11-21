"""GĐ4: độ nhạy chi phí × lead time, q* theo nhóm, biểu đồ đánh đổi, bảng chính sách và bootstrap theo chuỗi.

    python -m src.policy_analysis            # đọc results/inventory_sim_val.csv + forecasts/val.npz
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.inventory_sim import MODELS, simulate_order_up_to
from src.prob_models import CONFIG, QGRID, load_panel, pinball_loss, split_bounds
from src.utils import ROOT

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
BASE_RATIO = float(CONFIG["stockout_cost_ratio"])
MIN_GROUP_N = 30  # nhóm ít hơn 30 chuỗi → dùng chính sách của nhóm speed tương ứng
COLORS = {"WindowAverage": "#7f7f7f", "SeasonalNaive": "#8c564b", "CrostonSBA": "#bcbd22", "TSB": "#2ca02c",
          "LGBM-Q": "#1f77b4", "LGBM-PoisNB": "#17becf", "MLP-NB": "#d62728"}
STYLES = {"WindowAverage": "-", "SeasonalNaive": "-", "CrostonSBA": "--", "TSB": "--", "LGBM-Q": "-", "LGBM-PoisNB": ":", "MLP-NB": "-"}
SPEED_VI = {"fast": "nhanh", "medium": "trung bình", "slow": "chậm", "overall": "tất cả"}


def newsvendor_q(ratio: float) -> float:
    """Mức phục vụ tới hạn ratio/(1+ratio) rồi chọn mức q gần nhất trên lưới (quy tắc không tinh chỉnh)."""
    target = ratio / (1.0 + ratio)
    return float(QGRID[int(np.argmin(np.abs(np.array(QGRID) - target)))])


def q_star_table(sim: pd.DataFrame) -> pd.DataFrame:
    """q* tối thiểu hóa tổng chi phí cho từng (mô hình, L, cost_ratio, nhóm), kèm chi phí chính sách điểm."""
    quant = sim[sim.policy == "quantile"]
    idx = quant.groupby(["model", "L", "cost_ratio", "group_type", "group"]).total_cost.idxmin()
    best = quant.loc[idx].rename(columns={"q": "q_star", "total_cost": "cost_qstar", "fill_rate": "fill_qstar", "avg_inventory": "inv_qstar",
                                           "stockout_days": "stockout_days_qstar"})
    pt = sim[sim.policy == "point"][["model", "L", "cost_ratio", "group_type", "group", "total_cost", "fill_rate", "avg_inventory"]]
    pt = pt.rename(columns={"total_cost": "cost_point", "fill_rate": "fill_point", "avg_inventory": "inv_point"})
    out = best.merge(pt, on=["model", "L", "cost_ratio", "group_type", "group"])
    return out[["model", "L", "cost_ratio", "group_type", "group", "n_series", "q_star", "cost_qstar", "fill_qstar", "inv_qstar",
                "stockout_days_qstar", "cost_point", "fill_point", "inv_point"]].reset_index(drop=True)


def policy_table(sim: pd.DataFrame) -> pd.DataFrame:
    """Chính sách đề xuất theo (L, nhóm speed×demand_type) tại cost_ratio cơ sở: (mô hình, q) tối thiểu hóa tổng chi phí.
    Nhóm < MIN_GROUP_N chuỗi dùng chính sách của nhóm speed tương ứng (tránh chọn theo nhiễu)."""
    base = sim[(sim.cost_ratio == BASE_RATIO) & (sim.policy == "quantile")]
    rows = []
    for L in sorted(base.L.unique()):
        b = base[base.L == L]

        def pick(gt, g):
            s = b[(b.group_type == gt) & (b.group == g)]
            return s.loc[s.total_cost.idxmin()]

        groups = sorted(b[b.group_type == "demand_type_x_speed"].group.unique())
        for g in groups:
            best = pick("demand_type_x_speed", g)
            fallback = ""
            if best.n_series < MIN_GROUP_N:
                speed = g.split("_")[-1]
                best = pick("speed", speed)
                fallback = f"dùng nhóm speed={speed}"
            rows.append({"L": L, "group": g, "n_series_group": int(b[(b.group_type == "demand_type_x_speed") & (b.group == g)].n_series.iloc[0]),
                         "model": best.model, "q_star": best.q, "fill_rate": best.fill_rate, "stockout_days": best.stockout_days,
                         "avg_inventory": best.avg_inventory, "total_cost": best.total_cost, "ghi_chu": fallback})
    return pd.DataFrame(rows)


def equal_fill_inventory(sim: pd.DataFrame, model: str, L: int, gt: str, g: str) -> dict:
    """Tồn kho trung bình mà chính sách phân vị cần để ĐẠT CÙNG fill rate với chính sách điểm (nội suy tuyến tính trên đường q)."""
    s = sim[(sim.model == model) & (sim.L == L) & (sim.group_type == gt) & (sim.group == g) & (sim.cost_ratio == BASE_RATIO)]
    q = s[s.policy == "quantile"].sort_values("q")
    p = s[s.policy == "point"].iloc[0]
    fill, inv = q.fill_rate.to_numpy(), q.avg_inventory.to_numpy()
    if not (fill.min() <= p.fill_rate <= fill.max()):
        return {"inv_quantile_same_fill": np.nan, "inv_point": p.avg_inventory, "fill_point": p.fill_rate}
    return {"inv_quantile_same_fill": float(np.interp(p.fill_rate, fill, inv)), "inv_point": float(p.avg_inventory), "fill_point": float(p.fill_rate)}


# ----------------------------------------------------------------------------- biểu đồ
def _save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    return f"results/figures/{name}"


def plot_tradeoff(sim: pd.DataFrame, L: int = 2) -> str:
    """Hình 1: đường đánh đổi fill rate – tồn kho trung bình theo q, tách nhanh/trung bình/chậm; chấm vuông = chính sách điểm."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, sp in zip(axes, ["fast", "medium", "slow"]):
        for m in MODELS:
            s = sim[(sim.model == m) & (sim.L == L) & (sim.group_type == "speed") & (sim.group == sp) & (sim.cost_ratio == BASE_RATIO)]
            q = s[s.policy == "quantile"].sort_values("q")
            ax.plot(q.avg_inventory, q.fill_rate, STYLES[m], color=COLORS[m], marker="o", ms=3, label=m)
            p = s[s.policy == "point"].iloc[0]
            ax.scatter([p.avg_inventory], [p.fill_rate], marker="s", s=45, color=COLORS[m], edgecolor="black", zorder=5)
        ax.set(title=f"Hàng bán {SPEED_VI[sp]} (L={L})", xlabel="Tồn kho cuối ngày trung bình (đơn vị/chuỗi)", ylabel="Fill rate")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="Đường: q từ 0.5 đến 0.99\nÔ vuông: dự báo điểm", title_fontsize=8)
    fig.suptitle("Hình 1. Đánh đổi fill rate – tồn kho theo mức phân vị q, tách theo tốc độ bán")
    return _save(fig, "policy_tradeoff_fill_vs_inventory.png")


def plot_cost_by_q(sim: pd.DataFrame, L: int = 2, ratio: float = BASE_RATIO) -> str:
    """Hình 2: tổng chi phí theo q cho từng mô hình (tất cả và theo speed); sao = q*; đường ngang chấm = chính sách điểm."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (gt, g) in zip(axes.ravel(), [("overall", "overall"), ("speed", "fast"), ("speed", "medium"), ("speed", "slow")]):
        for m in MODELS:
            s = sim[(sim.model == m) & (sim.L == L) & (sim.group_type == gt) & (sim.group == g) & (sim.cost_ratio == ratio)]
            q = s[s.policy == "quantile"].sort_values("q")
            ax.plot(q.q, q.total_cost, STYLES[m], color=COLORS[m], marker="o", ms=3, label=m)
            ax.scatter([q.q.iloc[int(q.total_cost.argmin())]], [q.total_cost.min()], marker="*", s=130, color=COLORS[m], edgecolor="black", zorder=5)
            ax.axhline(s[s.policy == "point"].total_cost.iloc[0], color=COLORS[m], ls=":", lw=0.8, alpha=0.6)
        ax.set(title=f"Nhóm: {SPEED_VI[g]}", xlabel="Mức phân vị q", ylabel="Tổng chi phí TB/chuỗi (28 ngày, đơn vị tương đối)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, title="Sao = q*; đường chấm ngang = dự báo điểm", title_fontsize=8)
    fig.suptitle(f"Hình 2. Tổng chi phí theo q của từng mô hình (L={L}, chi phí thiếu/giữ = {ratio:g})")
    return _save(fig, "policy_total_cost_by_q.png")


def plot_qstar_heatmap(qs: pd.DataFrame, L: int = 2) -> str:
    """Hình 3: heatmap q* theo cost_ratio × nhóm hàng (speed × demand_type), mỗi mô hình một ô."""
    ratios = [float(r) for r in CONFIG["cost_ratios"]]
    groups = sorted(qs[qs.group_type == "demand_type_x_speed"].group.unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=True)
    im = None
    for ax, m in zip(axes.ravel(), MODELS):
        s = qs[(qs.model == m) & (qs.L == L) & (qs.group_type == "demand_type_x_speed")]
        mat = s.pivot(index="group", columns="cost_ratio", values="q_star").reindex(index=groups, columns=ratios)
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=min(QGRID), vmax=max(QGRID))
        ax.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
        ax.set_yticks(range(len(groups)), groups, fontsize=8)
        for i in range(len(groups)):
            for j in range(len(ratios)):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        ax.set(title=m, xlabel="Chi phí thiếu / chi phí giữ")
    axes.ravel()[-1].axis("off")
    fig.colorbar(im, ax=axes.ravel()[-1], label="q* (mức phân vị tối ưu)", fraction=0.6)
    fig.suptitle(f"Hình 3. q* tối thiểu hóa tổng chi phí theo tỷ lệ chi phí và nhóm hàng (L={L})")
    return _save(fig, "policy_qstar_heatmap.png")


def plot_errors(ev: pd.DataFrame, H: int = 3) -> str:
    """Hình 4: pinball trung bình và WAPE theo speed và demand_type cho từng mô hình."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = [("pinball_avg", "speed", "Pinball trung bình (mọi q)"), ("pinball_avg", "demand_type", "Pinball trung bình (mọi q)"),
             ("wape", "speed", "WAPE"), ("wape", "demand_type", "WAPE")]
    for ax, (metric, gt, ylab) in zip(axes.ravel(), specs):
        t = ev[(ev.H == H) & (ev.group_type == gt) & (ev.metric == metric)].pivot(index="group", columns="model", values="value")[MODELS]
        if gt == "speed":
            t = t.loc[["fast", "medium", "slow"]]
            t.index = [SPEED_VI[i] for i in t.index]
        t.plot.bar(ax=ax, width=0.85, color=[COLORS[m] for m in MODELS], legend=False)
        ax.set(xlabel="Nhóm tốc độ bán" if gt == "speed" else "Loại nhu cầu (demand_type)", ylabel=ylab)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Hình 4. Sai số dự báo cửa sổ H={H} theo nhóm hàng (validation)")
    return _save(fig, "policy_forecast_error_by_group.png")


# ----------------------------------------------------------------------------- bootstrap theo chuỗi
def per_series_sim(arr, panel, model: str, L: int, policy: str, q: float | None, ratio: float, split: str = "val") -> np.ndarray:
    """Tổng chi phí mô phỏng của TỪNG chuỗi (giữ + thiếu) cho một kịch bản."""
    H = L + int(CONFIG["review_period"])
    start, end = split_bounds(panel, split)
    demand = panel.Y[:, start:end + 1]
    if policy == "point":
        lv = np.ceil(arr[f"{model}|H{H}|mean"])
    else:
        lv = np.ceil(arr[f"{model}|H{H}|Q"][..., QGRID.index(q)])
    res = simulate_order_up_to(demand, lv[None], L)
    return float(CONFIG["holding_cost"]) * (res["hold_units"][0] + ratio * res["lost"][0])


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int | None = None) -> tuple[float, float, float]:
    """Trung bình và KTC 95% (percentile) của trung bình chênh lệch theo chuỗi, bootstrap có hoàn lại theo chuỗi."""
    rng = np.random.default_rng(int(CONFIG["seed"]) if seed is None else seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def research_question_stats(sim: pd.DataFrame, panel, arr, split: str = "val") -> pd.DataFrame:
    """Bootstrap theo chuỗi cho 3 câu hỏi: (1) phân vị newsvendor vs điểm; (3) MLP-NB vs TSB/Croston (chi phí, pinball)."""
    rows = []
    speeds = {"overall": np.ones(panel.n_series, bool), "fast": panel.speed == "fast", "medium": panel.speed == "medium", "slow": panel.speed == "slow"}
    for L in [int(x) for x in CONFIG["lead_times"]]:
        for ratio in [float(r) for r in CONFIG["cost_ratios"]]:
            qn = newsvendor_q(ratio)
            for model in MODELS:
                a = per_series_sim(arr, panel, model, L, "quantile", qn, ratio, split)
                b = per_series_sim(arr, panel, model, L, "point", None, ratio, split)
                for g, m in speeds.items():
                    mean, lo, hi = paired_bootstrap((a - b)[m])
                    rows.append({"question": "Q1: phan vi q_newsvendor - diem", "model": model, "L": L, "cost_ratio": ratio, "q": qn, "group": g,
                                 "mean_diff_cost": mean, "ci_lo": lo, "ci_hi": hi, "n": int(m.sum())})
    for L in [int(x) for x in CONFIG["lead_times"]]:
        for ratio in [float(r) for r in CONFIG["cost_ratios"]]:
            qn = newsvendor_q(ratio)
            ref = {m: per_series_sim(arr, panel, m, L, "quantile", qn, ratio, split) for m in ("TSB", "CrostonSBA", "MLP-NB", "LGBM-Q")}
            for other in ("TSB", "CrostonSBA", "LGBM-Q"):
                for g, m in speeds.items():
                    mean, lo, hi = paired_bootstrap((ref["MLP-NB"] - ref[other])[m])
                    rows.append({"question": f"Q3: MLP-NB - {other} (cung q_newsvendor)", "model": "MLP-NB", "L": L, "cost_ratio": ratio,
                                 "q": qn, "group": g, "mean_diff_cost": mean, "ci_lo": lo, "ci_hi": hi, "n": int(m.sum())})
    # pinball theo chuỗi: MLP-NB vs TSB/Croston (H=3)
    actual = arr["actual_H3"]
    pin = {}
    for m in ("MLP-NB", "TSB", "CrostonSBA", "LGBM-Q"):
        Q = arr[f"{m}|H3|Q"]
        pin[m] = np.mean([pinball_loss(actual, Q[..., k], q).mean(axis=1) for k, q in enumerate(QGRID)], axis=0)
    for other in ("TSB", "CrostonSBA", "LGBM-Q"):
        for g, m in speeds.items():
            mean, lo, hi = paired_bootstrap((pin["MLP-NB"] - pin[other])[m])
            rows.append({"question": f"Q3: pinball_avg MLP-NB - {other} (H=3)", "model": "MLP-NB", "L": 2, "cost_ratio": np.nan, "q": np.nan, "group": g,
                         "mean_diff_cost": mean, "ci_lo": lo, "ci_hi": hi, "n": int(m.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    sim = pd.read_csv(ROOT / "results" / "inventory_sim_val.csv")
    ev = pd.read_csv(ROOT / "results" / "forecast_eval_val_all.csv")
    qs = q_star_table(sim)
    qs.to_csv(ROOT / "results" / "q_star_val.csv", index=False)
    pol = policy_table(sim)
    pol.to_csv(ROOT / "results" / "policy_table_val.csv", index=False)
    eq = pd.DataFrame([{"model": m, "L": L, "group": g, **equal_fill_inventory(sim, m, L, gt, g)}
                       for m in MODELS for L in (2, 3) for gt, g in [("overall", "overall"), ("speed", "fast"), ("speed", "medium"), ("speed", "slow")]])
    eq.to_csv(TAB / "equal_fill_inventory_val.csv", index=False)
    panel = load_panel()
    arr = np.load(ROOT / "results" / "forecasts" / "val.npz", allow_pickle=True)
    rq = research_question_stats(sim, panel, arr)
    rq.to_csv(ROOT / "results" / "research_questions_val.csv", index=False)
    paths = [plot_tradeoff(sim), plot_cost_by_q(sim), plot_qstar_heatmap(qs), plot_errors(ev)]
    print("\n".join(paths))


if __name__ == "__main__":
    main()
