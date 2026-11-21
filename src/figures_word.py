"""GĐ9: vẽ lại hình để đưa vào báo cáo Word (không đổi số liệu; chỉ đọc CSV/npz có sẵn).

Quy cách: rộng 6.3 inch, dpi 300, chữ >= 8pt (số trong ô heatmap >= 7pt), bỏ tiền tố số hình trong tiêu đề, tiêu đề ghi tập dữ liệu/tham số.
    python -m src.figures_word
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.policy_analysis import COLORS, STYLES, q_star_table
from src.prob_models import CONFIG, QGRID
from src.utils import ROOT

FIG = ROOT / "results" / "figures"
W = 6.3
DPI = 300
RATIO = float(CONFIG["stockout_cost_ratio"])
L0 = 2
ALL = ["WindowAverage", "SeasonalNaive", "CrostonSBA", "TSB", "LGBM-Q", "LGBM-PoisNB", "MLP-NB", "DeepAR"]
SPEED_VI = {"fast": "nhanh", "medium": "trung bình", "slow": "chậm", "overall": "tất cả"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "figure.titlesize": 9, "axes.grid": True, "grid.alpha": 0.3})


def _sim(split: str) -> pd.DataFrame:
    """Mô phỏng lưới chính, 8 mô hình (7 mô hình cũ + DeepAR)."""
    return pd.concat([pd.read_csv(ROOT / "results" / f"inventory_sim_{split}.csv"), pd.read_csv(ROOT / "results" / f"inventory_sim_deepar_{split}.csv")], ignore_index=True)


def _save(fig, name: str) -> str:
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)
    return name


def _legend_below(fig, handles, labels, ncol=4, y=-0.005):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), handlelength=2.2, columnspacing=1.2)


def _grp(sim, L, gt, g):
    return sim[(sim.L == L) & (sim.group_type == gt) & (sim.group == g) & (sim.cost_ratio == RATIO)]


def cost_by_q(split: str, sim_val: pd.DataFrame, sim: pd.DataFrame) -> str:
    """Tổng chi phí theo q (L=2, tỷ lệ 10), sao tại q* chọn trên validation (cho từng nhóm); đường chấm = chính sách điểm."""
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.6), sharex=True)
    for ax, (gt, g) in zip(axes.ravel(), [("overall", "overall"), ("speed", "fast"), ("speed", "medium"), ("speed", "slow")]):
        for m in ALL:
            v = _grp(sim_val[sim_val.model == m], L0, gt, g)
            vq = v[v.policy == "quantile"].sort_values("q")
            qstar = float(vq.loc[vq.total_cost.idxmin(), "q"])
            s = _grp(sim[sim.model == m], L0, gt, g)
            q = s[s.policy == "quantile"].sort_values("q")
            ax.plot(q.q, q.total_cost, STYLES[m], color=COLORS[m], marker="o", ms=2.5, lw=1.2)
            ax.scatter([qstar], [float(q[np.isclose(q.q, qstar)].total_cost.iloc[0])], marker="*", s=70, color=COLORS[m], edgecolor="black", linewidths=0.6, zorder=5)
            ax.axhline(float(s[s.policy == "point"].total_cost.iloc[0]), color=COLORS[m], ls=":", lw=0.7, alpha=0.6)
        ax.set_title(f"Hàng bán {SPEED_VI[g]}" if gt == "speed" else "Tất cả hàng")
    for ax in axes[1]:
        ax.set_xlabel("Mức phân vị q")
    for ax in axes[:, 0]:
        ax.set_ylabel("Chi phí TB/chuỗi (28 ngày)")
    handles = [plt.Line2D([], [], color=COLORS[m], ls=STYLES[m], marker="o", ms=3) for m in ALL] + [
        plt.Line2D([], [], color="gray", marker="*", ls="", ms=8, markeredgecolor="black"), plt.Line2D([], [], color="gray", ls=":")]
    labels = ALL + ["q* (chọn trên validation)", "chính sách điểm"]
    ttl = "validation" if split == "val" else "test; q* chọn trên validation"
    fig.suptitle(f"Tổng chi phí theo mức phân vị q ({ttl}; L={L0}, chi phí thiếu/giữ = {RATIO:g})")
    fig.tight_layout(rect=[0, 0.11, 1, 0.97])
    _legend_below(fig, handles, labels, ncol=4, y=0.0)
    return _save(fig, "policy_total_cost_by_q.png" if split == "val" else "policy_total_cost_by_q_test.png")


def tradeoff(split: str, sim_val: pd.DataFrame, sim: pd.DataFrame) -> str:
    """Đánh đổi fill rate – tồn kho TB theo q; ô vuông = chính sách điểm; test dùng cùng q (đường q từ 0.5 đến 0.99)."""
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.6))
    for ax, sp in zip(axes.ravel()[:3], ["fast", "medium", "slow"]):
        for m in ALL:
            s = _grp(sim[sim.model == m], L0, "speed", sp)
            q = s[s.policy == "quantile"].sort_values("q")
            ax.plot(q.avg_inventory, q.fill_rate, STYLES[m], color=COLORS[m], marker="o", ms=2.5, lw=1.2)
            p = s[s.policy == "point"].iloc[0]
            ax.scatter([p.avg_inventory], [p.fill_rate], marker="s", s=28, color=COLORS[m], edgecolor="black", linewidths=0.6, zorder=5)
        ax.set_title(f"Hàng bán {SPEED_VI[sp]}")
        ax.set_xlabel("Tồn kho cuối ngày TB (đơn vị/chuỗi)")
        ax.set_ylabel("Fill rate")
    axes.ravel()[3].axis("off")
    handles = [plt.Line2D([], [], color=COLORS[m], ls=STYLES[m], marker="o", ms=3) for m in ALL] + [plt.Line2D([], [], color="gray", marker="s", ls="", ms=5, markeredgecolor="black")]
    labels = ALL + ["chính sách điểm"]
    axes.ravel()[3].legend(handles, labels, loc="center", frameon=False, title="Đường: q từ 0.5 đến 0.99", title_fontsize=8)
    ttl = "validation" if split == "val" else "test"
    fig.suptitle(f"Đánh đổi fill rate – tồn kho theo mức phân vị q ({ttl}; L={L0}, chi phí thiếu/giữ = {RATIO:g})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "policy_tradeoff_fill_vs_inventory.png" if split == "val" else "policy_tradeoff_fill_vs_inventory_test.png")


def _row_label(g: str) -> str:
    dt, sp = g.rsplit("_", 1)
    return f"{dt} – {SPEED_VI[sp]}"


def qstar_heatmaps(sim_val: pd.DataFrame) -> list[str]:
    """q* (chọn trên validation, tối thiểu hóa tổng chi phí) theo tỷ lệ chi phí × nhóm hàng; 8 mô hình chia 2 hình, mỗi hình 4 mô hình (L=2)."""
    qs = q_star_table(sim_val)
    ratios = [float(r) for r in CONFIG["cost_ratios"]]
    groups = sorted(qs[qs.group_type == "demand_type_x_speed"].group.unique())
    names = []
    for part, models, fname in [(1, ALL[:4], "policy_qstar_heatmap.png"), (2, ALL[4:], "policy_qstar_heatmap_b.png")]:
        fig, axes = plt.subplots(2, 2, figsize=(W, 6.6), sharey=True)
        im = None
        for ax, m in zip(axes.ravel(), models):
            s = qs[(qs.model == m) & (qs.L == L0) & (qs.group_type == "demand_type_x_speed")]
            mat = s.pivot(index="group", columns="cost_ratio", values="q_star").reindex(index=groups, columns=ratios)
            im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=min(QGRID), vmax=max(QGRID))
            ax.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
            ax.set_yticks(range(len(groups)), [_row_label(g) for g in groups])
            ax.grid(False)
            for i in range(len(groups)):
                for j in range(len(ratios)):
                    v = mat.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.85 else "black", fontsize=7)
            ax.set_title(m)
            ax.set_xlabel("Chi phí thiếu / chi phí giữ")
        fig.suptitle(f"q* chọn trên validation theo tỷ lệ chi phí và nhóm hàng (L={L0}; hình {part}/2)")
        fig.tight_layout(rect=[0, 0.12, 1, 0.97])
        cax = fig.add_axes([0.25, 0.075, 0.5, 0.016])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("q* (mức phân vị tối ưu)", fontsize=8)
        names.append(_save(fig, fname))
    return names


def forecast_error(split: str) -> str:
    """Sai số dự báo cửa sổ H=3 theo nhóm: pinball_avg và WAPE theo tốc độ bán và demand_type, 8 mô hình."""
    ev = pd.concat([pd.read_csv(ROOT / "results" / f"forecast_eval_{split}_all.csv"), pd.read_csv(ROOT / "results" / f"forecast_eval_deepar_{split}_all.csv")], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.8))
    specs = [("pinball_avg", "speed", "Pinball TB (mọi q)"), ("pinball_avg", "demand_type", "Pinball TB (mọi q)"), ("wape", "speed", "WAPE"), ("wape", "demand_type", "WAPE")]
    for ax, (metric, gt, ylab) in zip(axes.ravel(), specs):
        t = ev[(ev.H == 3) & (ev.group_type == gt) & (ev.metric == metric)].pivot(index="group", columns="model", values="value")[ALL]
        if gt == "speed":
            t = t.loc[["fast", "medium", "slow"]]
            t.index = [SPEED_VI[i] for i in t.index]
        t.plot.bar(ax=ax, width=0.85, color=[COLORS[m] for m in ALL], legend=False)
        ax.set_xlabel("Nhóm tốc độ bán" if gt == "speed" else "Loại nhu cầu (demand_type)")
        ax.set_ylabel(ylab)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in ALL]
    ttl = "validation" if split == "val" else "test"
    fig.suptitle(f"Sai số dự báo cửa sổ H=3 theo nhóm hàng ({ttl})")
    fig.tight_layout(rect=[0, 0.1, 1, 0.97])
    _legend_below(fig, handles, ALL, ncol=4, y=0.0)
    return _save(fig, "policy_forecast_error_by_group.png" if split == "val" else "policy_forecast_error_by_group_test.png")


def architecture() -> str:
    """Sơ đồ DeepAR-style: encoder GRU → bước dự báo → đầu NB → lấy mẫu ŷ (mũi tên đỏ quay lại bước k+1) → tổng cửa sổ → phân vị → order-up-to."""
    fig, ax = plt.subplots(figsize=(W, 6.2))
    ax.axis("off"); ax.grid(False)
    ax.set_xlim(0, 11.1); ax.set_ylim(0, 10.8)

    def box(x, y, w, h, txt, c):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="black", lw=0.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8)

    def arr(x1, y1, x2, y2, color="black", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color, ls=ls))

    box(0.1, 8.3, 3.2, 2.3, "Đầu vào mỗi ngày d\n$y_{d-1}/s$, $s=1+\\overline{y}_{ctx}$\nembedding chuỗi (8)\ncovariates đã biết:\nthứ, tháng, ngày, SNAP,\nsự kiện, giá tương đối", "#dbeafe")
    box(3.8, 8.3, 2.6, 2.3, "Encoder GRU\ncontext $C$ ngày\n(teacher forcing)\n→ trạng thái $h_t$", "#fde68a")
    box(6.9, 8.3, 2.6, 2.3, "Bước dự báo\n$k=1..H$: GRU nhận\n$\\hat y_{t+k-1}$ (k=1: $y_t$)\n+ covariates ngày $t+k$", "#fde68a")
    arr(3.3, 9.45, 3.8, 9.45); arr(6.4, 9.45, 6.9, 9.45)
    box(6.9, 5.4, 2.6, 2.3, "Đầu Negative\nBinomial\n$\\mu=s\\cdot softplus(\\cdot)$\n$\\alpha=(softplus(\\cdot)+\\epsilon)\\kappa$", "#bbf7d0")
    arr(8.2, 8.3, 8.2, 7.7)
    box(6.9, 2.5, 2.6, 2.3, "Lấy mẫu $\\hat y_{t+k}$\n(n=500 quỹ đạo,\nGamma–Poisson)", "#bbf7d0")
    arr(8.2, 5.4, 8.2, 4.8)
    # vòng lặp: ŷ_{t+k} → bước k+1 (đi theo mép phải, ngoài các khối)
    ax.plot([9.5, 10.1, 10.1], [3.65, 3.65, 9.45], color="#b91c1c", lw=1.3, ls="--")
    arr(10.1, 9.45, 9.5, 9.45, color="#b91c1c", ls="--")
    ax.text(10.7, 6.5, "$\\hat y_{t+k}$ làm đầu vào\nbước $k+1$", rotation=90, fontsize=8, ha="center", va="center", color="#b91c1c")
    box(3.8, 2.5, 2.6, 2.3, "Tổng cửa sổ\n$D_H=\\sum_{k=1}^{H}\\hat y_{t+k}$\n(mỗi quỹ đạo)", "#fecaca")
    arr(6.9, 3.65, 6.4, 3.65)
    box(0.1, 2.5, 3.2, 2.3, "Phân vị kinh nghiệm\n$\\hat Q_q(D_H)$ (số nguyên)\nq = 0.3 … 0.99", "#fecaca")
    arr(3.8, 3.65, 3.3, 3.65)
    box(0.1, 0.1, 3.2, 1.8, "Order-up-to\n$S=\\hat Q_q(D_H)$; đặt $\\max(0,S-IP)$,\nhàng về sau $L$ ngày", "#e9d5ff")
    arr(1.7, 2.5, 1.7, 1.9)
    ax.set_title("DeepAR-style: từ chuỗi doanh số tới mức đặt hàng order-up-to")
    fig.tight_layout()
    return _save(fig, "deepar_architecture.png")


def nll_curves() -> list[str]:
    """Đường NLL theo epoch: DeepAR (huấn luyện và holdout, 3 seed) và MLP-NB (holdout; 30 epoch cho 3 seed và 80 epoch cho seed 42)."""
    names = []
    cv = pd.read_csv(ROOT / "results" / "deepar_nll_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1))
    pal = {42: "#ff7f0e", 43: "#a85200", 44: "#ffc078"}  # tông cam của DeepAR
    for sd, g in cv.groupby("seed"):
        axes[0].plot(g.epoch, g.train_nll, marker="o", ms=2.5, color=pal[int(sd)], label=f"seed {sd}")
        axes[1].plot(g.epoch, g.holdout_nll, marker="o", ms=2.5, color=pal[int(sd)], label=f"seed {sd}")
        b = g.loc[g.holdout_nll.idxmin()]
        axes[1].scatter([b.epoch], [b.holdout_nll], marker="*", s=70, color=pal[int(sd)], edgecolor="black", linewidths=0.6, zorder=5)
    axes[0].set(title="NLL huấn luyện", xlabel="Epoch", ylabel="NLL TB/bước")
    axes[1].set(title="NLL holdout nội bộ (sao = epoch chọn)", xlabel="Epoch", ylabel="NLL TB/bước")
    for a in axes:
        a.legend(frameon=False)
    fig.suptitle("DeepAR: NLL theo epoch (3 seed; huấn luyện trên train)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    names.append(_save(fig, "deepar_nll_curves.png"))
    mc = pd.read_csv(ROOT / "results" / "mlp_nll_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1))
    for ax, H in zip(axes, (3, 4)):
        pal = {42: "#d62728", 43: "#7f1d1d", 44: "#f19b9b"}  # tông đỏ của MLP-NB
        for (seed, mx), g in mc[mc.H == H].groupby(["seed", "max_epochs"]):
            long = mx > 30
            col = "black" if long else pal[int(seed)]
            ax.plot(g.epoch, g.holdout_nll, "--" if long else "-", lw=1.4 if long else 1.0, marker=None if long else "o", ms=2, color=col, label=f"seed {seed}, tối đa {mx} epoch")
            ax.scatter([g.epoch.iloc[int(g.holdout_nll.argmin())]], [g.holdout_nll.min()], marker="*", s=70, color=col, edgecolor="black", linewidths=0.6, zorder=5)
        ax.set(title=f"H={H}", xlabel="Epoch", ylabel="NLL holdout nội bộ")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("MLP-NB: NLL holdout theo epoch (sao = epoch tốt nhất; train)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    names.append(_save(fig, "mlp_nll_curves.png"))
    return names


def main() -> None:
    sv, st = _sim("val"), _sim("test")
    out = [cost_by_q("val", sv, sv), cost_by_q("test", sv, st), tradeoff("val", sv, sv), tradeoff("test", sv, st)]
    out += qstar_heatmaps(sv)
    out += [forecast_error("val"), forecast_error("test"), architecture()] + nll_curves()
    print("\n".join(out))


if __name__ == "__main__":
    main()
