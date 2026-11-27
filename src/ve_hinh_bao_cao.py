"""GĐ9/9b: vẽ lại hình để đưa vào báo cáo Word (không đổi số liệu; chỉ đọc CSV/npz có sẵn).

Quy cách: rộng 6.3 inch, dpi 300, chữ >= 8pt (số trong ô heatmap >= 7pt), bỏ tiền tố số hình trong tiêu đề, tiêu đề ghi tập dữ liệu/tham số.
    python -m src.ve_hinh_bao_cao
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.phan_tich_chinh_sach import COLORS, STYLES, q_star_table
from src.mo_hinh_xac_suat import CONFIG, QGRID
from src.tien_ich import ROOT

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
    return pd.concat([pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_{split}.csv"), pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_deepar_{split}.csv")], ignore_index=True)


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
    return _save(fig, "chinh_sach_tong_chi_phi_theo_q.png" if split == "val" else "chinh_sach_tong_chi_phi_theo_q_test.png")


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
    return _save(fig, "chinh_sach_danh_doi_fill_ton_kho.png" if split == "val" else "chinh_sach_danh_doi_fill_ton_kho_test.png")


def _row_label(g: str) -> str:
    dt, sp = g.rsplit("_", 1)
    return f"{dt} – {SPEED_VI[sp]}"


def qstar_heatmaps(sim_val: pd.DataFrame) -> list[str]:
    """q* (chọn trên validation, tối thiểu hóa tổng chi phí) theo tỷ lệ chi phí × nhóm hàng; 8 mô hình chia 2 hình, mỗi hình 4 mô hình (L=2)."""
    qs = q_star_table(sim_val)
    ratios = [float(r) for r in CONFIG["cost_ratios"]]
    groups = sorted(qs[qs.group_type == "demand_type_x_speed"].group.unique())
    names = []
    for part, models, fname in [(1, ALL[:4], "chinh_sach_heatmap_q_sao.png"), (2, ALL[4:], "chinh_sach_heatmap_q_sao_b.png")]:
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
        fig.suptitle(f"q* chọn trên validation theo tỷ lệ chi phí và nhóm hàng (L={L0})")
        fig.tight_layout(rect=[0, 0.12, 1, 0.97])
        cax = fig.add_axes([0.25, 0.075, 0.5, 0.016])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("q* (mức phân vị tối ưu)", fontsize=8)
        names.append(_save(fig, fname))
    return names


def forecast_error(split: str) -> str:
    """Sai số dự báo cửa sổ H=3 theo nhóm: pinball_avg và WAPE theo tốc độ bán và demand_type, 8 mô hình."""
    ev = pd.concat([pd.read_csv(ROOT / "results" / f"danh_gia_du_bao_{split}_all.csv"), pd.read_csv(ROOT / "results" / f"danh_gia_du_bao_deepar_{split}_all.csv")], ignore_index=True)
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
    return _save(fig, "chinh_sach_sai_so_du_bao_theo_nhom.png" if split == "val" else "chinh_sach_sai_so_du_bao_theo_nhom_test.png")


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
    return _save(fig, "deepar_kien_truc.png")


def nll_curves() -> list[str]:
    """Đường NLL theo epoch: DeepAR (huấn luyện và holdout, 3 seed) và MLP-NB (holdout; 30 epoch cho 3 seed và 80 epoch cho seed 42)."""
    names = []
    cv = pd.read_csv(ROOT / "results" / "deepar_duong_nll.csv")
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
    names.append(_save(fig, "deepar_duong_nll.png"))
    mc = pd.read_csv(ROOT / "results" / "mlp_duong_nll.csv")
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
    names.append(_save(fig, "mlp_duong_nll.png"))
    return names


# ----------------------------------------------------------------------------- GĐ9b: hình tổng hợp val vs test
def final_val_vs_test_forecast() -> str:
    """Sai số dự báo cửa sổ H=3 tổng thể: validation vs test cho 8 mô hình (pinball TB mọi q và WAPE)."""
    ev = {s: pd.concat([pd.read_csv(ROOT / "results" / f"danh_gia_du_bao_{s}_all.csv"), pd.read_csv(ROOT / "results" / f"danh_gia_du_bao_deepar_{s}_all.csv")], ignore_index=True) for s in ("val", "test")}
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.6))
    x = np.arange(len(ALL))
    w = 0.38
    for ax, metric, title in zip(axes, ["pinball_avg", "wape"], ["Pinball trung bình (mọi q)", "WAPE (trung bình cửa sổ)"]):
        vals = {s: ev[s][(ev[s].H == 3) & (ev[s].group_type == "overall") & (ev[s].metric == metric)].set_index("model").value.reindex(ALL).values for s in ("val", "test")}
        ax.bar(x - w / 2, vals["val"], w, color=[COLORS[m] for m in ALL], edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, vals["test"], w, color=[COLORS[m] for m in ALL], edgecolor="black", linewidth=0.5, hatch="///")
        ax.set_xticks(x, ALL, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Pinball TB" if metric == "pinball_avg" else "WAPE")
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="black", lw=0.5), plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="black", lw=0.5, hatch="///")]
    fig.suptitle("Sai số dự báo cửa sổ H=3: validation và test")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    _legend_below(fig, handles, ["validation (nền đặc)", "test (nền gạch chéo)"], ncol=2, y=0.0)
    return _save(fig, "cuoi_du_bao_val_va_test.png")


def final_cost_val_vs_test() -> str:
    """Chi phí mô phỏng tại q* chọn trên validation: validation vs test, 8 mô hình, L=2 và L=3 (tỷ lệ chi phí 10); chú giải dưới hình (không gán nhãn trực tiếp)."""
    q = pd.read_csv(ROOT / "results" / "q_sao_co_deepar.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.6))
    markers = {"WindowAverage": "o", "SeasonalNaive": "s", "CrostonSBA": "^", "TSB": "v", "LGBM-Q": "D", "LGBM-PoisNB": "P", "MLP-NB": "X", "DeepAR": "*"}
    for ax, L in zip(axes, (2, 3)):
        t = q[(q.L == L) & (q.cost_ratio == RATIO)].set_index("model").reindex(ALL)
        for m in ALL:
            ax.scatter(t.loc[m, "cost_val"], t.loc[m, "cost_test"], color=COLORS[m], marker=markers[m], s=60 if m == "DeepAR" else 38, edgecolor="black", linewidths=0.5, zorder=3)
        lo = min(t.cost_val.min(), t.cost_test.min()) - 6
        hi = max(t.cost_val.max(), t.cost_test.max()) + 6
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"L={L}")
        ax.set_xlabel("Chi phí TB/chuỗi – validation")
        ax.set_ylabel("Chi phí TB/chuỗi – test")
    handles = [plt.Line2D([], [], color=COLORS[m], marker=markers[m], ls="", ms=7 if m == "DeepAR" else 5, markeredgecolor="black", markeredgewidth=0.5) for m in ALL] + [plt.Line2D([], [], color="k", ls="--")]
    fig.suptitle(f"Chi phí mô phỏng tại q* chọn trên validation: validation và test (chi phí thiếu/giữ = {RATIO:g})")
    fig.tight_layout(rect=[0, 0.16, 1, 0.94])
    _legend_below(fig, handles, ALL + ["bằng nhau (val = test)"], ncol=3, y=0.0)
    return _save(fig, "cuoi_chi_phi_val_va_test.png")


# ----------------------------------------------------------------------------- GĐ9b: hình EDA và phân loại chuỗi
DT_COLORS = {"smooth": "#1f77b4", "erratic": "#d62728", "intermittent": "#2ca02c", "lumpy": "#9467bd"}
DT_VI = {"smooth": "smooth", "erratic": "erratic", "intermittent": "intermittent", "lumpy": "lumpy"}


def eda_figures() -> list[str]:
    """Vẽ lại các hình EDA mùa vụ (train, sau ngày ra mắt): xu hướng, thứ, tháng, SNAP/sự kiện, tên sự kiện, tỷ lệ ngày 0."""
    from src import kham_pha_du_lieu as eda

    data = pd.read_parquet(ROOT / "data" / "processed" / "ca1_foods3.parquet")
    train = eda.load_train(data)
    act = eda.active_rows(train)
    tb = eda.seasonality_tables(act)
    names = []
    daily = train.groupby("date")["y"].sum()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(daily.index, daily.values, color="gray", alpha=0.45, lw=0.6, label="Tổng doanh số ngày")
    ax.plot(daily.index, daily.rolling(28, min_periods=1).mean(), color="#d62728", lw=1.6, label="Trung bình trượt 28 ngày")
    ax.set(title="Tổng doanh số CA_1 × FOODS_3 theo ngày (train)", xlabel="Ngày", ylabel="Tổng đơn vị bán")
    ax.legend(frameon=False)
    fig.tight_layout()
    names.append(_save(fig, "kham_pha_xu_huong_tong.png"))

    t = tb["weekday"]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.bar(eda.WEEKDAY_VI, t["mean_y"], color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax.set(title="Doanh số trung bình theo thứ trong tuần (train)", xlabel="Thứ", ylabel="Đơn vị/chuỗi/ngày")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    names.append(_save(fig, "kham_pha_theo_thu.png"))

    t = tb["month"]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.bar(t["month"].astype(str), t["mean_y"], color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set(title="Doanh số trung bình theo tháng (train)", xlabel="Tháng", ylabel="Đơn vị/chuỗi/ngày")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    names.append(_save(fig, "kham_pha_theo_thang.png"))

    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0))
    axes[0].bar(["Không SNAP", "Có SNAP"], tb["snap"]["mean_y"], color=["#7aa6c2", "#f4a261"], edgecolor="black", linewidth=0.5)
    axes[0].set(title="Theo ngày SNAP", ylabel="Đơn vị/chuỗi/ngày")
    axes[1].bar(["Không sự kiện", "Có sự kiện"], tb["event_flag"]["mean_y"], color=["#8ecae6", "#ffb703"], edgecolor="black", linewidth=0.5)
    axes[1].set(title="Theo ngày có sự kiện")
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.suptitle("Doanh số trung bình theo SNAP và sự kiện (train)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    names.append(_save(fig, "kham_pha_snap_su_kien.png"))

    en = tb["event_name"]
    fig, ax = plt.subplots(figsize=(W, 5.6))
    ax.barh(en["event_name_1"], en["index_vs_mean"], color="#6a4c93", edgecolor="black", linewidth=0.4)
    ax.axvline(1.0, color="black", ls="--", lw=0.9)
    ax.set(title="Chỉ số doanh số theo từng sự kiện (train; 1,0 = trung bình)", xlabel="Chỉ số so với trung bình")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ix = list(en["event_name_1"]).index("Christmas")
    ax.text(0.03, ix, "Christmas = 0 (cửa hàng đóng cửa)", fontsize=8, va="center", ha="left")
    fig.tight_layout()
    names.append(_save(fig, "kham_pha_ten_su_kien.png"))

    z = act.groupby("id")["y"].apply(lambda v: float((v == 0).mean()))
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.hist(z, bins=30, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axvline(z.median(), color="#d62728", ls="--", label=f"Trung vị = {z.median():.4f}".replace(".", ","))
    ax.set(title="Phân bố tỷ lệ ngày bán bằng 0 theo chuỗi (train, sau ngày ra mắt)", xlabel="Tỷ lệ ngày bằng 0", ylabel="Số chuỗi")
    ax.legend(frameon=False)
    fig.tight_layout()
    names.append(_save(fig, "kham_pha_ti_le_khong.png"))
    return names


def adi_cv2_scatter() -> str:
    """Phân tán ADI – CV² (train) màu theo demand_type, ghi nhãn ngưỡng ADI = 1,32 và CV² = 0,49 (thang log)."""
    f = pd.read_csv(ROOT / "results" / "phan_loai_chuoi.csv")
    f = f[np.isfinite(f.adi) & np.isfinite(f.cv2) & (f.cv2 > 0)]
    adi_t, cv2_t = float(CONFIG["adi_threshold"]), float(CONFIG["cv2_threshold"])
    fig, ax = plt.subplots(figsize=(W, 4.4))
    for dt, g in f.groupby("demand_type"):
        ax.scatter(g.adi, g.cv2, s=9, alpha=0.8, color=DT_COLORS[dt], label=f"{DT_VI[dt]} (n={len(g)})", linewidths=0)
    ax.axvline(adi_t, color="black", ls="--", lw=0.9)
    ax.axhline(cv2_t, color="black", ls="--", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.text(adi_t * 1.04, ax.get_ylim()[0] * 1.3, f"ADI = {adi_t:g}".replace(".", ","), fontsize=8, va="bottom")
    ax.text(0.995, cv2_t * 1.06, f"CV² = {cv2_t:g}".replace(".", ","), fontsize=8, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set(title="Phân bố ADI – CV² của 823 chuỗi theo demand_type (train)", xlabel="ADI (thang log)", ylabel="CV² (thang log)")
    ax.legend(title="demand_type", frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, "adi_cv2_phan_tan.png")


def main() -> None:
    sv, st = _sim("val"), _sim("test")
    out = [cost_by_q("val", sv, sv), cost_by_q("test", sv, st), tradeoff("val", sv, sv), tradeoff("test", sv, st)]
    out += qstar_heatmaps(sv)
    out += [forecast_error("val"), forecast_error("test"), architecture()] + nll_curves()
    out += [final_val_vs_test_forecast(), final_cost_val_vs_test()] + eda_figures() + [adi_cv2_scatter()]
    print("\n".join(out))


if __name__ == "__main__":
    main()
