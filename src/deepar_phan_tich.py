"""GĐ8: thêm DeepAR vào các bảng so sánh (không sửa file kết quả cũ): mô phỏng tồn kho, q*, bootstrap chênh lệch, hiệu chuẩn, hình.

    python -m src.deepar_phan_tich --what sim          # mo_phong_ton_kho_deepar_{val,test}[_ext].csv
    python -m src.deepar_phan_tich --what tables       # bảng so sánh + bootstrap + hình
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.hieu_chinh import calibration_table, summarize
from src.mo_phong_ton_kho import MODELS, quantile_to_levels, run_inventory_sim, simulate_order_up_to
from src.phan_tich_chinh_sach import FIG, TAB, paired_bootstrap, plot_cost_by_q, plot_tradeoff, q_star_table
from src.mo_hinh_xac_suat import CONFIG, QGRID, group_masks, load_panel, pinball_loss, split_bounds, window_targets
from src.tien_ich import ROOT

ALL = MODELS + ["DeepAR"]
HOLD = float(CONFIG["holding_cost"])
RATIOS_BOOT = [10.0, 20.0]
OTHERS = ["MLP-NB", "LGBM-Q", "TSB"]


def run_sims() -> None:
    """Mô phỏng chỉ cho DeepAR (lưới chính và lưới phụ) → file riêng; các file mô phỏng cũ KHÔNG bị ghi lại."""
    for split in ("val", "test"):
        p = ROOT / "results" / "forecasts" / f"deepar_{split}.npz"
        run_inventory_sim(split, forecast_path=p, out_path=ROOT / "results" / f"mo_phong_ton_kho_deepar_{split}.csv", models=["DeepAR"])
        run_inventory_sim(split, forecast_path=p, out_path=ROOT / "results" / f"mo_phong_ton_kho_deepar_{split}_ext.csv", ext=True, models=["DeepAR"])


def combined_sim(split: str) -> pd.DataFrame:
    """Bảng mô phỏng gộp 8 mô hình (lưới chính + lưới phụ q<0.5); các dòng của 7 mô hình cũ lấy nguyên từ file cũ."""
    parts = [pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_{split}.csv"), pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_{split}_ext.csv"),
             pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_deepar_{split}.csv"), pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_deepar_{split}_ext.csv")]
    return pd.concat(parts, ignore_index=True)


def old_rows_identical(split: str) -> bool:
    """Kiểm tra: các dòng của 7 mô hình cũ trong bảng gộp giống hệt file cũ (dung sai 1e-9)."""
    comb = combined_sim(split)
    old = pd.concat([pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_{split}.csv"), pd.read_csv(ROOT / "results" / f"mo_phong_ton_kho_{split}_ext.csv")], ignore_index=True)
    got = comb[comb.model != "DeepAR"].reset_index(drop=True)
    num = old.select_dtypes("number").columns
    return bool(len(got) == len(old) and (got[num].fillna(-1) - old[num].fillna(-1)).abs().max().max() < 1e-9 and (got.model == old.model).all())


def _levels(arr, model: str, H: int, q: float) -> np.ndarray:
    grid, key = (arr["quantile_grid"], "Q") if np.isclose(arr["quantile_grid"], q).any() else (arr["quantile_grid_ext"], "Qext")
    k = int(np.argmin(np.abs(np.asarray(grid) - q)))
    return quantile_to_levels(model, arr[f"{model}|H{H}|{key}"][..., k])


def _arr(model: str, split: str):
    path = ROOT / "results" / "forecasts" / (f"deepar_{split}.npz" if model == "DeepAR" else f"{split}.npz")
    return np.load(path, allow_pickle=True)


def q_star_val(comb_val: pd.DataFrame, model: str, L: int, ratio: float) -> float:
    """q* của mô hình trên VAL (tổng thể, lưới chính + phụ) cho (L, tỷ lệ chi phí)."""
    t = comb_val[(comb_val.model == model) & (comb_val.L == L) & (comb_val.cost_ratio == ratio) & (comb_val.policy == "quantile") & (comb_val.group_type == "overall")]
    return float(t.loc[t.total_cost.idxmin(), "q"])


def per_series_cost(panel, model: str, split: str, L: int, q: float, ratio: float) -> np.ndarray:
    """Tổng chi phí từng chuỗi (giữ + thiếu) của (mô hình, q) trên split."""
    H = L + int(CONFIG["review_period"])
    s, e = split_bounds(panel, split)
    res = simulate_order_up_to(panel.Y[:, s:e + 1], _levels(_arr(model, split), model, H, q)[None], L)
    return HOLD * (res["hold_units"][0] + ratio * res["lost"][0])


def bootstrap_costs(panel) -> pd.DataFrame:
    """DeepAR − {MLP-NB, LGBM-Q, TSB}: mỗi mô hình tại q* RIÊNG của nó (chọn trên val, tổng thể), tỷ lệ 10 và 20, L=2,3, val và test, theo nhóm tốc độ."""
    comb_val = combined_sim("val")
    rows = []
    speeds = {"overall": np.ones(panel.n_series, bool), "fast": panel.speed == "fast", "medium": panel.speed == "medium", "slow": panel.speed == "slow"}
    for L in (2, 3):
        for ratio in RATIOS_BOOT:
            qd = q_star_val(comb_val, "DeepAR", L, ratio)
            for other in OTHERS:
                qo = q_star_val(comb_val, other, L, ratio)
                for split in ("val", "test"):
                    d = per_series_cost(panel, "DeepAR", split, L, qd, ratio) - per_series_cost(panel, other, split, L, qo, ratio)
                    for g, m in speeds.items():
                        mean, lo, hi = paired_bootstrap(d[m])
                        rows.append({"so_sanh": f"DeepAR - {other}", "L": L, "cost_ratio": ratio, "split": split, "group": g, "q_DeepAR": qd, "q_other": qo,
                                     "mean_diff_cost": mean, "ci_lo": lo, "ci_hi": hi, "n": int(m.sum())})
    return pd.DataFrame(rows)


def bootstrap_pinball(panel) -> pd.DataFrame:
    """DeepAR − {MLP-NB, LGBM-Q, TSB}: chênh pinball_avg theo chuỗi (trung bình trên các gốc và mọi q lưới chính), H=3,4, val/test, theo nhóm tốc độ."""
    speeds = {"overall": np.ones(panel.n_series, bool), "fast": panel.speed == "fast", "medium": panel.speed == "medium", "slow": panel.speed == "slow"}
    rows = []
    for split in ("val", "test"):
        for H in (3, 4):
            from src.mo_hinh_xac_suat import origins_for_split
            org = origins_for_split(panel, split, H)
            actual = window_targets(panel.Y, H)[:, org]
            per = {}
            for m in ["DeepAR"] + OTHERS:
                Q = _arr(m, split)[f"{m}|H{H}|Q"]
                per[m] = np.mean([pinball_loss(actual, Q[..., k], q).mean(axis=1) for k, q in enumerate(QGRID)], axis=0)
            for other in OTHERS:
                for g, mask in speeds.items():
                    mean, lo, hi = paired_bootstrap((per["DeepAR"] - per[other])[mask])
                    rows.append({"so_sanh": f"DeepAR - {other}", "split": split, "H": H, "group": g, "mean_diff_pinball_avg": mean, "ci_lo": lo, "ci_hi": hi, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def forecast_tables() -> dict[str, pd.DataFrame]:
    """Bảng chỉ số dự báo gộp 8 mô hình: pinball_avg/WAPE/MASE (H=3,4; val/test) và hiệu chuẩn (P(y<q), P(y<=q))."""
    ev = {s: pd.concat([pd.read_csv(ROOT / "results" / f"danh_gia_du_bao_{s}_all.csv"), pd.read_csv(ROOT / "results" / f"danh_gia_du_bao_deepar_{s}_all.csv")], ignore_index=True) for s in ("val", "test")}
    rows = []
    for s, d in ev.items():
        for H in (3, 4):
            t = d[(d.H == H) & (d.group_type == "overall") & d.metric.isin(["pinball_avg", "wape", "mase"])].pivot(index="model", columns="metric", values="value").reindex(ALL)
            t.insert(0, "H", H); t.insert(0, "split", s)
            rows.append(t.reset_index())
    main = pd.concat(rows, ignore_index=True)
    cal = pd.concat([calibration_table(pd.read_csv(ROOT / "results" / f"danh_gia_du_bao_deepar_{s}_all.csv"), s) for s in ("val", "test")], ignore_index=True)
    return {"main": main, "cal": cal, "cal_sum": summarize(cal), "ev_val": ev["val"], "ev_test": ev["test"]}


def combined_q_star(split_val: pd.DataFrame, split_test: pd.DataFrame) -> pd.DataFrame:
    """Bảng q*(val) (lưới chính + phụ) cho 8 mô hình và chi phí/fill TEST tại q*(val), theo (L, tỷ lệ), tổng thể."""
    rows = []
    for L in (2, 3):
        for r in [float(x) for x in CONFIG["cost_ratios"]]:
            for m in ALL:
                q = q_star_val(split_val, m, L, r)
                a = split_val[(split_val.model == m) & (split_val.L == L) & (split_val.cost_ratio == r) & (split_val.policy == "quantile") & np.isclose(split_val.q, q) & (split_val.group_type == "overall")].iloc[0]
                b = split_test[(split_test.model == m) & (split_test.L == L) & (split_test.cost_ratio == r) & (split_test.policy == "quantile") & np.isclose(split_test.q, q) & (split_test.group_type == "overall")].iloc[0]
                rows.append({"model": m, "L": L, "cost_ratio": r, "q_star_val": q, "fill_val": a.fill_rate, "cost_val": a.total_cost, "fill_test": b.fill_rate, "cost_test": b.total_cost, "inv_test": b.avg_inventory})
    return pd.DataFrame(rows)


def plot_architecture() -> str:
    """Sơ đồ khối DeepAR → order-up-to (tiếng Việt)."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    boxes = [
        (0.2, 3.2, 2.6, 1.5, "Đầu vào mỗi bước d\n$y_{d-1}/scale$ ($scale=1+\\overline{y}_{ctx}$)\nembedding id (8)\ncovariates đã biết:\nthứ, tháng, ngày, SNAP,\nsự kiện, giá tương đối", "#dbeafe"),
        (3.4, 3.2, 2.2, 1.5, "Encoder GRU\n(context C ngày,\nteacher forcing)\nhidden h, L lớp", "#fde68a"),
        (6.2, 3.2, 2.2, 1.5, "Bước dự báo k=1..H\nGRU nhận mẫu $\\hat y_{t+k-1}$\n+ covariates ngày $t+k$", "#fde68a"),
        (9.0, 3.2, 2.2, 1.5, "Đầu NB\n$\\mu=scale\\cdot softplus(\\cdot)$\n$\\alpha=softplus(\\cdot)+\\epsilon$\n$y\\sim NB(\\mu,\\alpha)$", "#bbf7d0"),
        (6.2, 0.4, 2.4, 1.6, "Tổng cửa sổ\n$D_H=\\sum_{k=1}^{H}\\hat y_{t+k}$\n(n=500 quỹ đạo mẫu)", "#fecaca"),
        (3.4, 0.4, 2.4, 1.6, "Phân vị kinh nghiệm $\\hat Q_q(D_H)$\n(số nguyên; q∈0.3…0.99)", "#fecaca"),
        (0.2, 0.4, 2.6, 1.6, "Chính sách order-up-to\n$S=\\hat Q_q(D_H)$\nđặt $\\max(0,S-IP)$, về sau L ngày", "#e9d5ff"),
    ]
    for x, y, w, h, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="black"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
    arrow = dict(arrowstyle="->", lw=1.5)
    for (x1, y1, x2, y2) in [(2.8, 3.95, 3.4, 3.95), (5.6, 3.95, 6.2, 3.95), (8.4, 3.95, 9.0, 3.95), (7.4, 3.2, 7.4, 2.0), (6.2, 1.2, 5.8, 1.2), (3.4, 1.2, 2.8, 1.2)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow)
    ax.annotate("mẫu $\\hat y_{t+k}$ đưa lại làm đầu vào bước k+1", xy=(7.3, 4.75), xytext=(9.9, 5.35), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.4", lw=1.2), fontsize=8.5, ha="center")
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 5.8)
    ax.set_title("Hình G8.1. Sơ đồ DeepAR-style: encoder GRU → đầu Negative Binomial → lấy mẫu tự hồi quy → tổng cửa sổ → phân vị → order-up-to")
    fig.tight_layout()
    path = FIG / "deepar_kien_truc.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_nll_curves(curves: pd.DataFrame) -> str:
    """Hình: NLL huấn luyện (trung bình trên cửa sổ, mọi bước) và NLL holdout nội bộ (bước dự báo) theo epoch, 3 seed."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    for sd, g in curves.groupby("seed"):
        axes[0].plot(g.epoch, g.train_nll, marker="o", ms=3, label=f"seed {sd}")
        axes[1].plot(g.epoch, g.holdout_nll, marker="o", ms=3, label=f"seed {sd}")
        b = g.loc[g.holdout_nll.idxmin()]
        axes[1].scatter([b.epoch], [b.holdout_nll], marker="*", s=130, edgecolor="black", zorder=5)
    axes[0].set(title="NLL huấn luyện (mọi bước của cửa sổ)", xlabel="Epoch", ylabel="NLL trung bình / bước")
    axes[1].set(title="NLL holdout nội bộ (56 gốc cuối của train, các bước dự báo)\nsao = epoch được chọn", xlabel="Epoch", ylabel="NLL trung bình / bước")
    for a in axes:
        a.grid(alpha=0.3); a.legend()
    fig.suptitle("Hình G8.2. Đường NLL theo epoch của DeepAR (3 seed)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIG / "deepar_duong_nll.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def run_tables() -> None:
    """Dựng mọi bảng/hình GĐ8 từ các CSV/npz đã sinh."""
    panel = load_panel()
    TAB.mkdir(parents=True, exist_ok=True)
    assert old_rows_identical("val") and old_rows_identical("test"), "Dòng của mô hình cũ trong bảng gộp KHÁC bản cũ – DỪNG"
    cv, ct = combined_sim("val"), combined_sim("test")
    combined_q_star(cv, ct).to_csv(ROOT / "results" / "q_sao_co_deepar.csv", index=False)
    bootstrap_costs(panel).to_csv(ROOT / "results" / "deepar_bootstrap_chi_phi.csv", index=False)
    bootstrap_pinball(panel).to_csv(ROOT / "results" / "deepar_bootstrap_pinball.csv", index=False)
    t = forecast_tables()
    t["main"].to_csv(ROOT / "results" / "deepar_du_bao_bang_chinh.csv", index=False)
    t["cal"].to_csv(ROOT / "results" / "deepar_hieu_chinh_bang.csv", index=False)
    t["cal_sum"].to_csv(ROOT / "results" / "deepar_hieu_chinh_tom_tat.csv", index=False)
    # bảng theo nhóm tốc độ cho DeepAR + mô hình so sánh
    rows = []
    for s in ("val", "test"):
        d = t[f"ev_{s}"]
        for H in (3, 4):
            x = d[(d.H == H) & (d.group_type == "speed") & d.metric.isin(["pinball_avg", "wape", "mase"])].pivot_table(index=["model", "group"], columns="metric", values="value").reset_index()
            x.insert(0, "H", H); x.insert(0, "split", s)
            rows.append(x)
    pd.concat(rows, ignore_index=True).to_csv(ROOT / "results" / "deepar_du_bao_theo_toc_do.csv", index=False)
    hp = json.loads((ROOT / "results" / "sieu_tham_so_deepar.json").read_text(encoding="utf-8"))
    log = pd.read_csv(ROOT / "results" / "nhat_ky_tinh_chinh_deepar.csv")
    seeds = pd.read_csv(ROOT / "results" / "deepar_cac_seed.csv")
    tr = seeds.groupby("seed").train_seconds.first()
    pd.DataFrame([{**hp["config"], "dropout": CONFIG["deepar_dropout"], "lr": CONFIG["deepar_lr"], "batch_size": CONFIG["deepar_batch_size"], "windows_per_epoch": CONFIG["deepar_windows_per_epoch"],
                   "max_epochs": CONFIG["deepar_max_epochs"], "kappa": hp["kappa"], "n_samples": hp["n_samples"], "n_params": hp["n_params"],
                   "epochs_seed42/43/44": "/".join(str(hp["epochs_by_seed"][str(s)]) for s in CONFIG["mlp_seeds"]),
                   "train_seconds_seed42/43/44": "/".join(f"{tr.loc[s]:.0f}" for s in CONFIG["mlp_seeds"])}]).to_csv(TAB / "deepar_sieu_tham_so.csv", index=False)
    log.to_csv(TAB / "deepar_nhat_ky_tinh_chinh.csv", index=False)
    plot_architecture()
    plot_nll_curves(pd.read_csv(ROOT / "results" / "deepar_duong_nll.csv"))
    main_val = pd.concat([pd.read_csv(ROOT / "results" / "mo_phong_ton_kho_val.csv"), pd.read_csv(ROOT / "results" / "mo_phong_ton_kho_deepar_val.csv")], ignore_index=True)  # lưới chính như hình cũ
    plot_tradeoff(main_val, models=ALL)
    plot_cost_by_q(main_val, models=ALL)
    print("xong")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["sim", "tables"], required=True)
    a = ap.parse_args()
    run_sims() if a.what == "sim" else run_tables()


if __name__ == "__main__":
    main()
