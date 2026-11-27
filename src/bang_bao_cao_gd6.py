"""GĐ6: dựng bảng cho báo cáo 06 và hình đường NLL holdout của MLP-NB (đọc các CSV đã sinh)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tien_ich import ROOT

ORDER = ["WindowAverage", "SeasonalNaive", "CrostonSBA", "TSB", "LGBM-Q", "LGBM-PoisNB", "MLP-NB"]


def ktc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Bảng 'mean [lo; hi]' của cột chênh lệch `col` (đuôi _lo/_hi) theo mô hình × cost_ratio."""
    d = df.assign(txt=lambda x: x.apply(lambda r: f"{r[col]:.2f} [{r[col + '_lo']:.2f}; {r[col + '_hi']:.2f}]", axis=1))
    return d.pivot(index="model", columns="cost_ratio", values="txt").reindex(ORDER)


def significance_counts(sp: pd.DataFrame) -> pd.DataFrame:
    """Đếm số ô (mô hình × tỷ lệ) mà KTC 95% nằm hoàn toàn dưới 0 / chứa 0 / trên 0, cho từng so sánh và L."""
    rows = []
    for L, g in sp.groupby("L"):
        for name in ("k_minus_q", "q_minus_point", "k_minus_point"):
            lo, hi = g[f"{name}_lo"], g[f"{name}_hi"]
            rows.append({"L": L, "so_sanh": name, "KTC<0 (vế đầu rẻ hơn)": int((hi < 0).sum()), "KTC chứa 0": int(((lo <= 0) & (hi >= 0)).sum()),
                         "KTC>0 (vế đầu đắt hơn)": int((lo > 0).sum()), "tong_o": len(g)})
    return pd.DataFrame(rows)


def seed_summary(ms: pd.DataFrame) -> pd.DataFrame:
    """Trung bình ± độ lệch chuẩn (ddof=1) của pinball_avg/WAPE/MASE qua các seed, theo (split, H)."""
    rows = []
    for (split, H), g in ms.groupby(["split", "H"]):
        r = {"split": split, "H": H, "n_seed": len(g)}
        for m in ("pinball_avg", "wape", "mase"):
            r[m] = f"{g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}"
        r["epochs"] = "/".join(str(int(e)) for e in g.sort_values("seed").epochs)
        r["kappa"] = "/".join(f"{k:g}" for k in g.sort_values("seed").kappa)
        rows.append(r)
    return pd.DataFrame(rows)


def plot_nll(curves: pd.DataFrame) -> str:
    """Hình: NLL holdout theo epoch cho từng seed (tối đa 30 epoch) và lần chạy 80 epoch (seed 42), tách H=3 và H=4."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, H in zip(axes, (3, 4)):
        for (seed, mx), g in curves[curves.H == H].groupby(["seed", "max_epochs"]):
            long = mx > 30
            ax.plot(g.epoch, g.holdout_nll, "-" if not long else "--", lw=2.2 if long else 1.2, marker="o" if not long else None, ms=3,
                    label=f"seed {seed}, tối đa {mx} epoch")
            ax.scatter([g.epoch.iloc[int(g.holdout_nll.argmin())]], [g.holdout_nll.min()], marker="*", s=120, edgecolor="black", zorder=5)
        ax.set(title=f"H={H}", xlabel="Epoch", ylabel="NLL holdout nội bộ (56 gốc cuối của train)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Hình G1. Đường NLL holdout của MLP-NB theo epoch (sao = epoch tốt nhất)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = ROOT / "results" / "figures" / "mlp_duong_nll.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def bracket_pivot(tab: pd.DataFrame, split: str, H: int) -> pd.DataFrame:
    """Bảng model × q với chuỗi 'P(y<Q) / P(y<=Q)'."""
    t = tab[(tab.split == split) & (tab.H == H)].copy()
    t["txt"] = t.apply(lambda r: f"{r.P_lt:.3f} / {r.P_le:.3f}", axis=1)
    return t.pivot(index="model", columns="q", values="txt").reindex(ORDER)
