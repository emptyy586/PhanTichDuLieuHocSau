from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tien_ich import ROOT

FIG_DIR = ROOT / "results" / "figures"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_VI = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]


def load_train(data: pd.DataFrame) -> pd.DataFrame:
    """Chỉ giữ split train (EDA không nhìn val/test) và chuẩn hóa kiểu ngày."""
    out = data[data["split"] == "train"].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def active_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Chỉ giữ các ngày từ ngày bán đầu tiên của mỗi chuỗi (bỏ ngày trước khi ra mắt)."""
    first = train[train["y"] > 0].groupby("id")["date"].min().rename("first_sale")
    out = train.merge(first, on="id", how="inner")
    return out[out["date"] >= out["first_sale"]].copy()


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return f"results/figures/{name}"


def seasonality_tables(act: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tính các bảng mùa vụ: thứ trong tuần, tháng, SNAP, sự kiện (chỉ trên train, sau ngày ra mắt)."""
    tables: dict[str, pd.DataFrame] = {}
    overall = act["y"].mean()
    wd = act.groupby("weekday")["y"].mean().reindex(WEEKDAY_ORDER)
    tables["weekday"] = pd.DataFrame({"weekday": WEEKDAY_ORDER, "mean_y": wd.values, "index_vs_mean": wd.values / overall})
    mo = act.groupby("month")["y"].mean()
    tables["month"] = pd.DataFrame({"month": mo.index, "mean_y": mo.values, "index_vs_mean": mo.values / overall})
    snap = act.groupby("snap_CA")["y"].mean()
    tables["snap"] = pd.DataFrame({"snap_CA": snap.index, "mean_y": snap.values})
    ev = act.assign(has_event=act["event_name_1"].notna()).groupby("has_event")["y"].mean()
    tables["event_flag"] = pd.DataFrame({"has_event": ev.index, "mean_y": ev.values})
    et = act[act["event_type_1"].notna()].groupby("event_type_1")["y"].agg(["mean", "size"]).reset_index()
    et["index_vs_mean"] = et["mean"] / overall
    tables["event_type"] = et.rename(columns={"mean": "mean_y", "size": "n_rows"})
    en = act[act["event_name_1"].notna()].groupby("event_name_1")["y"].mean().sort_values()
    tables["event_name"] = pd.DataFrame({"event_name_1": en.index, "mean_y": en.values, "index_vs_mean": en.values / overall})
    return tables


def plot_all(train: pd.DataFrame, act: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Vẽ và lưu toàn bộ biểu đồ EDA mùa vụ; trả về danh sách đường dẫn."""
    paths: list[str] = []

    daily = train.groupby("date")["y"].sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily.values, color="gray", alpha=0.4, label="Tổng doanh số ngày")
    ax.plot(daily.index, daily.rolling(28, min_periods=1).mean(), color="darkred", lw=2, label="Trung bình trượt 28 ngày")
    ax.set(title="Hình 1. Tổng doanh số CA_1 × FOODS_3 theo ngày (train)", xlabel="Ngày", ylabel="Tổng đơn vị bán")
    ax.legend()
    paths.append(_save(fig, "kham_pha_xu_huong_tong.png"))

    t = tables["weekday"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(WEEKDAY_VI, t["mean_y"], color="darkorange")
    ax.set(title="Hình 2. Doanh số trung bình theo thứ trong tuần", xlabel="Thứ", ylabel="Đơn vị/chuỗi/ngày")
    paths.append(_save(fig, "kham_pha_theo_thu.png"))

    t = tables["month"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(t["month"].astype(str), t["mean_y"], color="seagreen")
    ax.set(title="Hình 3. Doanh số trung bình theo tháng", xlabel="Tháng", ylabel="Đơn vị/chuỗi/ngày")
    paths.append(_save(fig, "kham_pha_theo_thang.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    s = tables["snap"]
    axes[0].bar(["Không SNAP", "Có SNAP"], s["mean_y"], color=["#7aa6c2", "#f4a261"])
    axes[0].set(title="Hình 4a. Theo ngày SNAP", ylabel="Đơn vị/chuỗi/ngày")
    e = tables["event_flag"]
    axes[1].bar(["Không sự kiện", "Có sự kiện"], e["mean_y"], color=["#8ecae6", "#ffb703"])
    axes[1].set(title="Hình 4b. Theo ngày có sự kiện", ylabel="Đơn vị/chuỗi/ngày")
    paths.append(_save(fig, "kham_pha_snap_su_kien.png"))

    en = tables["event_name"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(en["event_name_1"], en["index_vs_mean"], color="#6a4c93")
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.set(title="Hình 5. Chỉ số doanh số theo từng sự kiện (1.0 = trung bình)", xlabel="Chỉ số so với trung bình", ylabel="")
    ax.tick_params(axis="y", labelsize=7)
    paths.append(_save(fig, "kham_pha_ten_su_kien.png"))

    z = act.groupby("id")["y"].apply(lambda s: float((s == 0).mean()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(z, bins=30, color="steelblue", edgecolor="black")
    ax.axvline(z.median(), color="red", ls="--", label=f"Trung vị = {z.median():.4f}")
    ax.set(title="Hình 6. Phân bố tỷ lệ ngày bán bằng 0 (sau ngày ra mắt)", xlabel="Tỷ lệ ngày bằng 0", ylabel="Số chuỗi")
    ax.legend()
    paths.append(_save(fig, "kham_pha_ti_le_khong.png"))
    return paths


def demand_speed_table(features: pd.DataFrame) -> pd.DataFrame:
    """Bảng số chuỗi theo demand_type × speed (có tổng)."""
    tab = pd.crosstab(features["demand_type"], features["speed"])
    for c in ["fast", "medium", "slow"]:
        if c not in tab:
            tab[c] = 0
    tab = tab[["fast", "medium", "slow"]]
    tab["total"] = tab.sum(axis=1)
    tab.loc["total"] = tab.sum()
    return tab
