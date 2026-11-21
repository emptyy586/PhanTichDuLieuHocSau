from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ROOT as PROJECT_ROOT, load_config, set_seed

ROOT = PROJECT_ROOT


def build_time_split(calendar: pd.DataFrame, val_days: int, test_days: int) -> pd.DataFrame:
    """Gán nhãn split theo thời gian: train/val/test, không xáo trộn."""
    calendar = calendar.sort_values("date").reset_index(drop=True).copy()
    total_days = len(calendar)
    train_end = total_days - val_days - test_days

    calendar["split"] = "train"
    calendar.iloc[train_end : train_end + val_days, calendar.columns.get_loc("split")] = "val"
    calendar.iloc[-test_days:, calendar.columns.get_loc("split")] = "test"
    return calendar


def load_processed_data(config: dict) -> pd.DataFrame:
    """Đọc và xử lý dữ liệu M5 theo store_id, dept_id đã chọn."""
    raw_dir = ROOT / config["raw_data_dir"]
    store_id = config["store_id"]
    dept_id = config["dept_id"]

    sales_cols = ["id", "item_id", "dept_id", "store_id"] + [f"d_{i}" for i in range(1, 1942)]
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv", usecols=sales_cols)
    sales = sales[(sales["store_id"] == store_id) & (sales["dept_id"] == dept_id)].copy()

    sales_long = sales.melt(
        id_vars=["id", "item_id", "dept_id", "store_id"],
        value_vars=[c for c in sales.columns if c.startswith("d_")],
        var_name="d",
        value_name="y",
    )
    sales_long["y"] = pd.to_numeric(sales_long["y"], errors="coerce")

    calendar = pd.read_csv(raw_dir / "calendar.csv")
    calendar = calendar[["date", "wm_yr_wk", "weekday", "month", "event_name_1", "event_type_1", "snap_CA", "d"]].copy()
    calendar = calendar[calendar["d"].isin(sales_long["d"].unique())].copy()

    sell_prices = pd.read_csv(raw_dir / "sell_prices.csv")
    sell_prices = sell_prices[(sell_prices["store_id"] == store_id)].copy()

    data = sales_long.merge(calendar, on="d", how="left")
    data = data.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    data = build_time_split(data, config["val_days"], config["test_days"])
    data["split"] = data["split"].astype("category")
    return data


def main() -> None:
    """Chạy pipeline dữ liệu chính và in các thống kê trọng yếu."""
    config = load_config(ROOT / "configs" / "config.yaml")
    set_seed(int(config["seed"]))

    raw_dir = ROOT / config["raw_data_dir"]
    processed_dir = ROOT / config["processed_data_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    sales_cols = ["id", "item_id", "dept_id", "store_id"] + [f"d_{i}" for i in range(1, 1942)]
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv", usecols=sales_cols)
    sales = sales[(sales["store_id"] == config["store_id"]) & (sales["dept_id"] == config["dept_id"])].copy()
    sales_long = sales.melt(
        id_vars=["id", "item_id", "dept_id", "store_id"],
        value_vars=[c for c in sales.columns if c.startswith("d_")],
        var_name="d",
        value_name="y",
    )
    sales_long["y"] = pd.to_numeric(sales_long["y"], errors="coerce")

    calendar = pd.read_csv(raw_dir / "calendar.csv")
    calendar = calendar[["date", "wm_yr_wk", "weekday", "month", "event_name_1", "event_type_1", "snap_CA", "d"]].copy()
    calendar = calendar[calendar["d"].isin(sales_long["d"].unique())].copy()
    calendar = build_time_split(calendar, config["val_days"], config["test_days"])

    sell_prices = pd.read_csv(raw_dir / "sell_prices.csv")
    sell_prices = sell_prices[(sell_prices["store_id"] == config["store_id"])].copy()

    data = sales_long.merge(calendar, on="d", how="left")
    data = data.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    data["split"] = data["split"].astype("category")
    output_path = processed_dir / "ca1_foods3.parquet"
    data.to_parquet(output_path, index=False)

    n_series = data["id"].nunique()
    n_days_per_series = data.groupby("id")["d"].nunique()
    split_counts = data["split"].value_counts()

    split_dates = (
        data.groupby("split", observed=False)["date"]
        .agg(["min", "max"])
        .rename(columns={"min": "start_date", "max": "end_date"})
    )

    missing_sell_price_ratio = data["sell_price"].isna().mean()

    print(f"shape: {data.shape}")
    print(f"so chuoi: {n_series}")
    print(f"so ngay moi chuoi: min={n_days_per_series.min()}, max={n_days_per_series.max()}, median={n_days_per_series.median():.1f}")
    print(f"so dong theo split: {dict(split_counts)}")
    print(f"ngay bat dau/ket thuc tung split:\n{split_dates}")
    print(f"ty le sell_price khuyet: {missing_sell_price_ratio:.4%}")

    assert n_days_per_series.nunique() == 1, "Mỗi chuỗi phải có cùng số ngày."
    assert (n_days_per_series == 1941).all(), "Mỗi chuỗi phải có đủ 1941 ngày."
    assert (data.groupby("id")["split"].apply(lambda s: (s == "val").sum() == 28)).all(), "Mỗi chuỗi phải có 28 ngày val."
    assert (data.groupby("id")["split"].apply(lambda s: (s == "test").sum() == 28)).all(), "Mỗi chuỗi phải có 28 ngày test."
    assert split_dates.loc["train", "end_date"] < split_dates.loc["val", "start_date"] < split_dates.loc["test", "start_date"], "Thu tu train < val < test khong dung."

    print(f"Da luu parquet: {output_path}")


if __name__ == "__main__":
    main()
