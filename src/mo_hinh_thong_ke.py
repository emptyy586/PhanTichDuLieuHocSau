from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import CrostonSBA, SeasonalNaive, TSB, WindowAverage

from src.tien_ich import ROOT, load_config, set_seed


CONFIG = load_config(ROOT / "configs" / "thong_so.yaml")
SEED = int(CONFIG["seed"])


def prepare_statsforecast_df(data: pd.DataFrame, split: str) -> pd.DataFrame:
    """Chuyển dataframe dài sang format chuẩn statsforecast: unique_id, ds, y."""
    frame = data[data["split"] == split][["id", "date", "y"]].copy()
    frame = frame.rename(columns={"id": "unique_id", "date": "ds"})
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return frame


def make_model_list() -> list:
    return [
        WindowAverage(window_size=28),
        SeasonalNaive(season_length=7),
        CrostonSBA(),
    ]


def compute_pinball_loss(actual: pd.DataFrame, forecast: pd.DataFrame, model_name: str, q: float = 0.5) -> float:
    """Tính pinball loss ở quantile q, dùng cho cả model điểm và model xác suất."""
    merged = actual[["unique_id", "ds", "y"]].merge(forecast[["unique_id", "ds", model_name]], on=["unique_id", "ds"], how="inner")
    err = merged["y"] - merged[model_name]
    loss = q * np.maximum(err, 0.0) + (1.0 - q) * np.maximum(-err, 0.0)
    return float(loss.mean())


def compute_metrics(train_df: pd.DataFrame, actual: pd.DataFrame, forecast: pd.DataFrame, model_name: str) -> dict:
    """Tính WAPE, MASE và pinball q=0.5 cho một mô hình trên validation."""
    merged = actual.merge(forecast[["unique_id", "ds", model_name]], on=["unique_id", "ds"], how="inner")
    merged["abs_err"] = (merged["y"] - merged[model_name]).abs()
    merged["abs_y"] = merged["y"].abs()

    wape = merged["abs_err"].sum() / merged["abs_y"].sum() if merged["abs_y"].sum() > 0 else np.nan

    per_series = []
    for unique_id, group in merged.groupby("unique_id"):
        train = train_df[train_df["unique_id"] == unique_id].sort_values("ds").copy()
        if train.empty:
            mase = np.nan
        else:
            first_sale = train.loc[train["y"] > 0, "ds"].min()
            if pd.isna(first_sale):
                mase = np.nan
            else:
                train = train[train["ds"] >= first_sale].copy()
                train_naive = train["y"].shift(7)
                mae_naive = (train["y"] - train_naive).abs().dropna().mean()
                if pd.isna(mae_naive) or mae_naive == 0:
                    mase = np.nan
                else:
                    mae_model = (group["y"] - group[model_name]).abs().mean()
                    mase = mae_model / mae_naive
        per_series.append({"unique_id": unique_id, "mase": mase})

    mase = pd.DataFrame(per_series)["mase"].mean()
    pinball_q50 = compute_pinball_loss(actual, forecast, model_name, q=0.5)
    return {"model": model_name, "wape": wape, "mase": mase, "pinball_q50": pinball_q50}


def build_tsb_with_validation(train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> tuple[TSB, dict]:
    """Chọn alpha_d / alpha_p trên 28 ngày cuối của train, không dùng validation split như trước."""
    candidates = [(0.05, 0.05), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
    best = None
    best_metrics = None

    train_sorted = train_df.sort_values(["unique_id", "ds"]).copy()
    valid_df = train_sorted.groupby("unique_id").tail(28).copy() if valid_df is None else valid_df.sort_values(["unique_id", "ds"]).copy()
    train_fit = train_sorted.groupby("unique_id").apply(lambda g: g.iloc[:-28] if len(g) > 28 else g.iloc[:0]).reset_index(drop=True)

    for alpha_d, alpha_p in candidates:
        model = TSB(alpha_d=alpha_d, alpha_p=alpha_p)
        sf = StatsForecast(models=[model], freq="D")
        sf.fit(train_fit)
        preds = sf.predict(h=28)
        s = compute_metrics(train_fit, valid_df, preds, model.alias)
        if best_metrics is None or s["wape"] < best_metrics["wape"]:
            best = model
            best_metrics = s

    if best is None or best_metrics is None:
        raise RuntimeError("TSB selection failed: no valid candidate produced metrics.")

    return best, best_metrics


def summarize_by_group(train_df: pd.DataFrame, actual: pd.DataFrame, forecast: pd.DataFrame, feature_df: pd.DataFrame, group_col: str, model_name: str) -> pd.DataFrame:
    """Tạo bảng tách theo nhóm (demand_type, speed, demand_type x speed)."""
    merged = actual.merge(forecast[["unique_id", "ds", model_name]], on=["unique_id", "ds"], how="inner")
    merged = merged.merge(feature_df[["id", group_col]], left_on="unique_id", right_on="id", how="left")
    rows = []
    for group_name, group in merged.groupby(group_col, dropna=False):
        abs_err = (group["y"] - group[model_name]).abs()
        abs_y = group["y"].abs()
        wape = abs_err.sum() / abs_y.sum() if abs_y.sum() > 0 else np.nan

        per_series = []
        for unique_id, series in group.groupby("unique_id"):
            train = train_df[train_df["unique_id"] == unique_id].sort_values("ds").copy()
            first_sale = train.loc[train["y"] > 0, "ds"].min()
            if pd.isna(first_sale):
                mase = np.nan
            else:
                train = train[train["ds"] >= first_sale].copy()
                train_naive = train["y"].shift(7)
                mae_naive = (train["y"] - train_naive).abs().dropna().mean()
                if pd.isna(mae_naive) or mae_naive == 0:
                    mase = np.nan
                else:
                    mae_model = (series["y"] - series[model_name]).abs().mean()
                    mase = mae_model / mae_naive
            per_series.append({"unique_id": unique_id, "mase": mase})

        mase = pd.DataFrame(per_series)["mase"].mean()
        err = group["y"] - group[model_name]
        loss = 0.5 * np.maximum(err, 0.0) + 0.5 * np.maximum(-err, 0.0)
        pinball_q50 = float(loss.mean())
        rows.append(
            {
                "model": model_name,
                "group": str(group_name),
                "n": group["unique_id"].nunique(),
                "wape": wape,
                "mase": mase,
                "pinball_q50": pinball_q50,
            }
        )
    return pd.DataFrame(rows)


def build_report_tables(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng tổng hợp gồm model tổng quát + tách theo demand_type / speed / demand_type x speed."""
    tsb_model, _ = build_tsb_with_validation(train_df)
    models = make_model_list() + [tsb_model]

    forecast_map: dict[str, pd.DataFrame] = {}
    for model in models:
        sf = StatsForecast(models=[model], freq="D")
        sf.fit(train_df)
        forecast_map[model.alias] = sf.predict(h=28)

    overall_rows = []
    for model in models:
        metrics = compute_metrics(train_df, valid_df, forecast_map[model.alias], model.alias)
        overall_rows.append({
            "model": model.alias,
            "group": "overall",
            "n": valid_df["unique_id"].nunique(),
            "wape": metrics["wape"],
            "mase": metrics["mase"],
            "pinball_q50": metrics["pinball_q50"],
        })
    overall_df = pd.DataFrame(overall_rows)

    feature_df = feature_df.copy()
    feature_df["demand_type_speed"] = feature_df["demand_type"] + "_" + feature_df["speed"]

    by_group_frames = []
    for col in ["demand_type", "speed", "demand_type_speed"]:
        rows = []
        for model in models:
            rows.append(summarize_by_group(train_df, valid_df, forecast_map[model.alias], feature_df, col, model.alias))
        by_group_frames.append(pd.concat(rows, ignore_index=True))

    final = pd.concat([overall_df, *by_group_frames], ignore_index=True)
    return final[["model", "group", "n", "wape", "mase", "pinball_q50"]].sort_values(["model", "group"]).reset_index(drop=True)


def main() -> None:
    set_seed(SEED)
    data = pd.read_parquet(ROOT / "data/processed/ca1_foods3.parquet")
    train_df = prepare_statsforecast_df(data, "train")
    valid_df = prepare_statsforecast_df(data, "val")
    features = pd.read_csv(ROOT / "results" / "phan_loai_chuoi.csv")

    report_df = build_report_tables(train_df, valid_df, features)
    output_path = ROOT / "results" / "thong_ke_co_so_val.csv"
    report_df.to_csv(output_path, index=False)

    buffer = StringIO()
    print("NGUYEN NHAN:", file=buffer)
    print("- TSB bi thieu truoc do vi make_model_list() chi tao 3 mo hinh va khong luu model duoc chon vao CSV, nen CSV chi co WindowAverage, SeasonalNaive, CrostonSBA.", file=buffer)
    print("- Sau khi sua, script da chay lai va TSB duoc chon qua validation theo WAPE, roi luu vao thong_ke_co_so_val.csv.", file=buffer)
    print("", file=buffer)
    print("OVERALL", file=buffer)
    print(report_df[report_df["group"] == "overall"].to_string(index=False), file=buffer)
    print("", file=buffer)
    print("DEMAND_TYPE", file=buffer)
    print(report_df[report_df["group"].isin(features["demand_type"].unique())].sort_values(["model", "group"]).to_string(index=False), file=buffer)
    print("", file=buffer)
    print("SPEED", file=buffer)
    print(report_df[report_df["group"].isin(features["speed"].unique())].sort_values(["model", "group"]).to_string(index=False), file=buffer)
    print("", file=buffer)
    print("DEMAND_TYPE_X_SPEED", file=buffer)
    print(report_df[report_df["group"].str.contains("_", na=False)].sort_values(["model", "group"]).to_string(index=False), file=buffer)

    report_text = buffer.getvalue()
    (ROOT / "results" / "bao_cao_kiem_tra.txt").write_text(report_text, encoding="utf-8")
    print(report_text, end="")


if __name__ == "__main__":
    main()
