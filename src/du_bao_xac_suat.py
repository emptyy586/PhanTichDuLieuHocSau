from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import TSB, WindowAverage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tien_ich import ROOT as PROJECT_ROOT, load_config, set_seed

ROOT = PROJECT_ROOT

CONFIG = load_config(ROOT / "configs" / "thong_so.yaml")
SEED = int(CONFIG["seed"])


def _as_float_list(values: Iterable[float] | None) -> list[float]:
    if values is None:
        return [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    return [float(v) for v in values]


def estimate_nb_params(point_mean: float, residuals: pd.Series | np.ndarray | None, fallback_var: float | None = None) -> tuple[str, float, float]:
    """Ước lượng tham số NB hoặc Poisson từ trung bình dự báo và phần dư lịch sử."""
    mu = float(max(point_mean, 1e-8))
    if residuals is not None:
        arr = np.asarray(residuals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 1:
            var = float(np.var(arr, ddof=1))
        elif arr.size == 1:
            var = float(abs(arr[0])) + 1e-6
        else:
            var = mu
    else:
        var = float(fallback_var if fallback_var is not None else mu)

    if not np.isfinite(var) or var <= 0:
        var = mu

    cap_var = max(mu, 1.0) * 25.0
    var = min(var, cap_var)

    if mu <= 1e-6 or var <= mu:
        return "poisson", mu, mu

    alpha = max((var - mu) / (mu ** 2), 1e-6)
    alpha = min(alpha, 1e2)
    n = 1.0 / alpha
    p = n / (n + mu)
    return "nb", max(n, 1e-6), max(p, 1e-6)


def sample_total_window(mean: float, variance: float, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mẫu tổng nhu cầu trong cửa sổ theo NB hoặc Poisson."""
    mean = float(max(mean, 0.0))
    if mean <= 1e-8:
        return np.zeros(n_samples, dtype=float)
    variance = float(max(variance, mean, 1e-8))
    variance = min(variance, max(mean, 1.0) * 25.0)
    rng = np.random.default_rng(SEED if rng is None else rng.integers(0, 10_000_000))
    if variance <= mean or not np.isfinite(variance):
        return rng.poisson(lam=max(mean, 0.0), size=n_samples).astype(float)

    mu_safe = max(mean, 1e-8)
    alpha = max((variance - mean) / (mu_safe ** 2), 1e-6)
    alpha = min(alpha, 1e2)
    n = 1.0 / alpha
    p = n / (n + mu_safe)
    # Gamma-Poisson: NB chính xác với mọi n > 0 (kể cả n < 1; trước đây ép n >= 1 làm trung bình bị phóng đại tới ~24 lần)
    lam = rng.gamma(shape=n, scale=mu_safe / n, size=n_samples)
    return rng.poisson(lam).astype(float)


def build_prob_quantiles(mean: float, variance: float, q_levels: list[float] | None = None, n_samples: int | None = None) -> dict[str, float]:
    q_levels = _as_float_list(q_levels)
    n_samples = int(CONFIG.get("n_samples", 500) if n_samples is None else n_samples)
    draws = sample_total_window(mean, variance, n_samples)
    qvals = np.quantile(draws, q_levels)
    return {f"q{q:.2f}": float(v) for q, v in zip(q_levels, qvals)}


def _point_and_distribution_for_history(history: pd.Series | np.ndarray, q_levels: list[float], n_samples: int | None = None) -> tuple[float, float, str, float, float, dict[str, float]]:
    arr = pd.Series(history).astype(float).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, "poisson", 0.0, 0.0, {f"q{q:.2f}": 0.0 for q in _as_float_list(q_levels)}

    point = float(np.mean(arr[-28:])) if arr.size >= 1 else 0.0
    residuals = arr - point
    var = float(np.var(residuals, ddof=1)) if len(residuals) > 1 else max(float(np.abs(residuals).sum()) if len(residuals) else 0.0, point, 1.0)
    dist_kind, n, p = estimate_nb_params(point, residuals, fallback_var=var)
    dist = build_prob_quantiles(point, max(var, point), q_levels=q_levels, n_samples=n_samples)
    return point, max(var, point), dist_kind, n, p, dist


def train_lightgbm_quantile(df: pd.DataFrame, target_col: str, feature_cols: list[str], q_levels: list[float]) -> dict[float, lgb.LGBMRegressor]:
    """Train một mô hình LightGBM cho từng quantile, dùng objective=quantile."""
    models: dict[float, lgb.LGBMRegressor] = {}
    for q in q_levels:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            learning_rate=float(CONFIG.get("lightgbm_learning_rate", 0.05)),
            num_leaves=int(CONFIG.get("lightgbm_num_leaves", 31)),
            n_estimators=int(CONFIG.get("lightgbm_n_estimators", 200)),
            random_state=SEED,
            subsample=0.8,
            colsample_bytree=0.9,
            verbosity=-1,
        )
        model.fit(df[feature_cols], df[target_col])
        models[q] = model
    return models


def build_lag_features(data: pd.DataFrame, q_levels: list[float] | None = None) -> pd.DataFrame:
    """[Legacy – GĐ2 dùng src/mo_hinh_ml.build_features] Tạo đặc trưng lag/rolling, giá, thứ, tháng. Chỉ dùng quá khứ.

    Đã bỏ các cột roll_mean_{lag} (trước đây đều bằng y dịch 1 ngày, trùng lặp, không phải trung bình trượt)."""
    q_levels = _as_float_list(q_levels)
    out = data.sort_values(["id", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    for lag in [7, 14, 28]:
        out[f"lag_{lag}"] = out.groupby("id")["y"].shift(lag)
    for lag in [7, 14, 28]:
        out[f"rolling_mean_{lag}"] = out.groupby("id")["y"].transform(lambda s: s.shift(1).rolling(window=lag, min_periods=1).mean())
        out[f"rolling_std_{lag}"] = out.groupby("id")["y"].transform(lambda s: s.shift(1).rolling(window=lag, min_periods=1).std())
    out["price_change"] = out.groupby("id")["sell_price"].diff()
    out["price_pct_change"] = out.groupby("id")["sell_price"].pct_change().fillna(0.0)
    out["weekday"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month
    out["is_snap"] = out["snap_CA"].fillna(0).astype(float)
    out["event_flag"] = out["event_name_1"].notna().astype(int)
    out["y"] = out["y"].astype(float)
    out["target"] = out["y"]
    for col in ["lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_7", "rolling_std_14", "rolling_std_28", "price_change", "price_pct_change", "weekday", "month", "is_snap", "event_flag", "sell_price"]:
        out[col] = out[col].fillna(0.0)
    out = out.replace([np.inf, -np.inf], 0.0)
    return out


def prepare_split_features(data: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn bị features và nhãn cho validation; chỉ dùng dữ liệu đến t-1."""
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    series_meta = pd.read_csv(ROOT / "results" / "phan_loai_chuoi.csv")
    if {"id", "demand_type", "speed"}.issubset(series_meta.columns):
        data = data.merge(series_meta[["id", "demand_type", "speed"]], on="id", how="left")
    return build_lag_features(data)


def make_baseline_forecast(train_df: pd.DataFrame) -> pd.DataFrame:
    """Trả về dự báo điểm bằng WindowAverage/TSB đơn giản cho toàn bộ validation trên cột y."""
    train_df = train_df.sort_values(["unique_id", "ds"]).copy()
    date_index = train_df["ds"].drop_duplicates().sort_values()
    out = []
    for unique_id, group in train_df.groupby("unique_id"):
        series = group.sort_values("ds").copy()
        hist = series["y"].to_numpy(dtype=float)
        point = float(np.nanmean(hist[-28:])) if len(hist) >= 28 else float(np.nanmean(hist))
        residuals = hist - point
        for ds in date_index:
            out.append({"unique_id": unique_id, "ds": ds, "point_mean": point, "residual_var": float(np.var(residuals, ddof=1)) if len(residuals) > 1 else max(point, 1.0)})
    return pd.DataFrame(out)


def rolling_origin_eval(data: pd.DataFrame, split: str = "val", n_samples: int | None = None) -> pd.DataFrame:
    """[Legacy] Dự báo baseline NB cũ. LƯU Ý: chỉ dùng lịch sử split=="train" nên KHÔNG phải rolling-origin thật.
    Dùng src/chay_du_bao.py (gốc t dùng dữ liệu đến hết ngày t) cho kết quả chính thức."""
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    q_levels = _as_float_list(CONFIG.get("quantile_levels"))
    L = int(CONFIG.get("lead_time", 2))
    R = int(CONFIG.get("review_period", 1))
    horizon = L + R
    n_samples = int(CONFIG.get("n_samples", 500) if n_samples is None else n_samples)
    rows = []
    val_dates = pd.Series(data.loc[data["split"] == split, "date"].sort_values().unique())

    for origin in val_dates:
        origin_ts = pd.Timestamp(origin)
        for unique_id, g in data[(data["split"] == "train")].groupby("id"):
            hist = g[g["date"] < origin_ts]["y"].astype(float)
            if hist.empty:
                continue
            point, var, dist_kind, n, p, daily_dist = _point_and_distribution_for_history(hist.to_numpy(), q_levels=q_levels, n_samples=n_samples)
            daily_target_date = origin_ts + pd.to_timedelta(horizon, unit="D")
            for q, value in daily_dist.items():
                rows.append({
                    "id": unique_id,
                    "origin_date": origin_ts,
                    "target_date": daily_target_date,
                    "level": "daily",
                    "horizon": 1,
                    "model": "baseline_nb",
                    "dist": dist_kind,
                    "q": float(q.replace("q", "")),
                    "quantile_value": value,
                    "point_mean": point,
                    "dist_n": n,
                    "dist_p": p,
                })

            window_mean = point * horizon
            window_var = max(var, point) * horizon
            window_dist = build_prob_quantiles(window_mean, window_var, q_levels=q_levels, n_samples=n_samples)
            for q, value in window_dist.items():
                rows.append({
                    "id": unique_id,
                    "origin_date": origin_ts,
                    "target_date": daily_target_date,
                    "level": "window",
                    "horizon": horizon,
                    "model": "baseline_nb",
                    "dist": dist_kind,
                    "q": float(q.replace("q", "")),
                    "quantile_value": value,
                    "point_mean": point,
                    "dist_n": n,
                    "dist_p": p,
                })
    return pd.DataFrame(rows)


def export_validation_outputs(data: pd.DataFrame, split: str = "val", n_samples: int | None = None) -> tuple[pd.DataFrame, Path]:
    """Xuất file đánh giá và file window quantile parquet chuẩn để dùng ở kết quả cuối."""
    forecast_df = rolling_origin_eval(data, split=split, n_samples=n_samples)
    summary_df = evaluate_validation_summary(data, forecast_df)
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "danh_gia_du_bao_val.csv"
    summary_df.to_csv(csv_path, index=False)

    window_df = forecast_df[forecast_df["level"] == "window"][ ["id", "origin_date", "q", "quantile_value"] ].copy()
    window_df = window_df.rename(columns={"origin_date": "date"})
    parquet_path = output_dir / "phan_vi_cua_so_val.parquet"
    window_df.to_parquet(parquet_path, index=False)
    return summary_df, csv_path


def evaluate_daily_summary(data: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Đánh giá theo ngày: phân vị 1 ngày vs actual 1 ngày."""
    q_levels = _as_float_list(CONFIG.get("quantile_levels"))
    val = data[data["split"] == "val"].copy()
    val["date"] = pd.to_datetime(val["date"])
    forecast_eval = forecast_df.copy()
    if "level" in forecast_eval.columns:
        forecast_eval = forecast_eval[forecast_eval["level"] == "daily"].copy()
    daily = forecast_eval[["id", "target_date", "q", "quantile_value"]].rename(columns={"target_date": "date"}).drop_duplicates(subset=["id", "date", "q"]).copy()
    daily["date"] = pd.to_datetime(daily["date"])
    merged = val[["id", "date", "y"]].merge(daily, on=["id", "date"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["model", "level", "group_type", "group", "metric", "q", "value"])
    out_rows = []
    for q in q_levels:
        sub = merged[merged["q"] == q]
        if sub.empty:
            continue
        actual = sub["y"].to_numpy(dtype=float)
        pred = sub["quantile_value"].to_numpy(dtype=float)
        abs_err = np.abs(actual - pred)
        abs_y = np.abs(actual)
        wape = float(np.sum(abs_err) / np.sum(abs_y)) if np.sum(abs_y) > 0 else np.nan
        error = actual - pred
        pinball = np.maximum(q * error, (q - 1.0) * error)
        coverage = float(np.mean(actual <= pred))
        out_rows.append({"model": "baseline_nb", "level": "daily", "group_type": "overall", "group": "overall", "metric": "wape", "q": q, "value": wape})
        out_rows.append({"model": "baseline_nb", "level": "daily", "group_type": "overall", "group": "overall", "metric": "pinball", "q": q, "value": float(np.mean(pinball))})
        out_rows.append({"model": "baseline_nb", "level": "daily", "group_type": "overall", "group": "overall", "metric": "coverage", "q": q, "value": coverage})
    return pd.DataFrame(out_rows)


def evaluate_window_summary(data: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Đánh giá theo cửa sổ: tổng dữ liệu L+R ngày vs tổng phân vị cửa sổ."""
    q_levels = _as_float_list(CONFIG.get("quantile_levels"))
    val = data[data["split"] == "val"].copy()
    val["date"] = pd.to_datetime(val["date"])
    L = int(CONFIG.get("lead_time", 2))
    R = int(CONFIG.get("review_period", 1))
    horizon = L + R
    forecast_eval = forecast_df.copy()
    if "level" in forecast_eval.columns:
        forecast_eval = forecast_eval[forecast_eval["level"] == "window"].copy()
    window_forecast = forecast_eval[["id", "origin_date", "q", "quantile_value"]].drop_duplicates(subset=["id", "origin_date", "q"]).copy()
    window_forecast["origin_date"] = pd.to_datetime(window_forecast["origin_date"])
    out_rows = []
    actual_rows = []
    for origin in sorted(val["date"].unique()):
        window_dates = pd.date_range(start=pd.Timestamp(origin) + pd.to_timedelta(1, unit="D"), periods=horizon, freq="D")
        for unique_id, group in val.groupby("id"):
            actual_total = group[group["date"].isin(window_dates)]["y"].sum()
            if pd.isna(actual_total):
                continue
            actual_rows.append({"id": unique_id, "origin_date": pd.Timestamp(origin), "actual_window_total": float(actual_total)})
    actual_window = pd.DataFrame(actual_rows)
    merged = actual_window.merge(window_forecast, on=["id", "origin_date"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["model", "level", "group_type", "group", "metric", "q", "value"])
    for q in q_levels:
        sub = merged[merged["q"] == q]
        if sub.empty:
            continue
        actual = sub["actual_window_total"].to_numpy(dtype=float)
        pred = sub["quantile_value"].to_numpy(dtype=float)
        abs_err = np.abs(actual - pred)
        abs_y = np.abs(actual)
        wape = float(np.sum(abs_err) / np.sum(abs_y)) if np.sum(abs_y) > 0 else np.nan
        error = actual - pred
        pinball = np.maximum(q * error, (q - 1.0) * error)
        coverage = float(np.mean(actual <= pred))
        out_rows.append({"model": "baseline_nb", "level": "window", "group_type": "overall", "group": "overall", "metric": "wape", "q": q, "value": wape})
        out_rows.append({"model": "baseline_nb", "level": "window", "group_type": "overall", "group": "overall", "metric": "pinball", "q": q, "value": float(np.mean(pinball))})
        out_rows.append({"model": "baseline_nb", "level": "window", "group_type": "overall", "group": "overall", "metric": "coverage", "q": q, "value": coverage})
    return pd.DataFrame(out_rows)


def evaluate_validation_summary(data: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp đồng thời cả daily và window trong schema chuẩn."""
    daily = evaluate_daily_summary(data, forecast_df)
    window = evaluate_window_summary(data, forecast_df)
    return pd.concat([daily, window], ignore_index=True)


def run_smoke_test() -> Path:
    """Chạy một kiểm tra nhanh để đảm bảo code syntax và tạo file kết quả mới dưới results/ mà không chạy mô hình nặng."""
    data = pd.read_parquet(ROOT / "data/processed/ca1_foods3.parquet")
    data = data[data["id"].isin(data["id"].drop_duplicates().head(20))].copy()
    model_forecast = rolling_origin_eval(data, split="val")
    eval_df = evaluate_validation_summary(data, model_forecast)
    out_path = ROOT / "results" / "danh_gia_du_bao_val_smoke.csv"
    eval_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline probabilistic forecast helpers for M5 stage A")
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke test only.")
    args = parser.parse_args()
    if args.smoke:
        out_path = run_smoke_test()
        print(f"SMOKE_OK {out_path}")
        return

    data = pd.read_parquet(ROOT / "data/processed/ca1_foods3.parquet")
    summary_df, csv_path = export_validation_outputs(data, split="val")
    print(f"EXPORT_OK {csv_path}")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
