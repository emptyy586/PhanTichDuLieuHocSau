"""Mô hình xác suất cho tổng nhu cầu cửa sổ D_H(t) = y[t+1] + ... + y[t+H].

Thiết kế thống nhất (xem configs/config.yaml):
- Gốc dự báo t: chỉ dùng dữ liệu đến hết ngày t. Giá/SNAP/sự kiện của t+1..t+H là biến tương lai đã biết.
- Mô hình thống kê (WindowAverage, SeasonalNaive, Croston-SBA, TSB) cho trung bình cửa sổ M(t);
  phân vị lấy từ Negative Binomial (mean=M, alpha ước lượng cuốn chiếu từ phần dư QUÁ KHỨ).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from src.utils import ROOT, load_config

CONFIG = load_config(ROOT / "configs" / "config.yaml")
SEED = int(CONFIG["seed"])
QGRID = [float(q) for q in CONFIG["quantile_grid"]]
EXT_QGRID = [float(q) for q in CONFIG["quantile_grid_ext"]]  # lưới phụ q<0.5 (GĐ6), chỉ dùng cho mô phỏng
ALPHA_MIN = float(CONFIG["alpha_min"])
ALPHA_MAX = float(CONFIG["alpha_max"])
MU_FLOOR = 1e-3


# ----------------------------------------------------------------------------- dữ liệu
@dataclass
class Panel:
    """Dữ liệu dạng bảng [chuỗi, ngày] đã sắp xếp; chỉ số cột = chỉ số ngày."""

    ids: np.ndarray
    dates: pd.DatetimeIndex
    Y: np.ndarray  # [S, T] doanh số
    price: np.ndarray  # [S, T] giá bán, NaN nếu chưa niêm yết
    snap: np.ndarray  # [T]
    event_name: np.ndarray  # [T] object (NaN nếu không có)
    event_type: np.ndarray  # [T]
    weekday: np.ndarray  # [T] 0=Thứ 2
    month: np.ndarray  # [T]
    demand_type: np.ndarray  # [S]
    speed: np.ndarray  # [S]
    val_start: int
    test_start: int

    @property
    def n_series(self) -> int:
        return self.Y.shape[0]

    @property
    def n_days(self) -> int:
        return self.Y.shape[1]

    @property
    def group_ds(self) -> np.ndarray:
        return np.array([f"{a}_{b}" for a, b in zip(self.demand_type, self.speed)])


def load_panel(config: dict | None = None) -> Panel:
    """Đọc parquet đã xử lý và dựng Panel [S, T]; phân loại chuỗi lấy từ results/series_classes.csv (tính trên train)."""
    df = pd.read_parquet(ROOT / "data" / "processed" / "ca1_foods3.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    ids = df["id"].drop_duplicates().to_numpy()
    S = len(ids)
    T = len(df) // S
    assert S * T == len(df), "Mỗi chuỗi phải có cùng số ngày."
    dates = pd.DatetimeIndex(df["date"].iloc[:T])
    Y = df["y"].to_numpy(dtype=float).reshape(S, T)
    price = df["sell_price"].to_numpy(dtype=float).reshape(S, T)
    first = df.iloc[:T]
    meta = pd.read_csv(ROOT / "results" / "series_classes.csv").set_index("id").loc[ids]
    split = first["split"].astype(str).to_numpy()
    return Panel(
        ids=ids,
        dates=dates,
        Y=Y,
        price=price,
        snap=first["snap_CA"].fillna(0).to_numpy(dtype=float),
        event_name=first["event_name_1"].to_numpy(dtype=object),
        event_type=first["event_type_1"].to_numpy(dtype=object),
        weekday=dates.weekday.to_numpy(),
        month=dates.month.to_numpy(),
        demand_type=meta["demand_type"].to_numpy(),
        speed=meta["speed"].to_numpy(),
        val_start=int(np.argmax(split == "val")),
        test_start=int(np.argmax(split == "test")),
    )


def window_targets(Y: np.ndarray, H: int) -> np.ndarray:
    """W[:, t] = y[t+1] + ... + y[t+H]; NaN nếu cửa sổ vượt quá dữ liệu."""
    S, T = Y.shape
    C = np.concatenate([np.zeros((S, 1)), np.cumsum(Y, axis=1)], axis=1)
    W = np.full((S, T), np.nan)
    W[:, : T - H] = C[:, H + 1 : T + 1] - C[:, 1 : T - H + 1]
    return W


def split_bounds(panel: Panel, split: str) -> tuple[int, int]:
    """Trả về (chỉ số ngày đầu, chỉ số ngày cuối) của split val/test."""
    if split == "val":
        return panel.val_start, panel.test_start - 1
    if split == "test":
        return panel.test_start, panel.n_days - 1
    raise ValueError(f"split không hợp lệ: {split}")


def origins_for_split(panel: Panel, split: str, H: int) -> np.ndarray:
    """Các gốc t sao cho cửa sổ mục tiêu t+1..t+H nằm hoàn toàn trong split."""
    start, end = split_bounds(panel, split)
    return np.arange(start - 1, end - H + 1)


# ----------------------------------------------------------------------------- mức dự báo ngày của mô hình thống kê
def window_average_level(Y: np.ndarray, window: int = 28) -> np.ndarray:
    """L[:, t] = trung bình y trong `window` ngày kết thúc tại t (chỉ quá khứ)."""
    S, T = Y.shape
    C = np.concatenate([np.zeros((S, 1)), np.cumsum(Y, axis=1)], axis=1)
    t = np.arange(T)
    lo = np.maximum(t + 1 - window, 0)
    return (C[:, t + 1] - C[:, lo]) / (t + 1 - lo)


def croston_sba_level(Y: np.ndarray, alpha: float) -> np.ndarray:
    """Mức dự báo/ngày của Croston-SBA tại cuối ngày t (hiệu chỉnh Syntetos-Boylan 1 - alpha/2).

    Khởi tạo tại lần bán đầu tiên của từng chuỗi; trước đó dự báo 0.
    """
    S, T = Y.shape
    z = np.zeros(S)
    p = np.ones(S)
    since = np.zeros(S)
    started = np.zeros(S, dtype=bool)
    out = np.zeros((S, T))
    for t in range(T):
        y = Y[:, t]
        pos = y > 0
        new = pos & ~started
        upd = pos & started
        z = np.where(new, y, np.where(upd, z + alpha * (y - z), z))
        interval = since + 1.0
        p = np.where(new, 1.0, np.where(upd, p + alpha * (interval - p), p))
        since = np.where(pos, 0.0, np.where(started, since + 1.0, since))
        started = started | pos
        out[:, t] = np.where(started, (1.0 - alpha / 2.0) * z / p, 0.0)
    return out


def tsb_level(Y: np.ndarray, alpha_d: float, alpha_p: float) -> np.ndarray:
    """Mức dự báo/ngày của TSB = xác suất có nhu cầu × độ lớn nhu cầu (cập nhật xác suất mỗi ngày)."""
    S, T = Y.shape
    z = np.zeros(S)
    p = np.zeros(S)
    started = np.zeros(S, dtype=bool)
    out = np.zeros((S, T))
    for t in range(T):
        y = Y[:, t]
        pos = y > 0
        new = pos & ~started
        z = np.where(new, y, np.where(pos, z + alpha_d * (y - z), z))
        p = np.where(new, 1.0, np.where(started, p + alpha_p * (pos.astype(float) - p), p))
        started = started | pos
        out[:, t] = np.where(started, z * p, 0.0)
    return out


def seasonal_naive_window_mean(Y: np.ndarray, H: int, season: int = 7) -> np.ndarray:
    """M[:, t] = sum_{k=1..H} y[t+k-season] (cần H <= season); NaN nếu chưa đủ lịch sử."""
    S, T = Y.shape
    assert H <= season
    M = np.zeros((S, T))
    for k in range(1, H + 1):
        shift = season - k
        M[:, shift:] += Y[:, : T - shift]
    M[:, : season - 1] = np.nan
    return M


def stat_window_mean(model: str, Y: np.ndarray, H: int, params: dict) -> np.ndarray:
    """Ma trận trung bình cửa sổ M[S, T] của mô hình thống kê (dự báo tại gốc t, chỉ dùng y[:t+1])."""
    if model == "WindowAverage":
        return H * window_average_level(Y, int(params.get("window", 28)))
    if model == "SeasonalNaive":
        return seasonal_naive_window_mean(Y, H, int(params.get("season", 7)))
    if model == "CrostonSBA":
        return H * croston_sba_level(Y, float(params["alpha"]))
    if model == "TSB":
        return H * tsb_level(Y, float(params["alpha_d"]), float(params["alpha_p"]))
    raise ValueError(model)


# ----------------------------------------------------------------------------- Negative Binomial
def nb_quantiles(mu: np.ndarray, alpha: np.ndarray, qs: list[float]) -> np.ndarray:
    """Phân vị chính xác của NB(mean=mu, alpha): var = mu + alpha*mu^2. Trả về [..., len(qs)]."""
    mu = np.maximum(np.asarray(mu, dtype=float), MU_FLOOR)
    alpha = np.clip(np.asarray(alpha, dtype=float), ALPHA_MIN, ALPHA_MAX)
    n = (1.0 / alpha)[..., None]
    p = (1.0 / (1.0 + alpha * mu))[..., None]
    out = nbinom.ppf(np.asarray(qs, dtype=float), n, p)
    return np.nan_to_num(out, nan=0.0)


def _range_sums(A: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Tổng A[:, lo..hi] (bao gồm hai đầu) cho từng cặp (lo[k], hi[k]); trả về [S, K]."""
    C = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum(A, axis=1)], axis=1)
    return C[:, hi + 1] - C[:, lo]


def estimate_alpha(Y: np.ndarray, M: np.ndarray, H: int, origins: np.ndarray, groups: np.ndarray,
                   n_hist: int, w_series: float, kappa: float = 1.0) -> np.ndarray:
    """Ước lượng alpha NB cuốn chiếu theo moment: alpha = sum(e^2 - M) / sum(M^2).

    Tại gốc t chỉ dùng các gốc lịch sử s <= t-H (cửa sổ s+1..s+H đã quan sát xong), tối đa n_hist gốc.
    Trộn alpha của chuỗi với alpha của nhóm (demand_type x speed) theo w_series, rồi nhân kappa.
    Trả về [S, len(origins)].
    """
    W = window_targets(Y, H)
    known = ~np.isnan(W) & ~np.isnan(M)  # gốc lịch sử có dự báo M và cửa sổ đã quan sát xong
    Mf = np.nan_to_num(M)
    A = np.where(known, (np.nan_to_num(W) - Mf) ** 2 - Mf, 0.0)
    B = np.where(known, Mf ** 2, 0.0)
    hi = np.maximum(origins - H, 0)
    lo = np.maximum(origins - H - n_hist + 1, 0)
    SA = _range_sums(A, hi, lo)
    SB = _range_sums(B, hi, lo)
    uniq = np.unique(groups)
    GA = np.zeros_like(SA)
    GB = np.zeros_like(SB)
    alpha_g = np.zeros_like(SA)
    for g in uniq:
        m = groups == g
        ga, gb = SA[m].sum(axis=0), SB[m].sum(axis=0)
        a_g = np.where(gb > 1e-9, ga / np.maximum(gb, 1e-9), 1.0)
        alpha_g[m] = a_g[None, :]
    alpha_i = np.where(SB > 1e-9, SA / np.maximum(SB, 1e-9), alpha_g)
    alpha = w_series * alpha_i + (1.0 - w_series) * alpha_g
    return np.clip(alpha, ALPHA_MIN, ALPHA_MAX) * kappa


def stat_forecast(model: str, panel: Panel, H: int, origins: np.ndarray, params: dict,
                  kappa: float, w_series: float, n_hist: int | None = None) -> dict[str, np.ndarray]:
    """Dự báo xác suất của một mô hình thống kê tại các gốc `origins`: mean [S,n], alpha [S,n], Q [S,n,nq]."""
    n_hist = int(CONFIG["alpha_hist_window"] if n_hist is None else n_hist)
    M = stat_window_mean(model, panel.Y, H, params)
    alpha = estimate_alpha(panel.Y, M, H, origins, panel.group_ds, n_hist, w_series, kappa)
    mean = np.nan_to_num(M[:, origins])
    return {"mean": mean, "alpha": alpha, "Q": nb_quantiles(mean, alpha, QGRID), "Qext": nb_quantiles(mean, alpha, EXT_QGRID)}


# ----------------------------------------------------------------------------- chỉ số
def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    """Pinball loss từng phần tử: max(q*e, (q-1)*e), e = actual - pred."""
    e = actual - pred
    return np.maximum(q * e, (q - 1.0) * e)


def mase_scale(panel: Panel, H: int) -> np.ndarray:
    """Hệ số MASE theo chuỗi: MAE trong mẫu (CHỈ train) của dự báo naive-theo-cửa-sổ W(s) ≈ W(s-H)."""
    W = window_targets(panel.Y, H)
    first = np.argmax(panel.Y > 0, axis=1)
    last_s = panel.val_start - 1 - H  # cửa sổ s+1..s+H nằm trong train
    scale = np.full(panel.n_series, np.nan)
    for i in range(panel.n_series):
        s = np.arange(first[i] + H, last_s + 1)
        if len(s) > 0:
            scale[i] = np.mean(np.abs(W[i, s] - W[i, s - H]))
    return scale


def group_masks(panel: Panel) -> list[tuple[str, str, np.ndarray]]:
    """Danh sách (group_type, group, mask chuỗi): overall, speed, demand_type, demand_type_x_speed."""
    out = [("overall", "overall", np.ones(panel.n_series, dtype=bool))]
    for gt, arr in [("speed", panel.speed), ("demand_type", panel.demand_type), ("demand_type_x_speed", panel.group_ds)]:
        for g in sorted(np.unique(arr)):
            out.append((gt, str(g), arr == g))
    return out


def evaluate_forecast(model: str, H: int, panel: Panel, origins: np.ndarray, fc: dict[str, np.ndarray]) -> pd.DataFrame:
    """Bảng chỉ số dạng dài (model,H,group_type,group,metric,q,value) cho một mô hình tại một H.

    Chỉ số: pinball(q), coverage(q)=P(y<=Q), coverage_lt(q)=P(y<Q) (với phân phối rời rạc hiệu chuẩn tốt thì coverage_lt <= q <= coverage), pinball_avg (trung bình mọi q), wape (trên trung bình cửa sổ), mase, n_series.
    """
    W = window_targets(panel.Y, H)[:, origins]
    Q, mean = fc["Q"], fc["mean"]
    scale = mase_scale(panel, H)
    rows = []
    pin = np.stack([pinball_loss(W, Q[..., k], q) for k, q in enumerate(QGRID)], axis=-1)  # [S,n,nq]
    cov = np.stack([(W <= Q[..., k]).astype(float) for k in range(len(QGRID))], axis=-1)
    cov_lt = np.stack([(W < Q[..., k]).astype(float) for k in range(len(QGRID))], axis=-1)
    ae = np.abs(W - mean)
    for gt, g, m in group_masks(panel):
        base = {"model": model, "H": H, "group_type": gt, "group": g}
        for k, q in enumerate(QGRID):
            rows.append({**base, "metric": "pinball", "q": q, "value": float(pin[m][..., k].mean())})
            rows.append({**base, "metric": "coverage", "q": q, "value": float(cov[m][..., k].mean())})
            rows.append({**base, "metric": "coverage_lt", "q": q, "value": float(cov_lt[m][..., k].mean())})
        rows.append({**base, "metric": "pinball_avg", "q": np.nan, "value": float(pin[m].mean())})
        denom = W[m].sum()
        rows.append({**base, "metric": "wape", "q": np.nan, "value": float(ae[m].sum() / denom) if denom > 0 else np.nan})
        ok = m & np.isfinite(scale) & (scale > 0)
        mase_i = ae[ok].mean(axis=1) / scale[ok]
        rows.append({**base, "metric": "mase", "q": np.nan, "value": float(mase_i.mean()) if len(mase_i) else np.nan})
        rows.append({**base, "metric": "n_series", "q": np.nan, "value": float(m.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tinh chỉnh siêu tham số (trên VAL)
def avg_pinball(panel: Panel, Hs: list[int], origins_by_H: dict[int, np.ndarray], fc_by_H: dict[int, dict]) -> float:
    """Pinball trung bình trên mọi q, mọi H (tiêu chí chọn siêu tham số)."""
    vals = []
    for H in Hs:
        W = window_targets(panel.Y, H)[:, origins_by_H[H]]
        Q = fc_by_H[H]["Q"]
        vals.append(np.mean([pinball_loss(W, Q[..., k], q).mean() for k, q in enumerate(QGRID)]))
    return float(np.mean(vals))
