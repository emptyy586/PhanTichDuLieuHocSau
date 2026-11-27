"""Đặc trưng không rò rỉ + LightGBM (phân vị trực tiếp, Poisson) + MLP-NB cho tổng nhu cầu cửa sổ D_H(t).

Đặc trưng tại gốc t chỉ dùng y[:t+1]; giá/SNAP/sự kiện/thứ của t+1..t+H là biến tương lai đã biết.
Các "chỉ số hiệu ứng" (thứ trong tuần, sự kiện) tính CHỈ trên các ngày ≤ `effect_end` (hết train khi chạy val;
hết val khi chạy test) nên không dùng thông tin của giai đoạn đang dự báo.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.mo_hinh_xac_suat import (CONFIG, QGRID, SEED, Panel, croston_sba_level, nb_quantiles, tsb_level, window_targets,
                             window_average_level)

EXT_QGRID = [float(q) for q in CONFIG["quantile_grid_ext"]]
SPEED_CODE = {"slow": 0, "medium": 1, "fast": 2}
DTYPE_CODE = {"smooth": 0, "erratic": 1, "intermittent": 2, "lumpy": 3}


def effect_tables(panel: Panel, effect_end: int) -> tuple[np.ndarray, dict]:
    """Chỉ số doanh số theo thứ (7 giá trị) và theo tên sự kiện, tính trên ngày [0, effect_end] sau ngày ra mắt."""
    Y = panel.Y[:, : effect_end + 1]
    first = np.argmax(panel.Y > 0, axis=1)
    active = np.arange(Y.shape[1])[None, :] >= first[:, None]
    tot_mean = Y[active].mean()
    wd = np.ones(7)
    for d in range(7):
        m = active & (panel.weekday[: effect_end + 1] == d)[None, :]
        wd[d] = Y[m].mean() / tot_mean
    ev: dict = {}
    names = panel.event_name[: effect_end + 1]
    for name in pd.unique(names[pd.notna(names)]):
        m = active & (names == name)[None, :]
        ev[name] = Y[m].mean() / tot_mean if m.any() else 1.0
    return wd, ev


def build_features(panel: Panel, H: int, origins: np.ndarray, effect_end: int) -> tuple[np.ndarray, list[str]]:
    """Ma trận đặc trưng [S, n_origins, F] tại các gốc `origins`. Chỉ dùng y[:t+1] và biến tương lai đã biết."""
    Y, price = panel.Y, panel.price
    S, T = Y.shape
    t = np.asarray(origins)
    C = np.concatenate([np.zeros((S, 1)), np.cumsum(Y, axis=1)], axis=1)
    C2 = np.concatenate([np.zeros((S, 1)), np.cumsum(Y ** 2, axis=1)], axis=1)
    NZ = np.concatenate([np.zeros((S, 1)), np.cumsum((Y > 0).astype(float), axis=1)], axis=1)
    feats: dict[str, np.ndarray] = {}

    def roll(Cm: np.ndarray, w: int) -> np.ndarray:
        lo = np.maximum(t + 1 - w, 0)
        return (Cm[:, t + 1] - Cm[:, lo]) / (t + 1 - lo)[None, :]

    for w in (7, 14, 28, 56, 180):
        feats[f"mean_{w}"] = roll(C, w)
    m28, s28 = roll(C, 28), roll(C2, 28)
    feats["std_28"] = np.sqrt(np.maximum(s28 - m28 ** 2, 0.0))
    feats["nz_rate_28"] = roll(NZ, 28)
    feats["nz_rate_90"] = roll(NZ, 90)
    feats["y_t"] = Y[:, t]
    feats["y_t_1"] = Y[:, np.maximum(t - 1, 0)]
    for lag in (7, 14, 21, 28):  # tổng nhu cầu cùng thứ của các tuần trước: sum_k y[t+k-lag]
        acc = np.zeros((S, len(t)))
        for k in range(1, H + 1):
            idx = t + k - lag
            acc += np.where(idx[None, :] >= 0, Y[:, np.maximum(idx, 0)], 0.0)
        feats[f"same_wd_lag{lag}"] = acc
    idx_grid = np.arange(T)[None, :]
    last_pos = np.maximum.accumulate(np.where(Y > 0, idx_grid, -1), axis=1)
    first_pos = np.argmax(Y > 0, axis=1)
    lp = last_pos[:, t]
    feats["days_since_sale"] = np.where(lp >= 0, np.minimum(t[None, :] - lp, 365), 365)
    age = t[None, :] - first_pos[:, None]
    feats["age_days"] = np.where(age >= 0, np.minimum(age, 1900), -1)
    feats["croston_level"] = croston_sba_level(Y, 0.1)[:, t]
    feats["tsb_level"] = tsb_level(Y, 0.1, 0.05)[:, t]
    feats["wavg28_level"] = window_average_level(Y, 28)[:, t]

    # biến tương lai đã biết (t+1..t+H)
    wd_idx, ev_idx = effect_tables(panel, effect_end)
    day_wd = wd_idx[panel.weekday]
    day_ev = np.array([ev_idx.get(n, 1.0) if pd.notna(n) else 1.0 for n in panel.event_name])
    day_has_ev = pd.notna(panel.event_name).astype(float)
    cs = lambda v: np.concatenate([[0.0], np.cumsum(v)])  # noqa: E731
    win = lambda v: (cs(v)[t + H + 1] - cs(v)[t + 1])  # noqa: E731
    feats["fut_wd_index"] = np.broadcast_to(win(day_wd) / H, (S, len(t))).copy()
    feats["fut_wd_next"] = np.broadcast_to(panel.weekday[np.minimum(t + 1, T - 1)].astype(float), (S, len(t))).copy()
    feats["fut_month"] = np.broadcast_to(panel.month[np.minimum(t + 1, T - 1)].astype(float), (S, len(t))).copy()
    feats["fut_snap_days"] = np.broadcast_to(win(panel.snap), (S, len(t))).copy()
    feats["fut_event_days"] = np.broadcast_to(win(day_has_ev), (S, len(t))).copy()
    feats["fut_event_index"] = np.broadcast_to(win(day_ev) / H, (S, len(t))).copy()
    # giá (NaN giữ nguyên): giá tương lai đã biết
    pc = np.concatenate([np.zeros((S, 1)), np.cumsum(np.nan_to_num(price), axis=1)], axis=1)
    pn = np.concatenate([np.zeros((S, 1)), np.cumsum(~np.isnan(price), axis=1)], axis=1)
    cnt_f = pn[:, t + H + 1] - pn[:, t + 1]
    fut_price = np.where(cnt_f > 0, (pc[:, t + H + 1] - pc[:, t + 1]) / np.maximum(cnt_f, 1), np.nan)
    lo28 = np.maximum(t + 1 - 28, 0)
    cnt_p = pn[:, t + 1] - pn[:, lo28]
    past_price = np.where(cnt_p > 0, (pc[:, t + 1] - pc[:, lo28]) / np.maximum(cnt_p, 1), np.nan)
    feats["price_fut"] = fut_price
    feats["price_ratio_fut_past28"] = fut_price / past_price
    feats["price_ratio_next_today"] = price[:, np.minimum(t + 1, T - 1)] / price[:, t]
    feats["speed_code"] = np.broadcast_to(np.array([SPEED_CODE[s] for s in panel.speed], dtype=float)[:, None], (S, len(t))).copy()
    feats["dtype_code"] = np.broadcast_to(np.array([DTYPE_CODE[s] for s in panel.demand_type], dtype=float)[:, None], (S, len(t))).copy()
    names = list(feats.keys())
    X = np.stack([np.asarray(feats[n], dtype=float) for n in names], axis=-1)
    X[~np.isfinite(X)] = np.nan
    return X, names


def training_origins(panel: Panel, H: int, last_label_day: int, n_days: int | None = None) -> np.ndarray:
    """Gốc huấn luyện: cửa sổ t+1..t+H kết thúc ≤ last_label_day; lấy `n_days` gốc gần nhất."""
    n_days = int(CONFIG["ml_train_days"] if n_days is None else n_days)
    hi = last_label_day - H
    return np.arange(max(hi - n_days + 1, 28), hi + 1)


def make_training_set(panel: Panel, H: int, last_label_day: int, effect_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Trả về (X [N,F], y [N], origin_of_row [N], names); bỏ các dòng chuỗi chưa có lần bán nào tại gốc."""
    org = training_origins(panel, H, last_label_day)
    X, names = build_features(panel, H, org, effect_end)
    y = window_targets(panel.Y, H)[:, org]
    keep = X[..., names.index("age_days")] >= 0
    org_grid = np.broadcast_to(org[None, :], keep.shape)
    return X[keep], y[keep], org_grid[keep], names


# ----------------------------------------------------------------------------- LightGBM
def _lgb_params(objective: str, alpha: float | None = None) -> dict:
    p = dict(
        objective=objective,
        learning_rate=float(CONFIG["lightgbm_learning_rate"]),
        num_leaves=int(CONFIG["lightgbm_num_leaves"]),
        n_estimators=int(CONFIG["lightgbm_n_estimators"]),
        min_child_samples=int(CONFIG["lightgbm_min_child_samples"]),
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    if alpha is not None:
        p["alpha"] = alpha
    return p


def fit_lgbm(X: np.ndarray, y: np.ndarray, extra_q: list[float] | None = None) -> dict:
    """Huấn luyện LightGBM trực tiếp trên D_H: 1 mô hình / phân vị (objective=quantile) + 1 mô hình Poisson.
    `extra_q`: các phân vị phụ (lưới mở rộng q<0.5), lưu riêng trong models["q_ext"], không ảnh hưởng kết quả chính."""
    models = {"q": {}, "q_ext": {}, "poisson": None}
    for q in QGRID:
        models["q"][q] = lgb.LGBMRegressor(**_lgb_params("quantile", q)).fit(X, y)
    for q in (extra_q or []):
        models["q_ext"][q] = lgb.LGBMRegressor(**_lgb_params("quantile", q)).fit(X, y)
    models["poisson"] = lgb.LGBMRegressor(**_lgb_params("poisson")).fit(X, y)
    return models


def predict_lgbm(models: dict, X: np.ndarray, shape: tuple[int, int], no_sale: np.ndarray) -> dict[str, np.ndarray]:
    """Dự báo [S,n]: Q (đã sắp tăng theo q để tránh cắt nhau, không âm) và mean Poisson; chuỗi chưa bán → 0."""
    S, n = shape
    Xf = X.reshape(S * n, -1)
    Q = np.stack([models["q"][q].predict(Xf) for q in QGRID], axis=-1)
    Q = np.maximum.accumulate(np.maximum(Q, 0.0), axis=-1).reshape(S, n, len(QGRID))
    mean = np.maximum(models["poisson"].predict(Xf), 0.0).reshape(S, n)
    Q[no_sale] = 0.0
    mean[no_sale] = 0.0
    out = {"Q": Q, "mean": mean}
    if models.get("q_ext"):
        ext_q = sorted(models["q_ext"])
        Qe = np.stack([models["q_ext"][q].predict(Xf) for q in ext_q], axis=-1)
        Qe = np.maximum(Qe, 0.0).reshape(S, n, len(ext_q))
        Qe = np.minimum.accumulate(np.minimum(Qe, Q[..., :1])[..., ::-1], axis=-1)[..., ::-1]  # không vượt phân vị chính đầu tiên, tăng dần
        Qe[no_sale] = 0.0
        out["Qext"] = np.maximum.accumulate(Qe, axis=-1)
    return out


# ----------------------------------------------------------------------------- MLP-NB
def _transform(X: np.ndarray, names: list[str], stats: dict | None) -> tuple[np.ndarray, dict]:
    """log1p cho đặc trưng mức/đếm, thay NaN bằng 0 + cờ thiếu cho giá, chuẩn hóa z-score theo thống kê train."""
    X = X.copy()
    price_cols = [i for i, n in enumerate(names) if n.startswith("price")]
    miss = np.isnan(X[:, price_cols]).astype(np.float32)
    log_cols = [i for i, n in enumerate(names) if n.startswith(("mean_", "std_", "y_t", "same_wd", "days_since", "age_days", "croston", "tsb", "wavg"))]
    X[:, log_cols] = np.log1p(np.clip(X[:, log_cols], 0, None))
    for i in price_cols:
        X[:, i] = np.where(np.isnan(X[:, i]), 0.0, np.log(np.clip(X[:, i], 1e-6, None)) if names[i] == "price_fut" else X[:, i] - 1.0)
    X = np.nan_to_num(X)
    X = np.concatenate([X, miss], axis=1)
    if stats is None:
        stats = {"mu": X.mean(axis=0), "sd": X.std(axis=0) + 1e-6}
    return ((X - stats["mu"]) / stats["sd"]).astype(np.float32), stats


def _nb_nll(torch, y, mu, alpha):
    """Âm log-hợp lý NB(mean=mu, var=mu+alpha*mu^2)."""
    r = 1.0 / alpha
    return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
             + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu))).mean()


def build_mlp(torch, n_in: int):
    """MLP hai đầu ra: log-hệ số nhân của mean (so với nền mean_28*H) và alpha."""
    hidden = list(CONFIG["mlp_hidden"])
    layers, d = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(float(CONFIG["mlp_dropout"]))]
        d = h
    layers += [torch.nn.Linear(d, 2)]
    return torch.nn.Sequential(*layers)


def _mlp_forward(torch, net, x, base):
    o = net(x)
    mu = base * torch.exp(torch.clamp(o[:, 0], -4.0, 4.0))
    alpha = torch.nn.functional.softplus(o[:, 1]) + 1e-3
    return mu, alpha


def fit_mlp_nb(Xtr: np.ndarray, ytr: np.ndarray, org_tr: np.ndarray, names: list[str], H: int,
               fixed_epochs: int | None = None, seed: int | None = None, max_epochs: int | None = None) -> dict:
    """Huấn luyện MLP-NB bằng NLL trực tiếp trên D_H. Chọn số epoch bằng tập kiểm tra nội bộ = các gốc cuối của train
    (có purge H ngày); nếu `fixed_epochs` cho trước thì dùng đúng số epoch đó (huấn luyện lại trên train+val)."""
    import torch

    seed = SEED if seed is None else int(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(seed)
    n_hold = int(CONFIG["mlp_holdout_origins"])
    last = org_tr.max()
    if fixed_epochs is None:
        hold = org_tr > last - n_hold
        tr = org_tr <= last - n_hold - H
    else:
        hold = np.zeros_like(org_tr, dtype=bool)
        tr = np.ones_like(org_tr, dtype=bool)
    Z, stats = _transform(Xtr[tr], names, None)
    base_all = lambda X: np.maximum(X[:, names.index("mean_28")] * H, 0.05).astype(np.float32)  # noqa: E731
    xt, yt, bt = torch.tensor(Z), torch.tensor(ytr[tr], dtype=torch.float32), torch.tensor(base_all(Xtr[tr]))
    if hold.any():
        Zh, _ = _transform(Xtr[hold], names, stats)
        xh, yh, bh = torch.tensor(Zh), torch.tensor(ytr[hold], dtype=torch.float32), torch.tensor(base_all(Xtr[hold]))
    net = build_mlp(torch, Z.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=float(CONFIG["mlp_lr"]))
    bs = int(CONFIG["mlp_batch_size"])
    max_ep = int(CONFIG["mlp_max_epochs"] if max_epochs is None else max_epochs) if fixed_epochs is None else fixed_epochs
    best = (np.inf, 0, None)
    hist = []
    t0 = time.time()
    for ep in range(1, max_ep + 1):
        net.train()
        perm = torch.tensor(rng.permutation(len(yt)))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            mu, al = _mlp_forward(torch, net, xt[b], bt[b])
            loss = _nb_nll(torch, yt[b], mu, al)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if hold.any():
            net.eval()
            with torch.no_grad():
                mu, al = _mlp_forward(torch, net, xh, bh)
                hv = float(_nb_nll(torch, yh, mu, al))
            hist.append(hv)
            if hv < best[0]:
                best = (hv, ep, {k: v.clone() for k, v in net.state_dict().items()})
    if hold.any():
        net.load_state_dict(best[2])
        best_epoch = best[1]
    else:
        best_epoch = fixed_epochs
    net.eval()
    return {"net": net, "stats": stats, "names": names, "H": H, "best_epoch": int(best_epoch), "holdout_nll": hist,
            "train_seconds": time.time() - t0, "n_train": int(len(yt)), "n_hold": int(hold.sum())}


def predict_mlp_nb(model: dict, X: np.ndarray, shape: tuple[int, int], no_sale: np.ndarray, kappa: float = 1.0) -> dict[str, np.ndarray]:
    """Dự báo MLP-NB: mean, alpha (nhân kappa), Q = phân vị NB chính xác."""
    import torch

    S, n = shape
    names, H = model["names"], model["H"]
    Xf = X.reshape(S * n, -1)
    Z, _ = _transform(Xf, names, model["stats"])
    base = np.maximum(Xf[:, names.index("mean_28")] * H, 0.05).astype(np.float32)
    with torch.no_grad():
        mu, al = _mlp_forward(torch, model["net"], torch.tensor(Z), torch.tensor(base))
    mu = mu.numpy().reshape(S, n)
    al = np.clip(al.numpy().reshape(S, n) * kappa, 1e-3, 20.0)
    mu[no_sale] = 0.0
    Q = nb_quantiles(mu, al, QGRID)
    Q[no_sale] = 0.0
    Qe = nb_quantiles(mu, al, EXT_QGRID)
    Qe[no_sale] = 0.0
    return {"mean": mu, "alpha": al, "Q": Q, "Qext": Qe}
