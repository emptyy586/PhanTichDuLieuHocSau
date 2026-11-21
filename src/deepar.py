"""DeepAR-style (GRU + đầu Negative Binomial, lấy mẫu tự hồi quy) cho tổng nhu cầu cửa sổ D_H(t) = y[t+1] + … + y[t+H].

Giao thức giống MLP-NB (src/ml_models.py, src/run_forecasts.py, src/mlp_seeds.py): cùng chia train/val/test, cùng gốc dự báo
(origins_for_split), H = L + R ∈ {3, 4}; epoch chọn bằng tập kiểm tra nội bộ (56 gốc cuối của train, purge H ngày); siêu tham số chọn trên
VAL bằng pinball trung bình trên lưới phân vị chính × cả hai H; khi chạy test huấn luyện lại trên train+val với siêu tham số/epoch đã chốt.

Đầu vào mỗi bước d: y_{d-1}/scale, embedding id (8 chiều), covariates đã biết (thứ, tháng, ngày trong tháng, SNAP, cờ sự kiện, giá tương đối so với giá
trung vị TRAIN của chuỗi, cờ giá thiếu). KHÔNG có doanh số tương lai, KHÔNG có đặc trưng mã hóa mục tiêu.
scale = 1 + mean(y trong context đến gốc). Đầu ra: mu = scale * softplus(.), alpha = softplus(.) + 1e-3; NLL của NB(mu, alpha).

    python -m src.deepar --stage tune_one --cfg-id K   # (K=0..7, chạy song song) huấn luyện + chấm điểm 1 cấu hình trên val
    python -m src.deepar --stage select                # chọn cấu hình + kappa trên val, ghi results/hparams_deepar.json
    python -m src.deepar --stage seed_val --seed S     # 3 seed, dự báo val; sau đó --stage freeze (đóng băng epoch từng seed)
    python -m src.deepar --stage seed_test --seed S    # TEST, chạy MỘT lần mỗi seed với siêu tham số đã chốt; rồi --stage collect_test
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch

from src.prob_models import (CONFIG, EXT_QGRID, QGRID, Panel, avg_pinball, evaluate_forecast, load_panel, origins_for_split, window_targets)
from src.tuning import horizons
from src.utils import ROOT

N_COV = 15  # 7 thứ + 2 tháng + 2 ngày-trong-tháng + SNAP + sự kiện + giá tương đối + giá thiếu
EMB_DIM = 8
H_MAX = 4
HPARAMS_PATH = ROOT / "results" / "hparams_deepar.json"
CACHE_DIR = ROOT / "results" / "deepar_cache"


# ----------------------------------------------------------------------------- dữ liệu
def build_covariates(panel: Panel, train_end: int) -> dict:
    """Covariates đã biết trước (cả tương lai): shared [T, 13], price_rel [S, T], price_miss [S, T] (giá trung vị chỉ từ ngày <= train_end)."""
    T = panel.n_days
    wd = np.eye(7, dtype=np.float32)[panel.weekday]
    month = panel.month.astype(np.float32)
    dom = panel.dates.day.to_numpy().astype(np.float32)
    shared = np.concatenate([
        wd,
        np.stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12), np.sin(2 * np.pi * dom / 31), np.cos(2 * np.pi * dom / 31)], axis=1).astype(np.float32),
        panel.snap.astype(np.float32)[:, None],
        pd.notna(panel.event_name).astype(np.float32)[:, None],
    ], axis=1)  # [T, 13]
    price = panel.price
    with np.errstate(all="ignore"):
        med = np.nanmedian(price[:, : train_end + 1], axis=1)
    med = np.where(np.isfinite(med) & (med > 0), med, 1.0)
    rel = price / med[:, None]
    miss = np.isnan(rel)
    rel = np.where(miss, 1.0, rel).astype(np.float32)
    return {"shared": shared, "price_rel": rel, "price_miss": miss.astype(np.float32)}


def first_sale_index(panel: Panel) -> np.ndarray:
    """Chỉ số ngày bán đầu tiên của từng chuỗi (cùng quy tắc `age_days` của các mô hình khác)."""
    return np.argmax(panel.Y > 0, axis=1)


# ----------------------------------------------------------------------------- mô hình
class DeepARNet(torch.nn.Module):
    """GRU nhiều lớp + đầu NB (mu, alpha)."""

    def __init__(self, n_series: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.emb = torch.nn.Embedding(n_series, EMB_DIM)
        self.gru = torch.nn.GRU(1 + EMB_DIM + N_COV, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """x: [B, T, 1+EMB+COV] → (raw [B, T, 2], h)."""
        out, h = self.gru(x, h)
        return self.head(self.drop(out)), h


def nb_params(raw: torch.Tensor, scale: torch.Tensor, kappa: float = 1.0):
    """(mu, alpha) từ đầu ra thô: mu = scale·softplus(.), alpha = (softplus(.)+1e-3)·kappa."""
    mu = scale * torch.nn.functional.softplus(raw[..., 0]) + 1e-6
    alpha = (torch.nn.functional.softplus(raw[..., 1]) + 1e-3) * kappa
    return mu, alpha


def nb_nll_elem(y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Âm log-hợp lý NB(mean=mu, var=mu+alpha·mu²) từng phần tử."""
    r = 1.0 / alpha
    ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0) + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu)))
    return -ll


class Data:
    """Tensor hóa dữ liệu một lần: Y, covariates, id; hàm gom batch cửa sổ (i, gốc t_o)."""

    def __init__(self, panel: Panel, train_end: int):
        self.panel = panel
        cov = build_covariates(panel, train_end)
        self.Y = torch.tensor(panel.Y, dtype=torch.float32)
        self.shared = torch.tensor(cov["shared"])
        self.rel = torch.tensor(cov["price_rel"])
        self.miss = torch.tensor(cov["price_miss"])
        self.first = first_sale_index(panel)
        self.T = panel.n_days

    def batch(self, ids: torch.Tensor, t_o: torch.Tensor, C: int, H: int):
        """Gom x [B, C+H, 1+COV] (chưa gồm embedding), target [B, C+H], scale [B], ngày [B, C+H]. Bước j ứng với ngày d_j = t_o - C + 1 + j."""
        B = len(ids)
        d = t_o[:, None] - C + 1 + torch.arange(C + H)[None, :]
        dc = d.clamp(0, self.T - 1)
        prev = (d - 1).clamp(0, self.T - 1)
        y = self.Y[ids[:, None], dc]
        y_prev = self.Y[ids[:, None], prev]
        ctx = self.Y[ids[:, None], dc[:, :C]]
        scale = 1.0 + ctx.mean(dim=1)
        cov = torch.cat([self.shared[dc], self.rel[ids[:, None], dc][..., None], self.miss[ids[:, None], dc][..., None]], dim=-1)  # [B, C+H, 15]
        return y_prev / scale[:, None], cov, y, scale


def make_input(net: DeepARNet, ids: torch.Tensor, y_prev_scaled: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    e = net.emb(ids)[:, None, :].expand(-1, cov.shape[1], -1)
    return torch.cat([y_prev_scaled[..., None], e, cov], dim=-1)


# ----------------------------------------------------------------------------- huấn luyện
def training_pairs(data: Data, last_label_day: int, n_days: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cặp (chuỗi, gốc) hợp lệ: nhãn đến gốc+H_MAX <= last_label_day, chuỗi đã có lần bán tại gốc, trong `n_days` gốc gần nhất."""
    n_days = int(CONFIG["ml_train_days"] if n_days is None else n_days)
    hi = last_label_day - H_MAX
    org = np.arange(max(hi - n_days + 1, 28), hi + 1)
    S = data.panel.n_series
    ii, tt = np.meshgrid(np.arange(S), org, indexing="ij")
    keep = data.first[:, None] <= tt
    return ii[keep], tt[keep]


def _epoch_loss(net, data, ids, t_o, C, bs, horizon_only: bool) -> float:
    net.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(ids), bs):
            i, t = torch.tensor(ids[s:s + bs]), torch.tensor(t_o[s:s + bs])
            yp, cov, y, scale = data.batch(i, t, C, H_MAX)
            raw, _ = net(make_input(net, i, yp, cov))
            mu, al = nb_params(raw, scale[:, None])
            nll = nb_nll_elem(y, mu, al)
            if horizon_only:
                nll = nll[:, C:]
            tot += float(nll.sum()); cnt += nll.numel()
    return tot / cnt


def fit_deepar(data: Data, cfg: dict, last_label_day: int, seed: int, fixed_epochs: int | None = None, max_epochs: int | None = None) -> dict:
    """Huấn luyện DeepAR bằng cửa sổ ngẫu nhiên (teacher forcing, NLL trên toàn cửa sổ). Epoch chọn bằng tập kiểm tra nội bộ
    (56 gốc cuối của train, purge H_MAX) nếu `fixed_epochs` là None; ngược lại dùng đúng số epoch đó (refit train+val)."""
    torch.manual_seed(seed)
    torch.set_num_threads(int(CONFIG.get("deepar_threads", 12)))
    rng = np.random.default_rng(seed)
    C, bs = int(cfg["context"]), int(CONFIG["deepar_batch_size"])
    ids_all, t_all = training_pairs(data, last_label_day)
    last = t_all.max()
    n_hold = int(CONFIG["mlp_holdout_origins"])
    if fixed_epochs is None:
        hold = t_all > last - n_hold
        tr = t_all <= last - n_hold - H_MAX
        hi_idx = np.flatnonzero(hold)
        hi_idx = rng.choice(hi_idx, size=min(len(hi_idx), int(CONFIG["deepar_holdout_windows"])), replace=False)
        hold_ids, hold_t = ids_all[hi_idx], t_all[hi_idx]
    else:
        tr = np.ones_like(t_all, dtype=bool)
    tr_idx = np.flatnonzero(tr)
    net = DeepARNet(data.panel.n_series, int(cfg["hidden"]), int(cfg["layers"]), float(CONFIG["deepar_dropout"]))
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=float(CONFIG["deepar_lr"]))
    max_ep = int(CONFIG["deepar_max_epochs"] if max_epochs is None else max_epochs) if fixed_epochs is None else int(fixed_epochs)
    n_win = int(CONFIG["deepar_windows_per_epoch"])
    best, curves = (np.inf, 0, None), []
    t0 = time.time()
    for ep in range(1, max_ep + 1):
        net.train()
        pick = rng.choice(tr_idx, size=min(n_win, len(tr_idx)), replace=False)
        run_loss, run_n = 0.0, 0
        for s in range(0, len(pick), bs):
            sel = pick[s:s + bs]
            i, t = torch.tensor(ids_all[sel]), torch.tensor(t_all[sel])
            yp, cov, y, scale = data.batch(i, t, C, H_MAX)
            raw, _ = net(make_input(net, i, yp, cov))
            mu, al = nb_params(raw, scale[:, None])
            loss = nb_nll_elem(y, mu, al).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(CONFIG["deepar_clip"]))
            opt.step()
            run_loss += float(loss.detach()) * len(sel); run_n += len(sel)
        rec = {"epoch": ep, "train_nll": run_loss / run_n}
        if fixed_epochs is None:
            hv = _epoch_loss(net, data, hold_ids, hold_t, C, bs, horizon_only=True)
            rec["holdout_nll"] = hv
            if hv < best[0]:
                best = (hv, ep, {k: v.clone() for k, v in net.state_dict().items()})
        curves.append(rec)
    if fixed_epochs is None:
        net.load_state_dict(best[2])
        best_epoch = best[1]
    else:
        best_epoch = int(fixed_epochs)
    net.eval()
    return {"net": net, "cfg": dict(cfg), "best_epoch": int(best_epoch), "curves": curves, "n_params": int(n_params), "train_seconds": time.time() - t0,
            "n_train_windows": int(len(tr_idx)), "seed": seed}


# ----------------------------------------------------------------------------- lấy mẫu tự hồi quy
def sample_quantile(sorted_samples: torch.Tensor, qs: list[float]) -> np.ndarray:
    """Phân vị kinh nghiệm kiểu 'inverted_cdf' (số nguyên): giá trị nhỏ nhất x với F(x) >= q. sorted_samples: [B, n] đã sắp tăng."""
    n = sorted_samples.shape[1]
    idx = [max(int(np.ceil(q * n - 1e-9)) - 1, 0) for q in qs]
    return sorted_samples[:, idx].numpy()


@torch.no_grad()
def sample_windows(model: dict, data: Data, ids: np.ndarray, t_o: np.ndarray, H: int, n_samples: int, kappa: float, seed: int, chunk: int = 400_000) -> torch.Tensor:
    """Mẫu tự hồi quy tổng D_H cho từng cặp (chuỗi, gốc): trả về [P, n_samples].

    Encoder (GRU) chạy teacher-forcing trên context d_0..t_o (chỉ dùng doanh số đã quan sát đến t_o); sau đó H bước: bước k nhận
    (y_{t_o+k-1} quan sát hoặc đã lấy mẫu, covariates ngày t_o+k đã biết), cho NB(mu, alpha·kappa), lấy mẫu Gamma–Poisson, đưa mẫu vào bước kế.
    """
    net, C = model["net"], int(model["cfg"]["context"])
    net.eval()
    torch.manual_seed(seed)
    P = len(ids)
    out = torch.zeros(P, n_samples)
    per = max(1, chunk // n_samples)
    for s in range(0, P, per):
        i, t = torch.tensor(ids[s:s + per]), torch.tensor(t_o[s:s + per])
        B = len(i)
        yp, cov, _, scale = data.batch(i, t, C, 0)  # context: các ngày d_0..t_o
        _, h = net(make_input(net, i, yp, cov))
        dk = (t[:, None] + 1 + torch.arange(H)[None, :]).clamp(0, data.T - 1)
        cov_f = torch.cat([data.shared[dk], data.rel[i[:, None], dk][..., None], data.miss[i[:, None], dk][..., None]], dim=-1)  # [B, H, 15]
        emb = net.emb(i)
        y_in = (data.Y[i, t] / scale)[:, None, None]
        total = torch.zeros(B, n_samples)
        for k in range(H):
            if k == 0:
                x = torch.cat([y_in, emb[:, None, :], cov_f[:, 0:1]], dim=-1)
                o, h = net.gru(x, h)
                mu, al = nb_params(net.head(o)[:, 0], scale, kappa)
                mu, al = mu[:, None].expand(B, n_samples), al[:, None].expand(B, n_samples)
                ysamp = torch.poisson(torch.distributions.Gamma(1.0 / al, 1.0 / (al * mu)).sample())
                h = h[:, :, None, :].expand(-1, -1, n_samples, -1).reshape(h.shape[0], B * n_samples, -1).contiguous()
                scale_n = scale[:, None].expand(B, n_samples).reshape(-1)
                emb_n = emb[:, None, :].expand(B, n_samples, -1).reshape(B * n_samples, -1)
            else:
                x = torch.cat([(ysamp.reshape(-1) / scale_n)[:, None, None], emb_n[:, None, :],
                               cov_f[:, k][:, None, :].expand(B, n_samples, -1).reshape(B * n_samples, 1, -1)], dim=-1)
                o, h = net.gru(x, h)
                mu, al = nb_params(net.head(o)[:, 0], scale_n, kappa)
                ysamp = torch.poisson(torch.distributions.Gamma(1.0 / al, 1.0 / (al * mu)).sample()).reshape(B, n_samples)
            total += ysamp
        out[s:s + B] = total
    return out


def forecast_deepar(model: dict, data: Data, panel: Panel, split: str, n_samples: int, kappa: float, seed: int) -> dict[int, dict[str, np.ndarray]]:
    """Dự báo cho mọi H: {H: {"Q": [S,n,9], "Qext": [S,n,2], "mean": [S,n]}} – cùng schema các mô hình khác; chuỗi chưa bán tại gốc → 0."""
    res = {}
    for H in horizons():
        org = origins_for_split(panel, split, H)
        S, n = panel.n_series, len(org)
        ids = np.repeat(np.arange(S), n)
        t_o = np.tile(org, S)
        smp = sample_windows(model, data, ids, t_o, H, n_samples, kappa, seed + 1000 * H)
        sorted_s, _ = torch.sort(smp, dim=1)
        no_sale = (data.first[ids] > t_o)
        Q = sample_quantile(sorted_s, QGRID).reshape(S, n, len(QGRID))
        Qe = sample_quantile(sorted_s, EXT_QGRID).reshape(S, n, len(EXT_QGRID))
        mean = smp.mean(dim=1).numpy().reshape(S, n)
        ns = no_sale.reshape(S, n)
        Q[ns] = 0.0; Qe[ns] = 0.0; mean[ns] = 0.0
        res[H] = {"Q": Q.astype(float), "Qext": Qe.astype(float), "mean": mean.astype(float)}
    return res


# ----------------------------------------------------------------------------- tuning + chạy (các stage chạy song song theo tiến trình)
def config_grid() -> list[dict]:
    """Lưới siêu tham số từ config: hidden × layers × context."""
    return [{"hidden": h, "layers": l, "context": c} for h, l, c in itertools.product(CONFIG["deepar_hidden"], CONFIG["deepar_layers"], CONFIG["deepar_context"])]


def _score(panel, fc) -> float:
    Hs = horizons()
    origins = {H: origins_for_split(panel, "val", H) for H in Hs}
    return avg_pinball(panel, Hs, origins, fc)


def _save_npz(path, panel, fc):
    arrays = {"ids": panel.ids, "quantile_grid": np.array(QGRID), "quantile_grid_ext": np.array(EXT_QGRID)}
    for H, d in fc.items():
        arrays[f"DeepAR|H{H}|Q"], arrays[f"DeepAR|H{H}|mean"], arrays[f"DeepAR|H{H}|Qext"] = d["Q"], d["mean"], d["Qext"]
    np.savez_compressed(path, **arrays)


def save_model(m: dict, name: str) -> None:
    """Lưu trọng số + metadata (cache, không đưa vào git)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state": m["net"].state_dict(), "cfg": m["cfg"], "best_epoch": m["best_epoch"], "curves": m["curves"], "n_params": m["n_params"],
                "train_seconds": m["train_seconds"], "seed": m["seed"]}, CACHE_DIR / f"{name}.pt")


def load_model(panel: Panel, name: str) -> dict:
    """Nạp mô hình đã lưu bởi save_model."""
    d = torch.load(CACHE_DIR / f"{name}.pt", weights_only=False)
    net = DeepARNet(panel.n_series, int(d["cfg"]["hidden"]), int(d["cfg"]["layers"]), float(CONFIG["deepar_dropout"]))
    net.load_state_dict(d["state"])
    net.eval()
    return {"net": net, "cfg": d["cfg"], "best_epoch": d["best_epoch"], "curves": d["curves"], "n_params": d["n_params"], "train_seconds": d["train_seconds"], "seed": d["seed"]}


def _threads(n: int) -> None:
    CONFIG["deepar_threads"] = n


def stage_tune_one(cfg_id: int) -> None:
    """Một cấu hình của lưới: huấn luyện seed đầu trên train, dự báo val (n_samples tuning, kappa=1), ghi dòng log + mô hình vào cache."""
    panel = load_panel()
    last = panel.val_start - 1
    data = Data(panel, last)
    cfg = config_grid()[cfg_id]
    seed = int(CONFIG["mlp_seeds"][0])
    m = fit_deepar(data, cfg, last, seed)
    t0 = time.time()
    fc = forecast_deepar(m, data, panel, "val", int(CONFIG["deepar_tune_samples"]), 1.0, seed)
    row = {"cfg_id": cfg_id, **cfg, "best_epoch": m["best_epoch"], "n_params": m["n_params"], "train_seconds": round(m["train_seconds"], 1),
           "sample_seconds": round(time.time() - t0, 1), "holdout_nll": min(r["holdout_nll"] for r in m["curves"]), "pinball_avg_val": _score(panel, fc)}
    save_model(m, f"tune_cfg{cfg_id}")
    (CACHE_DIR / f"tune_row_{cfg_id}.json").write_text(json.dumps(row), encoding="utf-8")
    print(row, flush=True)


def stage_select() -> None:
    """Chọn cấu hình tốt nhất theo pinball_avg VAL; chọn kappa (lưới config) cho cấu hình đó; ghi tuning_log_deepar.csv, hparams_deepar.json."""
    panel = load_panel()
    last = panel.val_start - 1
    data = Data(panel, last)
    rows = [json.loads((CACHE_DIR / f"tune_row_{k}.json").read_text(encoding="utf-8")) for k in range(len(config_grid()))]
    log = pd.DataFrame(rows)
    log.to_csv(ROOT / "results" / "tuning_log_deepar.csv", index=False)
    best_k = int(log.sort_values("pinball_avg_val").iloc[0].cfg_id)
    m = load_model(panel, f"tune_cfg{best_k}")
    seed = int(CONFIG["mlp_seeds"][0])
    kap = []
    for kappa in [float(x) for x in CONFIG["deepar_kappa_grid"]]:
        fc = forecast_deepar(m, data, panel, "val", int(CONFIG["deepar_tune_samples"]), kappa, seed)
        kap.append({"kappa": kappa, "pinball_avg_val": _score(panel, fc)})
        print(kap[-1], flush=True)
    pd.DataFrame(kap).to_csv(ROOT / "results" / "tuning_kappa_deepar.csv", index=False)
    kappa_best = min(kap, key=lambda r: r["pinball_avg_val"])["kappa"]
    HPARAMS_PATH.write_text(json.dumps({"config": m["cfg"], "cfg_id": best_k, "kappa": kappa_best, "n_samples": int(CONFIG["deepar_n_samples"]), "n_params": m["n_params"],
                                        "selection": f"pinball trung bình mọi q (lưới chính) và cả H trên VAL, seed {seed}, n_samples={int(CONFIG['deepar_tune_samples'])}",
                                        "epochs_by_seed": {}, "torch": torch.__version__}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("best cfg", best_k, m["cfg"], "kappa", kappa_best, flush=True)


def stage_seed_val(seed: int) -> None:
    """Cấu hình đã chọn: huấn luyện (seed đầu dùng lại mô hình tuning), dự báo val n_samples chính thức với kappa đã chọn; ghi dòng kết quả/đường NLL."""
    hp = json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))
    panel = load_panel()
    last = panel.val_start - 1
    data = Data(panel, last)
    if seed == int(CONFIG["mlp_seeds"][0]):
        m = load_model(panel, f"tune_cfg{hp['cfg_id']}")
    else:
        m = fit_deepar(data, hp["config"], last, seed)
    save_model(m, f"val_seed{seed}")
    t0 = time.time()
    fc = forecast_deepar(m, data, panel, "val", int(hp["n_samples"]), float(hp["kappa"]), seed)
    rows = []
    for H in horizons():
        ev1 = evaluate_forecast("DeepAR", H, panel, origins_for_split(panel, "val", H), fc[H])
        ov = ev1[(ev1.group_type == "overall") & ev1.metric.isin(["pinball_avg", "wape", "mase"])].set_index("metric").value
        rows.append({"seed": seed, "H": H, "split": "val", "kappa": hp["kappa"], "epochs": m["best_epoch"], **ov.to_dict(), "train_seconds": round(m["train_seconds"], 1),
                     "sample_seconds": round(time.time() - t0, 1), "n_params": m["n_params"]})
        if seed == int(CONFIG["mlp_seeds"][0]):
            ev_all = pd.concat([evaluate_forecast("DeepAR", Hh, panel, origins_for_split(panel, "val", Hh), fc[Hh]) for Hh in horizons()], ignore_index=True)
            ev_all.to_csv(ROOT / "results" / "forecast_eval_deepar_val_all.csv", index=False)
            _save_npz(ROOT / "results" / "forecasts" / "deepar_val.npz", panel, fc)
    pd.DataFrame(rows).to_csv(CACHE_DIR / f"seed_rows_val_{seed}.csv", index=False)
    pd.DataFrame([{"seed": seed, **r} for r in m["curves"]]).to_csv(CACHE_DIR / f"curves_{seed}.csv", index=False)
    print(f"val seed {seed} xong", rows, flush=True)


def stage_freeze() -> None:
    """Gom 3 seed val vào deepar_seeds.csv/deepar_nll_curves.csv và ghi epoch từng seed vào hparams_deepar.json (ĐÓNG BĂNG trước khi chạy test)."""
    hp = json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))
    seeds = [int(s) for s in CONFIG["mlp_seeds"]]
    rows = pd.concat([pd.read_csv(CACHE_DIR / f"seed_rows_val_{s}.csv") for s in seeds], ignore_index=True)
    rows.to_csv(ROOT / "results" / "deepar_seeds.csv", index=False)
    pd.concat([pd.read_csv(CACHE_DIR / f"curves_{s}.csv") for s in seeds], ignore_index=True).to_csv(ROOT / "results" / "deepar_nll_curves.csv", index=False)
    hp["epochs_by_seed"] = {str(s): int(rows[(rows.seed == s)].epochs.iloc[0]) for s in seeds}
    HPARAMS_PATH.write_text(json.dumps(hp, indent=2, ensure_ascii=False), encoding="utf-8")
    print("đã đóng băng", hp, flush=True)


def stage_seed_test(seed: int) -> None:
    """TEST: huấn luyện lại trên train+val với siêu tham số/epoch/kappa đã chốt (hparams_deepar.json), dự báo test. Mỗi seed chạy MỘT lần."""
    hp = json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))
    panel = load_panel()
    last = panel.test_start - 1
    data = Data(panel, panel.val_start - 1)  # giá trung vị chỉ từ train (như khi chạy val)
    m = fit_deepar(data, hp["config"], last, seed, fixed_epochs=int(hp["epochs_by_seed"][str(seed)]))
    save_model(m, f"test_seed{seed}")
    t0 = time.time()
    fc = forecast_deepar(m, data, panel, "test", int(hp["n_samples"]), float(hp["kappa"]), seed)
    rows = []
    for H in horizons():
        ev1 = evaluate_forecast("DeepAR", H, panel, origins_for_split(panel, "test", H), fc[H])
        ov = ev1[(ev1.group_type == "overall") & ev1.metric.isin(["pinball_avg", "wape", "mase"])].set_index("metric").value
        rows.append({"seed": seed, "H": H, "split": "test", "kappa": hp["kappa"], "epochs": m["best_epoch"], **ov.to_dict(), "train_seconds": round(m["train_seconds"], 1),
                     "sample_seconds": round(time.time() - t0, 1), "n_params": m["n_params"]})
    if seed == int(CONFIG["mlp_seeds"][0]):
        ev_all = pd.concat([evaluate_forecast("DeepAR", Hh, panel, origins_for_split(panel, "test", Hh), fc[Hh]) for Hh in horizons()], ignore_index=True)
        ev_all.to_csv(ROOT / "results" / "forecast_eval_deepar_test_all.csv", index=False)
        _save_npz(ROOT / "results" / "forecasts" / "deepar_test.npz", panel, fc)
    pd.DataFrame(rows).to_csv(CACHE_DIR / f"seed_rows_test_{seed}.csv", index=False)
    print(f"test seed {seed} xong", rows, flush=True)


def stage_collect_test() -> None:
    """Gom 3 seed test vào deepar_seeds.csv."""
    seeds = [int(s) for s in CONFIG["mlp_seeds"]]
    prev = pd.read_csv(ROOT / "results" / "deepar_seeds.csv")
    prev = prev[prev.split != "test"]
    pd.concat([prev] + [pd.read_csv(CACHE_DIR / f"seed_rows_test_{s}.csv") for s in seeds], ignore_index=True).to_csv(ROOT / "results" / "deepar_seeds.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune_one", "select", "seed_val", "freeze", "seed_test", "collect_test"], required=True)
    ap.add_argument("--cfg-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    _threads(a.threads)
    {"tune_one": lambda: stage_tune_one(a.cfg_id), "select": stage_select, "seed_val": lambda: stage_seed_val(a.seed), "freeze": stage_freeze,
     "seed_test": lambda: stage_seed_test(a.seed), "collect_test": stage_collect_test}[a.stage]()


if __name__ == "__main__":
    main()
