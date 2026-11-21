import hashlib

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

from src import deepar as D
from src.inventory_sim import quantile_to_levels
from src.prob_models import CONFIG, EXT_QGRID, QGRID, Panel, origins_for_split
from src.utils import ROOT
from tests.test_prob_models import make_panel


@pytest.fixture(scope="module")
def small():
    """Panel giả nhỏ + mô hình DeepAR rất nhỏ (huấn luyện 2 epoch) để test nhanh."""
    old = {k: CONFIG[k] for k in ("deepar_windows_per_epoch", "deepar_holdout_windows", "deepar_batch_size", "mlp_holdout_origins", "ml_train_days", "deepar_threads")}
    CONFIG.update(deepar_windows_per_epoch=300, deepar_holdout_windows=200, deepar_batch_size=64, mlp_holdout_origins=10, ml_train_days=120, deepar_threads=2)
    P = make_panel()
    P.Y[:, :3] = 0.0
    data = D.Data(P, P.val_start - 1)
    cfg = {"hidden": 8, "layers": 1, "context": 10}
    m = D.fit_deepar(data, cfg, P.val_start - 1, seed=1, max_epochs=2)
    yield P, data, m
    CONFIG.update(old)


def test_nb_nll_matches_scipy_logpmf():
    y = torch.tensor([0.0, 1.0, 4.0, 9.0])
    mu = torch.tensor([0.5, 2.0, 3.0, 7.0])
    al = torch.tensor([2.0, 0.3, 1.0, 0.1])
    got = D.nb_nll_elem(y, mu, al).numpy()
    ref = -nbinom.logpmf(y.numpy(), 1 / al.numpy(), 1 / (1 + al.numpy() * mu.numpy()))
    assert np.allclose(got, ref, atol=1e-5)


def test_no_leak_future_values_after_origin_do_not_change_forecast(small):
    P, data, m = small
    t0 = P.val_start + 4
    ids, t = np.array([0, 2, 4]), np.full(3, t0)
    a = D.sample_windows(m, data, ids, t, 3, 200, 1.0, seed=7)
    Y2 = P.Y.copy()
    Y2[:, t0 + 1:] = 999.0  # đổi toàn bộ doanh số SAU gốc (trong horizon và xa hơn)
    data2 = D.Data(Panel(**{**P.__dict__, "Y": Y2}), P.val_start - 1)
    b = D.sample_windows(m, data2, ids, t, 3, 200, 1.0, seed=7)
    assert torch.equal(a, b)


def test_same_seed_same_result_different_seed_different_weights(small):
    P, data, _ = small
    cfg = {"hidden": 8, "layers": 1, "context": 10}
    a = D.fit_deepar(data, cfg, P.val_start - 1, seed=3, fixed_epochs=1)
    b = D.fit_deepar(data, cfg, P.val_start - 1, seed=3, fixed_epochs=1)
    c = D.fit_deepar(data, cfg, P.val_start - 1, seed=4, fixed_epochs=1)
    wa, wb, wc = (x["net"].gru.weight_ih_l0.detach().numpy() for x in (a, b, c))
    assert np.allclose(wa, wb) and not np.allclose(wa, wc)
    t = np.array([P.val_start + 2, P.val_start + 2])
    s1 = D.sample_windows(a, data, np.array([0, 1]), t, 3, 100, 1.0, seed=5)
    s2 = D.sample_windows(b, data, np.array([0, 1]), t, 3, 100, 1.0, seed=5)
    assert torch.equal(s1, s2)


def test_first_step_sample_mean_matches_mu(small):
    P, data, m = small
    i, t = torch.tensor([1]), torch.tensor([P.val_start + 3])
    yp, cov, _, scale = data.batch(i, t, 10, 0)
    _, h = m["net"](D.make_input(m["net"], i, yp, cov))
    dk = (t[:, None] + 1).clamp(0, data.T - 1)
    cov_f = torch.cat([data.shared[dk], data.rel[i[:, None], dk][..., None], data.miss[i[:, None], dk][..., None]], dim=-1)
    x = torch.cat([(data.Y[i, t] / scale)[:, None, None], m["net"].emb(i)[:, None, :], cov_f[:, 0:1]], dim=-1)
    o, _ = m["net"].gru(x, h)
    mu, _ = D.nb_params(m["net"].head(o)[:, 0], scale)
    s = D.sample_windows(m, data, np.array([1]), np.array([P.val_start + 3]), 1, 20000, 1.0, seed=9)
    assert abs(float(s.mean()) - float(mu)) / float(mu) < 0.05


def test_forecast_schema_integer_nonnegative_monotone(small):
    P, data, m = small
    fc = D.forecast_deepar(m, data, P, "val", 100, 1.0, 11)
    for H in (3, 4):
        n = len(origins_for_split(P, "val", H))
        d = fc[H]
        assert d["Q"].shape == (P.n_series, n, len(QGRID)) and d["Qext"].shape == (P.n_series, n, len(EXT_QGRID)) and d["mean"].shape == (P.n_series, n)
        allq = np.concatenate([d["Qext"], d["Q"]], axis=-1)
        assert (allq >= 0).all() and np.array_equal(allq, np.round(allq)) and (np.diff(allq, axis=-1) >= 0).all()
        assert np.array_equal(quantile_to_levels("DeepAR", d["Q"]), d["Q"])  # đã là số nguyên: không làm tròn thêm


def test_sample_quantile_is_inverted_cdf():
    s = torch.tensor([[0.0, 1.0, 1.0, 2.0, 5.0]])
    assert D.sample_quantile(s, [0.2, 0.5, 0.6, 1.0]).tolist() == [[0.0, 1.0, 1.0, 5.0]]


def test_old_result_files_unchanged_since_baseline():
    """DeepAR không làm đổi file kết quả cũ: so sha256 với results/gd8_baseline_hashes.txt."""
    lines = (ROOT / "results" / "gd8_baseline_hashes.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 30
    for ln in lines:
        h, path = ln.split(" *")
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h, f"file cũ đã đổi: {path}"


def test_hparams_json_frozen():
    h = hashlib.sha256((ROOT / "results" / "hparams.json").read_bytes()).hexdigest()
    assert h.upper() == "8BC7D7507BF0167E83B1D31F831FB8EDB5EDF60C23A742C67B32B0F86366765B"
