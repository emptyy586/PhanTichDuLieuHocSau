import numpy as np

from src.inventory_sim import aggregate, simulate_order_up_to


def _random_case(seed=0, S=40, D=28, L=2):
    rng = np.random.default_rng(seed)
    demand = rng.poisson(rng.uniform(0.2, 5, size=(S, 1)), size=(S, D)).astype(float)
    n = D - (L + 1) + 1
    base = np.ceil(demand.mean(axis=1))[:, None, None] * (L + 1)
    levels = np.concatenate([base * f for f in (0.5, 1.0, 1.5, 2.0, 3.0)], axis=0).reshape(5, S, 1) * np.ones((1, 1, n))
    return demand, levels, L


def test_constant_demand_matches_hand_calculation():
    c, L, D = 2.0, 2, 28
    n = D - (L + 1) + 1
    demand = np.full((1, D), c)
    levels = np.full((1, 1, n), 6.0)  # S = c*(L+1)
    res = simulate_order_up_to(demand, levels, L, return_trace=True)
    end = res["trace"]["end"][:, 0, 0]
    assert end[0] == 4.0 and (end[1:-1] == 2.0).all() and end[-1] == 0.0  # tính tay: 4, 2, ..., 2, 0 (ngày cuối không còn hàng về)
    assert res["lost"][0, 0] == 0 and res["stockout_days"][0, 0] == 0
    assert res["sales"][0, 0] == c * D
    assert res["hold_units"][0, 0] == 4 + 2 * 26 and res["orders_total"][0, 0] == 2 * (n - 1)
    assert (res["trace"]["order"][: n - 1, 0, 0] == 2.0).all() and (res["trace"]["order"][n - 1:] == 0).all()


def test_order_accounting_and_inventory_position_after_ordering():
    """Tổng đặt = tổng đã về + hàng còn đang về cuối kỳ; vị trí tồn kho sau đặt hàng = max(S, vị trí trước đó) mọi ngày đặt."""
    for L in (2, 3):
        demand, levels, _ = _random_case(L=L)
        rng = np.random.default_rng(L)
        levels = np.ceil(levels * (0.6 + 0.8 * rng.random(levels.shape)))  # mức S đổi theo ngày
        C, S, n = levels.shape
        D = demand.shape[1]
        tr = simulate_order_up_to(demand, levels, L, return_trace=True)["trace"]
        order, received, end = tr["order"], tr["received"], tr["end"]  # [D, C, S]; chỉ số ngày d-1
        assert order.min() >= 0 and np.allclose(order[n - 1:], 0)  # không đặt khi hết gốc có dự báo
        still_in_pipe = sum(order[d - 1] for d in range(1, D + 1) if d + L > D)
        assert np.allclose(order.sum(axis=0), received.sum(axis=0) + still_in_pipe)
        for d in range(1, n):
            pipe_before = sum(order[e - 1] for e in range(1, d) if e + L > d) if d > 1 else 0.0
            ip_before = end[d - 1] + pipe_before
            assert np.allclose(ip_before + order[d - 1], np.maximum(levels[:, :, d], ip_before))
        assert np.allclose(sim_sales_plus_lost(demand, levels, L), demand.sum(axis=1)[None, :])


def sim_sales_plus_lost(demand, levels, L):
    res = simulate_order_up_to(demand, levels, L)
    return res["sales"] + res["lost"]


def test_costs_nonnegative_and_fill_rate_range_and_monotone_in_level():
    demand, levels, L = _random_case(seed=3)
    res = simulate_order_up_to(demand, levels, L)
    masks = [("overall", "overall", np.ones(demand.shape[0], dtype=bool))]
    rows = aggregate(res, demand, masks, 1.0, [1.0, 10.0])
    assert all(r["holding_cost"] >= 0 and r["stockout_cost"] >= 0 and r["total_cost"] >= 0 for r in rows)
    assert all(0.0 <= r["fill_rate"] <= 1.0 for r in rows)
    fills = [r["fill_rate"] for r in rows if r["cost_ratio"] == 1.0]  # case tăng dần theo mức S
    assert all(b >= a - 1e-12 for a, b in zip(fills, fills[1:]))
    invs = [r["avg_inventory"] for r in rows if r["cost_ratio"] == 1.0]
    assert all(b >= a - 1e-12 for a, b in zip(invs, invs[1:]))


def test_reproducible_same_inputs():
    demand, levels, L = _random_case(seed=5)
    a = simulate_order_up_to(demand, levels, L)
    b = simulate_order_up_to(demand.copy(), levels.copy(), L)
    assert all(np.array_equal(a[k], b[k]) for k in a)
