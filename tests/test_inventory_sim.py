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


def test_inventory_conservation_every_day():
    demand, levels, L = _random_case()
    res = simulate_order_up_to(demand, levels, L, return_trace=True)
    tr = res["trace"]
    prev_end = levels[:, :, 0]
    for d in range(tr["end"].shape[0]):
        assert np.allclose(tr["start"][d], prev_end)
        assert np.allclose(tr["start"][d] + tr["received"][d] - tr["sales"][d], tr["end"][d])  # tồn đầu + nhận - bán = tồn cuối
        prev_end = tr["end"][d]
    assert np.allclose(res["sales"] + res["lost"], demand.sum(axis=1)[None, :])


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
