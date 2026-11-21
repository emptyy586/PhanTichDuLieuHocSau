"""So khớp mô phỏng vector hóa với một bản tham chiếu viết độc lập (vòng lặp từng chuỗi, danh sách hàng đang về)."""
import numpy as np
import pytest

from src.inventory_sim import simulate_order_up_to


def _reference(demand_row, levels_row, L):
    D, n = len(demand_row), len(levels_row)
    onhand, pipe = levels_row[0], []  # pipe: (ngày hàng về, số lượng)
    sales = lost = hold = stockout_days = orders = 0.0
    for d in range(1, D + 1):
        y = demand_row[d - 1]
        s = min(onhand, y)
        onhand -= s
        sales += s
        lost += y - s
        stockout_days += (y - s) > 1e-12
        onhand += sum(q for day, q in pipe if day == d)  # hàng về cuối ngày d
        pipe = [(day, q) for day, q in pipe if day != d]
        hold += onhand
        if d <= n - 1:  # đặt hàng cuối ngày d với mức order-up-to của gốc d
            order = max(0.0, levels_row[d] - (onhand + sum(q for _, q in pipe)))
            if order > 0:
                pipe.append((d + L, order))
            orders += order
    return np.array([sales, lost, hold, stockout_days, orders])


@pytest.mark.parametrize("L", [2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_vectorized_matches_reference(L, seed):
    rng = np.random.default_rng(seed)
    S, D, C = 20, 28, 3
    n = D - (L + 1) + 1
    demand = rng.poisson(rng.uniform(0.05, 6, (S, 1)), (S, D)).astype(float)
    levels = np.ceil(rng.uniform(0, 2.5, (C, S, 1)) * (demand.mean(1)[None, :, None] * (L + 1) + 0.5)
                     * (1 + 0.3 * rng.random((C, S, n))))
    res = simulate_order_up_to(demand, levels, L)
    for c in range(C):
        for i in range(S):
            got = np.array([res["sales"][c, i], res["lost"][c, i], res["hold_units"][c, i],
                            res["stockout_days"][c, i], res["orders_total"][c, i]])
            assert np.allclose(got, _reference(demand[i], levels[c, i], L))
