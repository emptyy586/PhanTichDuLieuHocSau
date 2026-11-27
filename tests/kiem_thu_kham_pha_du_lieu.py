import pandas as pd

from src.kham_pha_du_lieu import active_rows, demand_speed_table, load_train, seasonality_tables


def _toy():
    dates = pd.date_range("2020-01-01", periods=14)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"id": "A", "date": d, "y": 0 if i < 3 else 2, "split": "train", "weekday": d.day_name(),
                     "month": d.month, "snap_CA": 0, "event_name_1": None, "event_type_1": None})
    rows.append({"id": "A", "date": pd.Timestamp("2020-02-01"), "y": 99, "split": "val", "weekday": "Saturday",
                 "month": 2, "snap_CA": 0, "event_name_1": None, "event_type_1": None})
    return pd.DataFrame(rows)


def test_load_train_excludes_val():
    assert (load_train(_toy())["split"] == "train").all()


def test_active_rows_drops_pre_launch_days():
    act = active_rows(load_train(_toy()))
    assert len(act) == 11 and act["y"].min() == 2


def test_weekday_table_has_seven_rows_and_mean_two():
    t = seasonality_tables(active_rows(load_train(_toy())))
    assert len(t["weekday"]) == 7 and abs(t["month"]["mean_y"].iloc[0] - 2.0) < 1e-12


def test_demand_speed_table_totals():
    f = pd.DataFrame({"demand_type": ["smooth", "lumpy", "lumpy"], "speed": ["fast", "slow", "slow"]})
    t = demand_speed_table(f)
    assert t.loc["total", "total"] == 3 and t.loc["lumpy", "slow"] == 2
