from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT, load_config


CONFIG = load_config(ROOT / "configs" / "config.yaml")
ADI_THRESHOLD = float(CONFIG["adi_threshold"])
CV2_THRESHOLD = float(CONFIG["cv2_threshold"])


def classify_demand(adi: float, cv2: float) -> str:
    """Phân loại theo tiêu chí Syntetos-Boylan."""
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "smooth"
    if adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        return "erratic"
    if adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "intermittent"
    return "lumpy"


def classify_speed(avg_demand: float, q33: float, q67: float) -> str:
    """Phân loại tốc độ bán theo tam phân vị doanh số trung bình."""
    if avg_demand <= q33:
        return "slow"
    if avg_demand <= q67:
        return "medium"
    return "fast"


def compute_series_features(data: pd.DataFrame) -> pd.DataFrame:
    """Tính ADI, CV², doanh số trung bình và phân loại chuỗi trên tập train."""
    train = data[data["split"] == "train"].copy()
    train = train.sort_values(["id", "date"]).reset_index(drop=True)

    rows: list[dict] = []
    q33 = train.groupby("id")["y"].mean().quantile(0.33)
    q67 = train.groupby("id")["y"].mean().quantile(0.67)

    for item_id, series in train.groupby("id"):
        series = series.sort_values("date").copy()
        nonzero_mask = series["y"] > 0
        if nonzero_mask.any():
            first_sale_date = series.loc[nonzero_mask, "date"].min()
            series = series[series["date"] >= first_sale_date].copy()
        y = series["y"].astype(float).to_numpy()
        total_days = len(y)
        nonzero = y[y > 0]

        if len(nonzero) == 0:
            adi = np.inf
            cv2 = np.inf
            avg_demand = 0.0
            zero_rate = 1.0
        else:
            adi = total_days / len(nonzero)
            cv2 = float(np.var(nonzero, ddof=1) / np.mean(nonzero) ** 2) if len(nonzero) > 1 else 0.0
            avg_demand = float(np.mean(y))
            zero_rate = float(np.mean(y == 0))

        demand_type = classify_demand(adi, cv2)
        speed = classify_speed(avg_demand, q33, q67)

        rows.append(
            {
                "id": item_id,
                "adi": adi,
                "cv2": cv2,
                "avg_demand": avg_demand,
                "zero_rate": zero_rate,
                "demand_type": demand_type,
                "speed": speed,
            }
        )

    return pd.DataFrame(rows)


def save_series_classes(data: pd.DataFrame, output_path: Path | str) -> pd.DataFrame:
    """Tạo csv kết quả phân loại chuỗi và in bảng tóm tắt."""
    features = compute_series_features(data)
    features.to_csv(output_path, index=False)

    counts = features["demand_type"].value_counts().sort_index()
    speed_by_demand = pd.crosstab(features["demand_type"], features["speed"])

    print("so chuoi theo demand_type:")
    print(counts)
    print("\nma tran cheo demand_type x speed:")
    print(speed_by_demand)

    return features


def plot_adi_cv2(features: pd.DataFrame, output_path: Path | str) -> None:
    """Vẽ đồ thị phân tán ADI-CV² với đường ngưỡng và màu theo demand_type."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for demand_type, group in features.groupby("demand_type"):
        ax.scatter(group["adi"], group["cv2"], s=18, alpha=0.8, label=demand_type)

    ax.axvline(ADI_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.axhline(CV2_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ADI")
    ax.set_ylabel("CV²")
    ax.set_title("Hình 2.1. Phân bố ADI - CV² theo demand_type")
    ax.legend(title="Demand type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    data = pd.read_parquet(ROOT / "data/processed/ca1_foods3.parquet")
    features = save_series_classes(data, ROOT / "results/series_classes.csv")
    plot_adi_cv2(features, ROOT / "results/figures/adi_cv2_scatter.png")

    zero_series = features[features["zero_rate"] == 1.0]
    if not zero_series.empty:
        print(f"so chuoi toan 0 trong train: {len(zero_series)}")
        print("de xuat: khong xoa trong pipeline; danh dau la all-zero, de phan loai 'intermittent' hoac 'lumpy' va can duoc xe ly trong mo hinh sau.")
    else:
        print("khong co chuoi toan 0 trong train.")


if __name__ == "__main__":
    main()
