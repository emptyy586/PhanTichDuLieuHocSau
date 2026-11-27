from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho numpy và random để tái lập được kết quả."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path | None = None) -> dict:
    """Đọc file cấu hình YAML từ dự án."""
    config_path = Path(path) if path is not None else ROOT / "configs" / "thong_so.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
