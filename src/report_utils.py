"""Tiện ích dựng báo cáo markdown: bảng từ DataFrame và chèn output pytest nguyên văn."""
from __future__ import annotations

import subprocess
import sys

import pandas as pd

from src.utils import ROOT


def md_table(df: pd.DataFrame, digits: int = 4, index: bool = True) -> str:
    """Chuyển DataFrame thành bảng markdown, làm tròn số thực tới `digits` chữ số thập phân."""
    d = df.reset_index() if index else df.copy()
    d.columns = [str(c) for c in d.columns]
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def run_pytest_verbose() -> str:
    """Chạy `python -m pytest -v` và trả về output nguyên văn."""
    out = subprocess.run([sys.executable, "-m", "pytest", "-v", "--color=no"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8")
    return out.stdout
