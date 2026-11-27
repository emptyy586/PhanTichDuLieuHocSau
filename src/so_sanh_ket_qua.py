"""So sánh số liệu các file CSV giữa hai thư mục (ví dụ results_backup và results).
Chạy:  python so_sanh_ket_qua.py results_backup results"""
import pathlib
import sys

import numpy as np
import pandas as pd

a, b = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for f in sorted(a.glob("*.csv")):
    g = b / f.name
    if not g.exists():
        print("THIẾU   ", f.name)
        continue
    x, y = pd.read_csv(f), pd.read_csv(g)
    if x.shape != y.shape:
        print("KHÁC KÍCH THƯỚC", f.name, x.shape, y.shape)
        continue
    num = x.select_dtypes("number").columns
    d = float(np.nanmax(np.abs(x[num].to_numpy() - y[num].to_numpy()))) if len(num) else 0.0
    print(("OK      " if d < 1e-6 else "LỆCH    ") + f.name + f"  (chênh lớn nhất {d:.2e})")
