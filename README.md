# Đề tài 17 – Dự báo nhu cầu xác suất và mô phỏng chính sách bổ sung hàng tồn kho (M5, CA_1 × FOODS_3)

Dự báo phân phối nhu cầu cửa sổ (lead time + chu kỳ đặt) cho 823 chuỗi, chọn mức order-up-to theo phân vị và mô phỏng tồn kho. Chia dữ liệu: train đến 2016-03-27, validation 28 ngày (2016-03-28→04-24), test 28 ngày (2016-04-25→05-22). Báo cáo từng giai đoạn: `results/reports/`; bản nháp báo cáo cuối: `results/reports/bao_cao_cuoi_cung.md`.

## Chạy lại toàn bộ từ đầu (Windows, PowerShell, Python 3.12)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # bản CPU

# Dữ liệu M5 gốc (calendar.csv, sales_train_evaluation.csv, sell_prices.csv) đặt trong data/raw/
python src/du_lieu.py                      # -> data/processed/ca1_foods3.parquet
python -m src.dac_trung                  # -> results/phan_loai_chuoi.csv, hình ADI-CV²
python -m src.chay_du_bao --split val # ~7 phút CPU: tinh chỉnh + dự báo val -> sieu_tham_so.json, danh_gia_du_bao_val_all.csv
python -m src.mo_phong_ton_kho --split val # mô phỏng tồn kho val -> mo_phong_ton_kho_val.csv
python -m src.phan_tich_chinh_sach           # q*, bảng chính sách, hình, bootstrap (val)
python -m src.chay_du_bao --split test   # dùng sieu_tham_so.json đã chốt (KHÔNG tinh chỉnh lại), ~4 phút
python -m src.mo_phong_ton_kho --split test
python -m src.danh_gia_cuoi                # đánh giá chính sách chốt từ val trên test
# --- Bổ sung GĐ6 ---
python -m src.mo_phong_ton_kho --split val --ext ; python -m src.mo_phong_ton_kho --split test --ext   # lưới q phụ 0.3/0.4
python -m src.diem_an_toan              # chính sách điểm có hệ số an toàn k vs phân vị (val chọn, test đánh giá), cả theo nhóm tốc độ
python -m src.hieu_chinh               # P(y<q̂), P(y<=q̂)
python -m src.loai_bo_dac_trung                  # LightGBM-Q bỏ giá/SNAP/sự kiện
python -m src.mlp_cac_seed                 # MLP-NB 3 seed + đường NLL (~16 phút)
# --- GĐ7 ---
python -m src.loai_bo_dac_trung --effect-cols   # bỏ riêng fut_wd_index, fut_event_index
# (mo_phong_ton_kho/diem_an_toan/phan_tich_chinh_sach/danh_gia_cuoi chạy lại sau chay_du_bao: xem thứ tự ở trên; làm tròn LGBM-Q nằm trong quantile_to_levels)
# --- GĐ8: DeepAR-style (GRU + NB, lấy mẫu tự hồi quy) ---
# tuning 8 cấu hình song song (3 luồng mỗi tiến trình), chọn trên val; KHÔNG sửa results/sieu_tham_so.json
for K in 0 1 2 3 4 5 6 7; do python -m src.deepar --stage tune_one --cfg-id $K --threads 3 & done; wait
python -m src.deepar --stage select                   # chọn cấu hình + kappa -> results/sieu_tham_so_deepar.json
for S in 42 43 44; do python -m src.deepar --stage seed_val --seed $S --threads 4 & done; wait
python -m src.deepar --stage freeze                   # đóng băng epoch từng seed TRƯỚC khi chạy test
for S in 42 43 44; do python -m src.deepar --stage seed_test --seed $S --threads 4 & done; wait   # test: một lần
python -m src.deepar --stage collect_test
python -m src.deepar_phan_tich --what sim ; python -m src.deepar_phan_tich --what tables
# --- GĐ9/9b: vẽ lại TOÀN BỘ hình cho báo cáo Word (6.3 inch, 300 dpi; không đổi số liệu). Chạy SAU phan_tich_chinh_sach/deepar_phan_tich vì các bước đó cũng ghi cùng tên file hình ---
python -m src.ve_hinh_bao_cao
python -m pytest -v                     # 64 test (cần đã chạy các bước trên vì có test đọc file kết quả)

# Notebook (Restart & Run All)
cd notebooks
python -m jupyter nbconvert --to notebook --execute --inplace 01_kham_pha_du_lieu.ipynb 02_phan_loai.ipynb 03_thong_ke_co_so.ipynb 04_ket_qua_cuoi.ipynb
```
Seed (42) và mọi tham số (lưới phân vị, lead time, chi phí, lưới tinh chỉnh, cấu hình ML) nằm trong `configs/thong_so.yaml`. Bootstrap của GĐ4 cần `python -m src.phan_tich_chinh_sach`; bootstrap trên test (`results/cau_hoi_nghien_cuu_test.csv`) sinh bằng `research_question_stats(..., split="test")` trong `src/phan_tich_chinh_sach.py`.

## Cấu trúc thư mục
- `configs/thong_so.yaml` – cấu hình.
- `data/raw/`, `data/processed/` – dữ liệu (không đưa vào git).
- `src/` – (GĐ6: `diem_an_toan.py`, `hieu_chinh.py`, `loai_bo_dac_trung.py`, `mlp_cac_seed.py`, `bang_bao_cao_gd6.py`) `du_lieu.py` (split), `dac_trung.py` (ADI/CV², speed), `kham_pha_du_lieu.py`, `mo_hinh_thong_ke.py` + `du_bao_xac_suat.py` (baseline/legacy), `mo_hinh_xac_suat.py` (Panel, Croston/TSB/WindowAverage/SeasonalNaive, NB, chỉ số), `tinh_chinh.py`, `mo_hinh_ml.py` (đặc trưng, LightGBM, MLP-NB), `chay_du_bao.py`, `mo_phong_ton_kho.py`, `phan_tich_chinh_sach.py`, `danh_gia_cuoi.py`, `tien_ich_bao_cao.py`.
- `tests/` – pytest (64 test, cấu hình trong `pytest.ini`; `kiem_thu_gd8.py` kiểm DeepAR và bất biến file cũ; gồm `kiem_thu_mo_phong_tham_chieu.py`: so khớp mô phỏng với bản tham chiếu độc lập) (dự báo, rò rỉ, mô phỏng tồn kho, kiểm tra tích hợp file kết quả).
- `notebooks/` – 01 EDA, 02 phân loại, 03 baseline, 04 kết quả cuối.
- `results/` – kết quả, `results/figures/` hình, `results/tables/` bảng, `results/forecasts/{val,test}.npz` dự báo, `results/reports/` báo cáo.
- `archive/` – file tạm đã dọn (không dùng).

## File kết quả chính
| File | Nội dung |
|---|---|
| `results/phan_loai_chuoi.csv`, `results/tables/so_luong_nhom_toc_do_nhu_cau.csv` | phân loại chuỗi (train) |
| `results/forecasts/val.npz`, `test.npz` | phân vị (9 mức), trung bình, thực tế cho 7 mô hình × H∈{3,4} |
| `results/danh_gia_du_bao_val_all.csv`, `danh_gia_du_bao_test_all.csv` | pinball, coverage, WAPE, MASE theo nhóm |
| `results/sieu_tham_so.json`, `nhat_ky_tinh_chinh.csv` | siêu tham số chốt trên val |
| `results/mo_phong_ton_kho_val.csv`, `mo_phong_ton_kho_test.csv` | mô phỏng tồn kho mọi mô hình × q × L × cost_ratio |
| `results/q_sao_val.csv`, `bang_chinh_sach_val.csv` | q* và bảng chính sách đề xuất |
| `results/danh_gia_chinh_sach_val_test.csv`, `chuyen_q_sao_test.csv`, `danh_gia_chinh_sach_bootstrap.csv`, `cau_hoi_nghien_cuu_{val,test}.csv` | so sánh val/test, bootstrap |
| `results/diem_an_toan_test.csv`, `q_sao_ext_val.csv`, `mo_phong_ton_kho_{val,test}_ext.csv` | GĐ6: điểm có hệ số an toàn vs phân vị, q* lưới mở rộng |
| `results/hieu_chinh_bang.csv`, `hieu_chinh_tom_tat.csv`, `loai_bo_bien_ngoai_sinh.csv`, `mlp_cac_seed.csv`, `mlp_duong_nll.csv` | GĐ6: hiệu chuẩn rời rạc, ablation ngoại sinh, 3 seed MLP-NB |
| `results/figures/chinh_sach_*.png`, `cuoi_*.png`, `kham_pha_*.png` | biểu đồ (tiếng Việt) |
| `results/reports/00…08_*.md`, `bao_cao_cuoi_cung.md` | báo cáo |

## Lưu ý
- Chi phí mô phỏng là **tương đối** (giữ 1.0/đơn vị-ngày; thiếu = cost_ratio × 1.0), không phải chi phí thực.
- MLP-NB là MLP có đầu ra Negative Binomial, không phải DeepAR/N-HiTS/TFT. WRMSSE không được tính (lý do trong báo cáo 02).
- Lưu ý GĐ9/9b: MỌI hình báo cáo Word do `src/ve_hinh_bao_cao.py` vẽ (`python -m src.ve_hinh_bao_cao`; gồm hình chính sách/dự báo, `cuoi_*`, `kham_pha_*`, `adi_cv2_phan_tan`, sơ đồ DeepAR, đường NLL; không tiền tố số hình, có bản `_test`, heatmap q* chia 2 hình `chinh_sach_heatmap_q_sao.png`/`_b.png`); không đổi CSV/npz/json. Notebook 01/02 và các bước phân tích cũ cũng ghi cùng tên file hình ở phong cách cũ, vì vậy chạy `src.ve_hinh_bao_cao` SAU CÙNG.
- Lưu ý GĐ8: thêm DeepAR-style (`src/deepar.py`, `src/deepar_phan_tich.py`, siêu tham số `results/sieu_tham_so_deepar.json`); mọi số liệu của 7 mô hình cũ KHÔNG đổi (sha256 các file kết quả cũ ở `results/ma_bam_co_so_gd8.txt`, tag git `pre-gd8`). DeepAR không hơn MLP-NB/LightGBM (xem `08_deepar.md`).
- Lưu ý GĐ7: mô phỏng làm tròn phân vị LGBM-Q về số nguyên gần nhất (NB giữ nguyên; chính sách điểm vẫn ceil); lưới kappa {0.25…2.0}; chính sách theo nhóm tốc độ không tốt hơn chính sách chung ở L=3 (xem `07_sua_loi.md`). Số liệu trước GĐ7: `archive/pre_gd7/`.
- Lưu ý GĐ6: kết luận "phân vị tốt hơn điểm" chỉ đúng so với điểm THUẦN; so với điểm có hệ số an toàn chỉ còn hơn ở tỷ lệ chi phí cao; biến ngoại sinh giá/SNAP/sự kiện đóng góp ≤ 1% pinball (xem `06_bo_sung.md`).
- Pipeline cũ (`du_bao_xac_suat.rolling_origin_eval`, `export_validation_outputs`) chỉ giữ để tham chiếu, không dùng cho kết quả chính thức.

## Chỉ mục kết quả
Bảng chỉ mục các file trong `results/` (file nào do lệnh nào tạo, file đóng băng): xem [`results/README.md`](results/README.md).
