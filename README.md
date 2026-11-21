# Đề tài 17 – Dự báo nhu cầu xác suất và mô phỏng chính sách bổ sung hàng tồn kho (M5, CA_1 × FOODS_3)

Dự báo phân phối nhu cầu cửa sổ (lead time + chu kỳ đặt) cho 823 chuỗi, chọn mức order-up-to theo phân vị và mô phỏng tồn kho. Chia dữ liệu: train đến 2016-03-27, validation 28 ngày (2016-03-28→04-24), test 28 ngày (2016-04-25→05-22). Báo cáo từng giai đoạn: `results/reports/`; bản nháp báo cáo cuối: `results/reports/final_report.md`.

## Chạy lại toàn bộ từ đầu (Windows, PowerShell, Python 3.12)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # bản CPU

# Dữ liệu M5 gốc (calendar.csv, sales_train_evaluation.csv, sell_prices.csv) đặt trong data/raw/
python src/data.py                      # -> data/processed/ca1_foods3.parquet
python -m src.features                  # -> results/series_classes.csv, hình ADI-CV²
python -m src.run_forecasts --split val # ~7 phút CPU: tinh chỉnh + dự báo val -> hparams.json, forecast_eval_val_all.csv
python -m src.inventory_sim --split val # mô phỏng tồn kho val -> inventory_sim_val.csv
python -m src.policy_analysis           # q*, bảng chính sách, hình, bootstrap (val)
python -m src.run_forecasts --split test   # dùng hparams.json đã chốt (KHÔNG tinh chỉnh lại), ~4 phút
python -m src.inventory_sim --split test
python -m src.final_eval                # đánh giá chính sách chốt từ val trên test
# --- Bổ sung GĐ6 ---
python -m src.inventory_sim --split val --ext ; python -m src.inventory_sim --split test --ext   # lưới q phụ 0.3/0.4
python -m src.safety_point              # chính sách điểm có hệ số an toàn k vs phân vị (val chọn, test đánh giá), cả theo nhóm tốc độ
python -m src.calibration               # P(y<q̂), P(y<=q̂)
python -m src.ablation                  # LightGBM-Q bỏ giá/SNAP/sự kiện
python -m src.mlp_seeds                 # MLP-NB 3 seed + đường NLL (~16 phút)
# --- GĐ7 ---
python -m src.ablation --effect-cols   # bỏ riêng fut_wd_index, fut_event_index
# (sim/safety_point/policy_analysis/final_eval chạy lại sau run_forecasts: xem thứ tự ở trên; làm tròn LGBM-Q nằm trong quantile_to_levels)
# --- GĐ8: DeepAR-style (GRU + NB, lấy mẫu tự hồi quy) ---
# tuning 8 cấu hình song song (3 luồng mỗi tiến trình), chọn trên val; KHÔNG sửa results/hparams.json
for K in 0 1 2 3 4 5 6 7; do python -m src.deepar --stage tune_one --cfg-id $K --threads 3 & done; wait
python -m src.deepar --stage select                   # chọn cấu hình + kappa -> results/hparams_deepar.json
for S in 42 43 44; do python -m src.deepar --stage seed_val --seed $S --threads 4 & done; wait
python -m src.deepar --stage freeze                   # đóng băng epoch từng seed TRƯỚC khi chạy test
for S in 42 43 44; do python -m src.deepar --stage seed_test --seed $S --threads 4 & done; wait   # test: một lần
python -m src.deepar --stage collect_test
python -m src.deepar_analysis --what sim ; python -m src.deepar_analysis --what tables
python -m pytest -v                     # 64 test (cần đã chạy các bước trên vì có test đọc file kết quả)

# Notebook (Restart & Run All)
cd notebooks
python -m jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_classification.ipynb 03_baseline_stats.ipynb 04_final_results.ipynb
```
Seed (42) và mọi tham số (lưới phân vị, lead time, chi phí, lưới tinh chỉnh, cấu hình ML) nằm trong `configs/config.yaml`. Bootstrap của GĐ4 cần `python -m src.policy_analysis`; bootstrap trên test (`results/research_questions_test.csv`) sinh bằng `research_question_stats(..., split="test")` trong `src/policy_analysis.py`.

## Cấu trúc thư mục
- `configs/config.yaml` – cấu hình.
- `data/raw/`, `data/processed/` – dữ liệu (không đưa vào git).
- `src/` – (GĐ6: `safety_point.py`, `calibration.py`, `ablation.py`, `mlp_seeds.py`, `gd6_report_tables.py`) `data.py` (split), `features.py` (ADI/CV², speed), `eda.py`, `models_stats.py` + `forecast_prob.py` (baseline/legacy), `prob_models.py` (Panel, Croston/TSB/WindowAverage/SeasonalNaive, NB, chỉ số), `tuning.py`, `ml_models.py` (đặc trưng, LightGBM, MLP-NB), `run_forecasts.py`, `inventory_sim.py`, `policy_analysis.py`, `final_eval.py`, `report_utils.py`.
- `tests/` – pytest (64 test; `test_gd8.py` kiểm DeepAR và bất biến file cũ; gồm `test_inventory_reference.py`: so khớp mô phỏng với bản tham chiếu độc lập) (dự báo, rò rỉ, mô phỏng tồn kho, kiểm tra tích hợp file kết quả).
- `notebooks/` – 01 EDA, 02 phân loại, 03 baseline, 04 kết quả cuối.
- `results/` – kết quả, `results/figures/` hình, `results/tables/` bảng, `results/forecasts/{val,test}.npz` dự báo, `results/reports/` báo cáo.
- `archive/` – file tạm đã dọn (không dùng).

## File kết quả chính
| File | Nội dung |
|---|---|
| `results/series_classes.csv`, `results/tables/demand_speed_counts.csv` | phân loại chuỗi (train) |
| `results/forecasts/val.npz`, `test.npz` | phân vị (9 mức), trung bình, thực tế cho 7 mô hình × H∈{3,4} |
| `results/forecast_eval_val_all.csv`, `forecast_eval_test_all.csv` | pinball, coverage, WAPE, MASE theo nhóm |
| `results/hparams.json`, `tuning_log.csv` | siêu tham số chốt trên val |
| `results/inventory_sim_val.csv`, `inventory_sim_test.csv` | mô phỏng tồn kho mọi mô hình × q × L × cost_ratio |
| `results/q_star_val.csv`, `policy_table_val.csv` | q* và bảng chính sách đề xuất |
| `results/policy_eval_val_test.csv`, `test_qstar_transfer.csv`, `policy_eval_bootstrap.csv`, `research_questions_{val,test}.csv` | so sánh val/test, bootstrap |
| `results/safety_point_test.csv`, `q_star_ext_val.csv`, `inventory_sim_{val,test}_ext.csv` | GĐ6: điểm có hệ số an toàn vs phân vị, q* lưới mở rộng |
| `results/calibration_table.csv`, `calibration_summary.csv`, `ablation_exog.csv`, `mlp_seeds.csv`, `mlp_nll_curves.csv` | GĐ6: hiệu chuẩn rời rạc, ablation ngoại sinh, 3 seed MLP-NB |
| `results/figures/policy_*.png`, `final_*.png`, `eda_*.png` | biểu đồ (tiếng Việt) |
| `results/reports/00…08_*.md`, `final_report.md` | báo cáo |

## Lưu ý
- Chi phí mô phỏng là **tương đối** (giữ 1.0/đơn vị-ngày; thiếu = cost_ratio × 1.0), không phải chi phí thực.
- MLP-NB là MLP có đầu ra Negative Binomial, không phải DeepAR/N-HiTS/TFT. WRMSSE không được tính (lý do trong báo cáo 02).
- Lưu ý GĐ8: thêm DeepAR-style (`src/deepar.py`, `src/deepar_analysis.py`, siêu tham số `results/hparams_deepar.json`); mọi số liệu của 7 mô hình cũ KHÔNG đổi (sha256 các file kết quả cũ ở `results/gd8_baseline_hashes.txt`, tag git `pre-gd8`). DeepAR không hơn MLP-NB/LightGBM (xem `08_deepar.md`).
- Lưu ý GĐ7: mô phỏng làm tròn phân vị LGBM-Q về số nguyên gần nhất (NB giữ nguyên; chính sách điểm vẫn ceil); lưới kappa {0.25…2.0}; chính sách theo nhóm tốc độ không tốt hơn chính sách chung ở L=3 (xem `07_sua_loi.md`). Số liệu trước GĐ7: `archive/pre_gd7/`.
- Lưu ý GĐ6: kết luận "phân vị tốt hơn điểm" chỉ đúng so với điểm THUẦN; so với điểm có hệ số an toàn chỉ còn hơn ở tỷ lệ chi phí cao; biến ngoại sinh giá/SNAP/sự kiện đóng góp ≤ 1% pinball (xem `06_bo_sung.md`).
- Pipeline cũ (`forecast_prob.rolling_origin_eval`, `export_validation_outputs`) chỉ giữ để tham chiếu, không dùng cho kết quả chính thức.
