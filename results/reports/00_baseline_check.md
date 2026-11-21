# Báo cáo Giai đoạn 0 – Kiểm tra nền và dọn dẹp

## 1. Đã làm gì
- Chạy `pytest`, `python src/data.py`; đọc CSV kết quả, kiểm tra parquet, notebook, config.
- Sửa `.gitignore`: thêm `*.zip`, `.pytest_cache/`, `data/raw/` (đã có sẵn `data/`, `.venv/`, `__pycache__/`). Chưa xóa/di chuyển file nào.
- Tạo thư mục `results/reports/` và file này.
- Độ dài mã hiện có: data.py 125, features.py 134, forecast_prob.py 356, models_stats.py 218, utils.py 23, test_forecast_prob.py 96 dòng.

## 2. Lệnh đã chạy và output
`python -m pytest -v` (Python 3.12.10, pytest 9.2.0): **7 passed, 0 failed, 0 skipped** trong 1.33s
```
test_daily_target_date_tracks_horizon_offset PASSED
test_evaluate_validation_summary_matches_by_date_and_id PASSED
test_join_rows_do_not_duplicate PASSED
test_n_samples_changes_quantile_output PASSED
test_pinball_known_answer PASSED
test_poisson_distribution_scaling PASSED
test_sample_total_window_does_not_blow_up_for_zero_mean PASSED
```
`python src/data.py` (7.3s):
```
shape: (1597443, 15)
so chuoi: 823 ; so ngay moi chuoi: min=1941, max=1941
split rows: train 1551355, val 23044, test 23044
test: 2016-04-25..2016-05-22 ; train: 2011-01-29..2016-03-27 ; val: 2016-03-28..2016-04-24
ty le sell_price khuyet: 19.8606%
```

## 3. Xác minh hiện trạng
| Mục | Kết quả thực tế |
|---|---|
| Số chuỗi | 823 (CA_1 × FOODS_3) ✔ |
| Split | val 28 ngày (28/03–24/04/2016), test 28 ngày (25/04–22/05/2016), train 1885 ngày ✔, không chồng lấn |
| Seed | `config.yaml` seed=42; `data.py`, `forecast_prob.py`, `models_stats.py` đều đọc từ config ✔ (không hard-code) |
| Config thiếu | Chưa có `holding_cost`, `stockout_cost`, `lead_times`, `initial_inventory_rule` (đúng như dự kiến) |
| Notebook | 01_eda: 2 ô code, 0 có output; 02_classification: 2 ô, 0 output; 03_baseline_stats: 1 ô, 0 output ✔ chưa có output |
| `train_lightgbm_quantile` | Có tại `src/forecast_prob.py:103`, chưa có file kết quả LightGBM ✔ |
| Tests | 7 test, chỉ phủ `forecast_prob`; chưa có test cho data/features/models_stats |
| Thư viện | lightgbm 4.7.0, statsforecast 2.1.1, scipy 1.18.1, pandas 2.3.3, numpy 2.5.4, matplotlib 3.11.2, nbconvert 7.17.2. **Chưa cài**: torch, neuralforecast, gluonts |

Bảng demand_type × speed (từ `results/series_classes.csv`, 823 chuỗi):

| demand_type | fast | medium | slow | Tổng |
|---|---|---|---|---|
| smooth | 118 | 2 | 0 | 120 |
| erratic | 45 | 0 | 0 | 45 |
| intermittent | 74 | 203 | 142 | 419 |
| lumpy | 89 | 116 | 34 | 239 |

Coverage NB trên validation (`results/forecast_eval_val.csv`, overall) – xác nhận vấn đề (a):

| q | daily | window |
|---|---|---|
| 0.5 | 0.6748 | 0.6050 |
| 0.7 | 0.8192 | 0.7554 |
| 0.8 | 0.8814 | 0.8325 |
| 0.9 | 0.9358 | 0.9055 |
| 0.95 | 0.9565 | 0.9382 |
| 0.99 | 0.9760 | 0.9632 |

→ Dạng daily lệch lên (quá rộng) ở mọi q, nặng nhất ở q thấp (+0.175 tại q=0.5, +0.119 tại q=0.7). Dạng window lệch ít hơn nhưng vẫn vượt 0.05 ở q=0.5 (+0.105) và q=0.7 (+0.055).

Baseline thống kê trên val (`results/baseline_stats_val.csv`, overall): WindowAverage WAPE 0.6285 / MASE 0.8783 / pinball_q50 0.8408; TSB 0.6305 / 0.8808 / 0.8434; CrostonSBA 0.6921 / 0.9560 / 0.9259; SeasonalNaive 0.7398 / 1.0394 / 0.9896. Chỉ có pinball q50 → xác nhận vấn đề (b). Chưa có kết quả test, học sâu, mô phỏng, WRMSSE → xác nhận (c).

## 4. File kết quả hiện có
`results/baseline_stats_val.csv`, `forecast_eval_val.csv`, `forecast_eval_val_smoke.csv`, `series_classes.csv`, `window_quantiles_val.parquet`, `check_report.txt`, `figures/adi_cv2_scatter.png`.

## 5. Cấu hình
Seed 42; lead_time 2; review_period 1; quantile_levels [0.5,0.7,0.8,0.9,0.95,0.99]; n_samples 500; refit_every 7. Chạy CPU.

## 6. Vấn đề / bất thường
- **File thừa** (đề xuất, CHƯA xóa):
  | File | Lý do |
  |---|---|
  | `tmp_check.py` | script 3 dòng in shape calendar, tạm |
  | `tmp_diag.py`, `tmp_diag_output.txt` | chẩn đoán tạm |
  | `results/series_classes_old.csv` | cùng id với bản hiện tại nhưng **khác 57 demand_type và 145 speed**, adi lệch tối đa 80.22 (cv2 giống) → bản cũ, tính theo cách khác; không file nào nên dùng |
  | `results/forecast_eval_val_smoke.csv` | kết quả smoke test, trùng cấu trúc với `forecast_eval_val.csv` |
  | `m5-forecasting-accuracy.zip` (48 MB) | gói dữ liệu gốc đã giải nén trong `data/raw/` (đã ignore); giữ hay không do bạn quyết |
  | `data/raw/sample_submission.csv`, `data/raw/sales_train_validation.csv` | không được tham chiếu (grep). LƯU Ý: `sales_train_evaluation.csv` **ĐƯỢC DÙNG** bởi `src/data.py` dòng 36 và 72 (id có hậu tố `_evaluation`) – ghi chú ban đầu "không dùng" là SAI, đã sửa |
- Thư mục này **không phải git repo** (`.gitignore` hiện chưa có tác dụng cho đến khi `git init`).
- `check_report.txt` cho biết TSB từng bị thiếu trong CSV do lỗi `make_model_list()` đã sửa; không còn ảnh hưởng.
- README mô tả "Giai đoạn A đang phát triển" – sẽ cập nhật ở Giai đoạn 5.
- 19.86% `sell_price` khuyết (mặt hàng chưa bán/chưa niêm yết) – cần quy tắc xử lý ở LightGBM/học sâu và ở mô phỏng (Giai đoạn 2).
- Chưa có WRMSSE; tính được cần giá × doanh số train (có sẵn), sẽ làm ở Giai đoạn 2.5 nếu đúng công thức M5 cho một cửa hàng/ một dept (lưu ý: chỉ 823 chuỗi, không có đủ 12 cấp gộp của M5 → có thể chỉ tính WRMSSE cấp 12 bản địa; sẽ nói rõ).

## 7. Cần bạn xác nhận
1. Xóa hay chuyển vào `archive/`? Đề xuất chuyển `archive/`: `tmp_check.py`, `tmp_diag.py`, `tmp_diag_output.txt`, `series_classes_old.csv`, `forecast_eval_val_smoke.csv`. Còn zip và 2 file raw thừa: giữ nguyên?
2. Có `git init` thư mục này không?
3. Đồng ý sang Giai đoạn 1 (EDA mùa vụ + phân loại)?

---
## Cập nhật sau xác nhận của người dùng
- Đã chuyển vào `archive/` (grep trước: không còn tham chiếu trong src/tests/notebooks, trừ `src/forecast_prob.py:335` chỉ *ghi* ra `forecast_eval_val_smoke.csv`, sẽ được thay bằng đầu ra mới ở GĐ2): `tmp_check.py`, `tmp_diag.py`, `tmp_diag_output.txt`, `series_classes_old.csv`, `forecast_eval_val_smoke.csv`, `data/raw/sample_submission.csv`, `data/raw/sales_train_validation.csv`. Giữ nguyên `sales_train_evaluation.csv` và zip.
- `git init` đã chạy; `archive/` đã thêm vào `.gitignore`. Tác giả commit: "Thành".
- Chạy lại sau dọn dẹp: `python -m pytest -v` → 7 passed, 0 failed, 0 skipped (7 test như mục 2); `python src/data.py` → 823 chuỗi, train 1551355 / val 23044 / test 23044 dòng, sell_price khuyết 19.8606%.
