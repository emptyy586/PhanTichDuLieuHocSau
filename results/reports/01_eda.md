# Báo cáo GĐ1 – EDA mùa vụ và phân loại chuỗi

## 1. Đã làm gì
- Tạo `src/eda.py` (hàm: `load_train`, `active_rows`, `seasonality_tables`, `plot_all`, `demand_speed_table`) và `tests/test_eda.py` (4 test).
- Viết lại sạch `notebooks/01_eda.ipynb` và `notebooks/02_classification.ipynb` (bản cũ có ô code đặt sai thứ tự, chưa có output; EDA cũ dùng cả val/test). Bản mới **chỉ dùng split train**, tính từ ngày bán đầu tiên mỗi chuỗi. Đã chạy `jupyter nbconvert --execute --inplace` (kernel python3), 0 ô lỗi.
- Notebook 02 kiểm tra `results/series_classes.csv` khớp với tính lại từ `compute_series_features` (assert pass).
- Không sửa `src/features.py`. `03_baseline_stats.ipynb` chưa động tới (cell cũ gọi `compute_metrics` sai số tham số; sẽ thay ở GĐ2).

## 2. Lệnh đã chạy và pytest
`jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_classification.ipynb`; `python -m pytest -v`:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\lt\261\PTDLHS\DA\.venv\Scripts\python.exe
cachedir: .pytest_cache
Fugue tests will be initialized with options:
rootdir: d:\lt\261\PTDLHS\DA
plugins: anyio-4.15.1, fugue-0.9.7, platformdirs-4.14.0
collecting ... collected 11 items

tests/test_eda.py::test_load_train_excludes_val PASSED                   [  9%]
tests/test_eda.py::test_active_rows_drops_pre_launch_days PASSED         [ 18%]
tests/test_eda.py::test_weekday_table_has_seven_rows_and_mean_two PASSED [ 27%]
tests/test_eda.py::test_demand_speed_table_totals PASSED                 [ 36%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_daily_target_date_tracks_horizon_offset PASSED [ 45%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_evaluate_validation_summary_matches_by_date_and_id PASSED [ 54%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_join_rows_do_not_duplicate PASSED [ 63%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_n_samples_changes_quantile_output PASSED [ 72%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_pinball_known_answer PASSED [ 81%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_poisson_distribution_scaling PASSED [ 90%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_sample_total_window_does_not_blow_up_for_zero_mean PASSED [100%]

============================= 11 passed in 1.96s ==============================

```

## 3. Kết quả chính (copy từ `results/tables/*.csv`; train, sau ngày ra mắt; 1 232 461 dòng chuỗi-ngày / 823 chuỗi)
Số chuỗi theo demand_type × speed (`results/tables/demand_speed_counts.csv`):

| demand_type | fast | medium | slow | total |
|---|---|---|---|---|
| erratic | 45 | 0 | 0 | 45 |
| intermittent | 74 | 203 | 142 | 419 |
| lumpy | 89 | 116 | 34 | 239 |
| smooth | 118 | 2 | 0 | 120 |
| total | 326 | 321 | 176 | 823 |

Doanh số trung bình (đơn vị/chuỗi/ngày) theo thứ (`eda_weekday.csv`): Thứ 2 2.9874; Thứ 3 2.6134; Thứ 4 2.5597; Thứ 5 2.5949; Thứ 6 3.1364; Thứ 7 3.9512; CN 4.1099.
Theo tháng (`eda_month.csv`): tháng 1 2.8860 … tháng 8 3.4019 (cao nhất), tháng 12 2.8454 (thấp nhất). SNAP: 0 → 3.0195, 1 → 3.3778. Ngày có sự kiện 3.0128 vs không 3.1480.

### Nhận xét (có số liệu)
1. **Mùa vụ tuần rất rõ**: CN (chỉ số 1.3102) và Thứ 7 (1.2596) cao hơn Thứ 4 (0.8160) khoảng 1.6 lần (4.1099/2.5597). → Dự báo cửa sổ 3–4 ngày phải thấy được thứ trong tuần; WindowAverage 28 ngày tự động cân tuần nhưng cửa sổ L+R ngắn thì không.
2. **Mùa vụ năm vừa phải**: chỉ số tháng từ 0.9071 (tháng 12) đến 1.0845 (tháng 8).
3. **SNAP**: ngày SNAP cao hơn 11.9% (3.3778/3.0195 − 1). Cờ "có sự kiện" nói chung không tăng doanh số (3.0128 < 3.1480) vì gộp nhiều loại ngược chiều: National 0.8821, Sporting 1.0715. Sự kiện cụ thể phân hóa mạnh: OrthodoxEaster 1.3641, SuperBowl 1.3460, LaborDay 1.3299; Thanksgiving 0.6986, NewYear 0.6987, Christmas 0.0000 (cửa hàng đóng, doanh số bằng 0 cả ngày). → Cần đặc trưng sự kiện theo tên/loại cho LightGBM chứ không chỉ cờ nhị phân.
4. **Phân bố tốc độ/độ gián đoạn**: 419/823 chuỗi (50.9%) là intermittent và 239 (29.0%) là lumpy; chỉ 120 smooth + 45 erratic. Median zero_rate: smooth 0.1390, intermittent 0.5094. Nhóm slow gần như toàn intermittent (142/176) hoặc lumpy (34/176); medium/slow chiếm 497/823 = 60.4% số chuỗi.
5. **Nhóm fast** có trung bình tối đa 66.7268 đơn vị/ngày nhưng trung vị chỉ 3.6981 → đuôi phải rất dài, cần cẩn thận khi gộp (nên báo cáo theo nhóm, không chỉ overall).

## 4. File sinh ra
`results/figures/eda_trend_total.png`, `eda_weekday.png`, `eda_month.png`, `eda_snap_event.png`, `eda_event_names.png`, `eda_zero_rate.png`, `adi_cv2_scatter.png` (vẽ lại); `results/tables/eda_{weekday,month,snap,event_flag,event_type,event_name}.csv`, `results/tables/demand_speed_counts.csv`.

## 5. Cấu hình
Seed 42 (không có bước ngẫu nhiên). ADI ngưỡng 1.32, CV² 0.49 (config). Python 3.12.10, pandas 2.3.3, matplotlib 3.11.2, nbconvert 7.17.2.

## 6. Vấn đề / giả định
- Hình 5 (chỉ số theo sự kiện) chỉ dùng `event_name_1`; cột `event_name_2` không được tải trong `data.py` nên bỏ qua (ít ngày hơn).
- Chỉ số sự kiện là trung bình thô, chưa kiểm soát thứ trong tuần (ví dụ ngày lễ rơi vào cuối tuần) → chỉ là mô tả, không phải hiệu ứng nhân quả.
- Ngày Christmas có doanh số 0 do đóng cửa; mô phỏng ở val/test không chứa Christmas (28/03–22/05).
- Ngưỡng speed (q33/q67 doanh số trung bình trên train, tính theo toàn bộ chuỗi gồm cả ngày trước ra mắt) là thiết kế cũ, giữ nguyên.

## 7. Chưa làm
Chưa phân tích tương tác giá–doanh số (không yêu cầu ở GĐ1).
