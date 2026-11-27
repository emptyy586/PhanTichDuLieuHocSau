# Báo cáo GĐ7 – Sửa lỗi phát hiện khi kiểm tra chéo

## 1. Đã làm gì (file tạo/sửa)
- **Việc 1 – làm tròn LGBM-Q**: thêm `quantile_to_levels` (`src/mo_phong_ton_kho.py`): phân vị NB giữ nguyên (đã là số nguyên), phân vị LGBM-Q làm tròn gần nhất `max(0, floor(q̂+0.5))`; dùng thống nhất trong `mo_phong_ton_kho.py`, `diem_an_toan.py`, `phan_tich_chinh_sach.py`, `danh_gia_cuoi.py`. Chính sách điểm giữ `ceil` (xem mục 3.1).
- **Việc 2 – lưới kappa**: `configs/thong_so.yaml` `tune_kappa` → {0.25, 0.35, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0}; chạy lại `run_forecasts` (val tinh chỉnh, test dùng siêu tham số đã chốt) và `mlp_seeds`.
- **Việc 3**: `src/diem_an_toan.py` thêm `evaluate_by_speed` (k*, q* chọn riêng trong từng nhóm speed → `results/diem_an_toan_theo_toc_do_test.csv`); `src/loai_bo_dac_trung.py` thêm `effect_mask`/`--effect-cols` (→ `results/loai_bo_hieu_ung_cot.csv`); `src/danh_gia_cuoi.py` ghi thêm `cau_hoi_nghien_cuu_test.csv`.
- **Việc 4 – test**: `tests/kiem_thu_mo_phong_tham_chieu.py` (10 test, đúng nội dung được cung cấp), thay test tautology `test_inventory_conservation_every_day` bằng `test_order_accounting_and_inventory_position_after_ordering` (`tests/kiem_thu_mo_phong_ton_kho.py`), `tests/kiem_thu_gd7.py` (5 test: làm tròn LGBM-Q, NB không đổi, mức mô phỏng nguyên/không âm trên dự báo thật, đơn điệu kappa, mặt nạ cột hiệu ứng), cập nhật `tests/kiem_thu_danh_gia_cuoi.py` (NB là số nguyên).
- **Báo cáo/tài liệu**: `bao_cao_cuoi_cung.md` dựng lại hoàn toàn từ CSV mới; `05_kiem_tra_cuoi.md`, `06_bo_sung.md` dựng lại; `README.md` cập nhật; `02`, `03`, `04` giữ nguyên nội dung cũ (có ghi chú đầu trang: số liệu trước GĐ7). Bản cũ của mọi CSV/báo cáo/hparams ở `archive/pre_gd7/`.
- Số liệu chạy: `run_forecasts` val/test, `inventory_sim` val/test (cả `--ext`), `safety_point`, `policy_analysis`, `final_eval`, `calibration`, `mlp_seeds`, `ablation --effect-cols`.

## 2. Lệnh đã chạy và pytest
```
python -m src.chay_du_bao --split val ; python -m src.chay_du_bao --split test
python -m src.mo_phong_ton_kho --split val ; ... --split test ; ... --split val --ext ; ... --split test --ext
python -m src.phan_tich_chinh_sach ; python -m src.danh_gia_cuoi ; python -m src.hieu_chinh ; python -m src.diem_an_toan
python -m src.mlp_cac_seed ; python -m src.loai_bo_dac_trung --effect-cols
python -m pytest -v
```
OUTPUT NGUYÊN VĂN `python -m pytest -v`:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\lt\261\PTDLHS\DA\.venv\Scripts\python.exe
cachedir: .pytest_cache
Fugue tests will be initialized with options:
rootdir: D:\lt\261\PTDLHS\DA
plugins: anyio-4.15.1, fugue-0.9.7, platformdirs-4.14.0
collecting ... collected 56 items

tests/kiem_thu_kham_pha_du_lieu.py::test_load_train_excludes_val PASSED                   [  1%]
tests/kiem_thu_kham_pha_du_lieu.py::test_active_rows_drops_pre_launch_days PASSED         [  3%]
tests/kiem_thu_kham_pha_du_lieu.py::test_weekday_table_has_seven_rows_and_mean_two PASSED [  5%]
tests/kiem_thu_kham_pha_du_lieu.py::test_demand_speed_table_totals PASSED                 [  7%]
tests/kiem_thu_danh_gia_cuoi.py::test_group_policy_levels_follow_group_choice PASSED [  8%]
tests/kiem_thu_danh_gia_cuoi.py::test_case_levels_point_and_quantile PASSED     [ 10%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_daily_target_date_tracks_horizon_offset PASSED [ 12%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_evaluate_validation_summary_matches_by_date_and_id PASSED [ 14%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_join_rows_do_not_duplicate PASSED [ 16%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_n_samples_changes_quantile_output PASSED [ 17%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_pinball_known_answer PASSED [ 19%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_poisson_distribution_scaling PASSED [ 21%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_sample_total_window_does_not_blow_up_for_zero_mean PASSED [ 23%]
tests/kiem_thu_gd6.py::test_k_grid_is_0p5_to_3p0_step_0p1 PASSED             [ 25%]
tests/kiem_thu_gd6.py::test_series_cost_matches_hold_plus_ratio_times_lost PASSED [ 26%]
tests/kiem_thu_gd6.py::test_mase_scale_uses_train_only_and_equals_hand_calculation PASSED [ 28%]
tests/kiem_thu_gd6.py::test_ext_quantiles_below_main_grid_and_monotone PASSED [ 30%]
tests/kiem_thu_gd6.py::test_lgbm_ext_quantiles_not_above_first_main_quantile PASSED [ 32%]
tests/kiem_thu_gd6.py::test_exog_mask_drops_only_price_snap_event_columns PASSED [ 33%]
tests/kiem_thu_gd6.py::test_mlp_seed_changes_weights_but_same_seed_reproduces PASSED [ 35%]
tests/kiem_thu_gd7.py::test_lgbm_q_levels_are_nonnegative_integers_and_round_to_nearest PASSED [ 37%]
tests/kiem_thu_gd7.py::test_nb_models_are_not_changed_by_rounding PASSED     [ 39%]
tests/kiem_thu_gd7.py::test_real_forecasts_levels_used_in_simulation_are_integers PASSED [ 41%]
tests/kiem_thu_gd7.py::test_smaller_kappa_gives_narrower_upper_tail_and_spread PASSED [ 42%]
tests/kiem_thu_gd7.py::test_effect_mask_drops_only_the_two_effect_columns PASSED [ 44%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[0-2] PASSED [ 46%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[0-3] PASSED [ 48%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[1-2] PASSED [ 50%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[1-3] PASSED [ 51%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[2-2] PASSED [ 53%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[2-3] PASSED [ 55%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[3-2] PASSED [ 57%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[3-3] PASSED [ 58%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[4-2] PASSED [ 60%]
tests/kiem_thu_mo_phong_tham_chieu.py::test_vectorized_matches_reference[4-3] PASSED [ 62%]
tests/kiem_thu_mo_phong_ton_kho.py::test_constant_demand_matches_hand_calculation PASSED [ 64%]
tests/kiem_thu_mo_phong_ton_kho.py::test_order_accounting_and_inventory_position_after_ordering PASSED [ 66%]
tests/kiem_thu_mo_phong_ton_kho.py::test_costs_nonnegative_and_fill_rate_range_and_monotone_in_level PASSED [ 67%]
tests/kiem_thu_mo_phong_ton_kho.py::test_reproducible_same_inputs PASSED        [ 69%]
tests/kiem_thu_dau_ra.py::test_val_npz_shapes_cover_all_series_and_origins_and_monotone PASSED [ 71%]
tests/kiem_thu_dau_ra.py::test_val_eval_csv_schema_and_coverage_matches_npz PASSED [ 73%]
tests/kiem_thu_dau_ra.py::test_actual_in_npz_equals_window_sum_of_val_demand PASSED [ 75%]
tests/kiem_thu_dau_ra.py::test_md_table_rounds_to_four_decimals PASSED      [ 76%]
tests/kiem_thu_dau_ra.py::test_test_npz_origins_inside_test_split_and_shapes PASSED [ 78%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_pinball_known_answer PASSED              [ 80%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_nb_quantiles_match_scipy_and_cdf_definition PASSED [ 82%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_nb_quantiles_monotone_in_q_and_zero_mean PASSED [ 83%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_legacy_sampler_mean_not_inflated_when_n_below_one PASSED [ 85%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_window_targets_values_and_nan_tail PASSED [ 87%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_origins_window_inside_split PASSED       [ 89%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_croston_tsb_constant_demand PASSED       [ 91%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_no_leak_stat_forecast_and_alpha PASSED   [ 92%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_no_leak_features PASSED                  [ 94%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_estimate_alpha_known_poisson_is_small PASSED [ 96%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_evaluate_forecast_coverage_and_shapes PASSED [ 98%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_lgbm_quantiles_monotone_and_shape PASSED [100%]

============================= 56 passed in 12.60s =============================
```

## 3. Kết quả
### 3.1 Việc 1 – làm tròn LGBM-Q: TRƯỚC (ceil) / SAU (round)
Bằng chứng ban đầu được xác nhận: ở nhóm slow, L=2, tỷ lệ 10, q=0.5 (val), tồn kho TB và fill rate của 7 mô hình:
| model | inv_TRUOC | inv_SAU | fill_TRUOC | fill_SAU |
|---|---|---|---|---|
| WindowAverage | 0.7843 | 0.7843 | 0.4935 | 0.4935 |
| SeasonalNaive | 1.3145 | 1.3145 | 0.6116 | 0.6116 |
| CrostonSBA | 0.7567 | 0.7370 | 0.4300 | 0.4225 |
| TSB | 0.9207 | 0.9018 | 0.5546 | 0.5481 |
| LGBM-Q | 1.3770 | 0.9000 | 0.6597 | 0.5441 |
| LGBM-PoisNB | 0.8937 | 0.8858 | 0.5319 | 0.5287 |
| MLP-NB | 0.9649 | 0.9649 | 0.5675 | 0.5675 |

Trước: LGBM-Q giữ tồn nhiều nhất và fill cao nhất; sau: LGBM-Q về mức của các mô hình NB (tồn cuối ngày và fill trong cùng khoảng với MLP-NB/LGBM-PoisNB).
LGBM-Q, tổng thể, tỷ lệ 10, theo q (val; cột q0.3… không có vì lưới chính bắt đầu 0.5):
| L | chi_so | phien_ban | q0.5 | q0.6 | q0.7 | q0.8 | q0.85 | q0.9 | q0.95 | q0.975 | q0.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | total_cost | TRUOC (ceil) | 280.5431 | 264.3026 | 255.8554 | 258.6173 | 270.2296 | 290.6403 | 327.0887 | 376.3876 | 443.4836 |
| 2 | total_cost | SAU (round) | 288.6683 | 269.0972 | 256.4447 | 254.4751 | 263.5152 | 282.7290 | 316.0717 | 364.1762 | 429.8955 |
| 2 | fill_rate | TRUOC (ceil) | 0.7863 | 0.8349 | 0.8785 | 0.9188 | 0.9399 | 0.9579 | 0.9759 | 0.9865 | 0.9933 |
| 2 | fill_rate | SAU (round) | 0.7603 | 0.8130 | 0.8614 | 0.9069 | 0.9310 | 0.9510 | 0.9723 | 0.9846 | 0.9924 |
| 2 | avg_inventory | TRUOC (ceil) | 4.3025 | 5.0226 | 5.8878 | 7.0631 | 8.0433 | 9.2526 | 11.0378 | 13.0822 | 15.6608 |
| 2 | avg_inventory | SAU (round) | 3.8979 | 4.6063 | 5.4515 | 6.5988 | 7.5657 | 8.7865 | 10.5462 | 12.5945 | 15.1512 |
| 3 | total_cost | TRUOC (ceil) | 289.2564 | 277.7169 | 274.3378 | 283.1774 | 298.8663 | 324.7789 | 373.8991 | 426.2795 | 504.2564 |
| 3 | total_cost | SAU (round) | 293.3925 | 279.4532 | 273.0911 | 278.0559 | 292.2880 | 316.2187 | 362.4800 | 413.9174 | 491.2928 |
| 3 | fill_rate | TRUOC (ceil) | 0.7975 | 0.8405 | 0.8806 | 0.9205 | 0.9403 | 0.9603 | 0.9767 | 0.9863 | 0.9933 |
| 3 | fill_rate | SAU (round) | 0.7776 | 0.8229 | 0.8663 | 0.9107 | 0.9327 | 0.9547 | 0.9737 | 0.9846 | 0.9924 |
| 3 | avg_inventory | TRUOC (ceil) | 4.9131 | 5.6505 | 6.6030 | 7.9858 | 9.0777 | 10.5374 | 12.7300 | 14.8580 | 17.8308 |
| 3 | avg_inventory | SAU (round) | 4.5284 | 5.2421 | 6.1758 | 7.5408 | 8.6388 | 10.0824 | 12.2414 | 14.3701 | 17.3431 |

q*, fill rate, chi phí và hạng (1 = rẻ nhất trong 7 mô hình theo chi phí tại q*(val) của chính mô hình), tỷ lệ 10:
| L | phien_ban | model | q*_val | fill_val | cost_val | fill_test | cost_test | hang_val(1-7) | hang_test(1-7) | cost_diem_test |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | TRUOC | LGBM-Q | 0.7000 | 0.8785 | 255.8554 | 0.8744 | 261.4253 | 2 | 3 | 283.9842 |
| 2 | TRUOC | MLP-NB | 0.8000 | 0.9062 | 252.4508 | 0.9034 | 258.5030 | 1 | 1 | 275.8068 |
| 2 | TRUOC | LGBM-PoisNB | 0.8000 | 0.9002 | 256.4374 | 0.9024 | 259.9672 | 3 | 2 | 283.9842 |
| 2 | SAU | LGBM-Q | 0.8000 | 0.9069 | 254.4751 | 0.9028 | 258.5018 | 2 | 1 | 283.9842 |
| 2 | SAU | MLP-NB | 0.8000 | 0.9062 | 252.4508 | 0.9034 | 258.5030 | 1 | 2 | 275.8068 |
| 2 | SAU | LGBM-PoisNB | 0.8000 | 0.9014 | 256.5772 | 0.9035 | 260.0826 | 3 | 3 | 283.9842 |
| 3 | TRUOC | LGBM-Q | 0.7000 | 0.8806 | 274.3378 | 0.8776 | 279.8408 | 2 | 3 | 294.0911 |
| 3 | TRUOC | MLP-NB | 0.7000 | 0.8695 | 270.0267 | 0.8633 | 278.5772 | 1 | 1 | 290.9210 |
| 3 | TRUOC | LGBM-PoisNB | 0.7000 | 0.8632 | 274.4982 | 0.8648 | 279.8323 | 3 | 2 | 294.0911 |
| 3 | SAU | LGBM-Q | 0.7000 | 0.8663 | 273.0911 | 0.8642 | 279.3499 | 2 | 2 | 294.0911 |
| 3 | SAU | MLP-NB | 0.7000 | 0.8695 | 270.0267 | 0.8633 | 278.5772 | 1 | 1 | 290.9210 |
| 3 | SAU | LGBM-PoisNB | 0.7000 | 0.8637 | 274.4885 | 0.8653 | 279.7606 | 3 | 3 | 294.0911 |

- q* của LGBM-Q (L=2): 0.7 → 0.8; chi phí test tại q*(val): 261.4253 → 258.5018.
- **Thứ hạng**: ở L=2 trên test LGBM-Q chuyển từ hạng 3 lên hạng 1 nhưng chênh lệch với MLP-NB chỉ -0.0012 đơn vị chi phí (hòa thực tế); trên val LGBM-Q vẫn hạng 2 sau MLP-NB; ở L=3 LGBM-Q lên từ hạng 3 lên hạng 2 (test), MLP-NB vẫn hạng 1 (cả val và test). LGBM-PoisNB tụt 1 hạng ở L=2 (test) vì LGBM-Q vượt lên. Nói cách khác: sau khi làm tròn công bằng, LGBM-Q và MLP-NB gần như ngang nhau về chi phí mô phỏng; kết luận "MLP-NB rẻ nhất ở mọi tỷ lệ ≥5" trước đây cần được hiểu là "MLP-NB và LGBM-Q ngang nhau".
- **Chính sách điểm vẫn dùng ceil**, như yêu cầu. Lý do công bằng: ceil áp dụng giống nhau cho mọi mô hình (kể cả mean Poisson của LightGBM và mean NB của các mô hình khác) nên không làm lệch so sánh GIỮA mô hình. Lý do không công bằng: ceil cộng trung bình +0.5 đơn vị so với làm tròn gần nhất, trong khi phân vị được làm tròn gần nhất/giữ số nguyên, nên điểm THUẦN (k=1) hơi bảo thủ hơn phân vị cùng cấp; điều này ảnh hưởng so sánh phân vị vs điểm thuần (câu 1) nhưng k* (điểm an toàn) hấp thụ phần lớn khác biệt.

### 3.2 Việc 2 – lưới kappa
**Đính chính tiền đề**: lưới kappa đang dùng ở GĐ2–6 là [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0] (đã có 0.1 và 0.2) nên kappa=0.5 KHÔNG chạm cận lưới ở các lần chạy trước; lưới mới theo yêu cầu có cận dưới 0.25 (cao hơn cận cũ). Trong lần chạy mới kappa chọn ở cận mới 0.25: không có mô hình/seed.
kappa và tiêu chí chọn (val), TRƯỚC/SAU:
| model | kappa_TRUOC | kappa_SAU | w_series_TRUOC | w_series_SAU | pinball_avg_val_TRUOC (tieu chi chon) | pinball_avg_val_SAU |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1.1837 | 1.1837 |
| SeasonalNaive | 0.5000 | 0.5000 | 1.0000 | 1.0000 | 1.5132 | 1.5132 |
| CrostonSBA | 0.5000 | 0.6000 | 0.5000 | 0.5000 | 1.3868 | 1.3860 |
| TSB | 0.5000 | 0.6000 | 0.5000 | 0.5000 | 1.1387 | 1.1377 |
| LGBM-PoisNB | 0.7500 | 0.8000 | 1.0000 | 1.0000 | 1.0247 | 1.0246 |
| MLP-NB | 0.5000 | 0.5000 |  |  | 0.9977 | 0.9977 |

kappa của MLP-NB theo seed (chọn chung H=3,4 trên val):
| seed | kappa_TRUOC | kappa_SAU |
|---|---|---|
| 42 | 0.5000 | 0.5000 |
| 43 | 0.5000 | 0.6000 |
| 44 | 0.5000 | 0.6000 |

pinball_avg TRƯỚC/SAU cho các mô hình/giai đoạn có thay đổi (mô hình có kappa đổi: CrostonSBA, LGBM-PoisNB, TSB; không đổi: LGBM-Q, MLP-NB, SeasonalNaive, WindowAverage):
| model | split | H | pinball_avg_TRUOC | pinball_avg_SAU | doi |
|---|---|---|---|---|---|
| CrostonSBA | val | 3 | 1.2539 | 1.2526 | -0.0014 |
| CrostonSBA | val | 4 | 1.5197 | 1.5195 | -0.0002 |
| CrostonSBA | test | 3 | 1.2659 | 1.2631 | -0.0028 |
| CrostonSBA | test | 4 | 1.5402 | 1.5381 | -0.0021 |
| TSB | val | 3 | 1.0398 | 1.0382 | -0.0016 |
| TSB | val | 4 | 1.2377 | 1.2372 | -0.0004 |
| TSB | test | 3 | 1.0799 | 1.0772 | -0.0027 |
| TSB | test | 4 | 1.3023 | 1.2991 | -0.0032 |
| LGBM-PoisNB | val | 3 | 0.9302 | 0.9300 | -0.0002 |
| LGBM-PoisNB | val | 4 | 1.1192 | 1.1192 | 0.0000 |
| LGBM-PoisNB | test | 3 | 0.9487 | 0.9490 | 0.0003 |
| LGBM-PoisNB | test | 4 | 1.1501 | 1.1506 | 0.0005 |

Coverage P(y<q̂) và P(y≤q̂) TRƯỚC/SAU (H=3):
| model | split | H | q | P(y<q)_TRUOC | P(y<q)_SAU | P(y<=q)_TRUOC | P(y<=q)_SAU |
|---|---|---|---|---|---|---|---|
| CrostonSBA | val | 3 | 0.5000 | 0.3920 | 0.3855 | 0.5445 | 0.5384 |
| CrostonSBA | val | 3 | 0.8000 | 0.6921 | 0.6987 | 0.7853 | 0.7905 |
| CrostonSBA | val | 3 | 0.9000 | 0.8131 | 0.8210 | 0.8707 | 0.8774 |
| CrostonSBA | val | 3 | 0.9500 | 0.8795 | 0.8885 | 0.9201 | 0.9262 |
| CrostonSBA | test | 3 | 0.5000 | 0.3713 | 0.3642 | 0.5236 | 0.5165 |
| CrostonSBA | test | 3 | 0.8000 | 0.6864 | 0.6923 | 0.7833 | 0.7882 |
| CrostonSBA | test | 3 | 0.9000 | 0.8119 | 0.8196 | 0.8738 | 0.8804 |
| CrostonSBA | test | 3 | 0.9500 | 0.8805 | 0.8890 | 0.9215 | 0.9278 |
| LGBM-PoisNB | val | 3 | 0.5000 | 0.3937 | 0.3918 | 0.5989 | 0.5978 |
| LGBM-PoisNB | val | 3 | 0.8000 | 0.7259 | 0.7280 | 0.8372 | 0.8385 |
| LGBM-PoisNB | val | 3 | 0.9000 | 0.8614 | 0.8641 | 0.9124 | 0.9140 |
| LGBM-PoisNB | val | 3 | 0.9500 | 0.9211 | 0.9227 | 0.9504 | 0.9515 |
| LGBM-PoisNB | test | 3 | 0.5000 | 0.4100 | 0.4076 | 0.5893 | 0.5872 |
| LGBM-PoisNB | test | 3 | 0.8000 | 0.7490 | 0.7511 | 0.8386 | 0.8401 |
| LGBM-PoisNB | test | 3 | 0.9000 | 0.8662 | 0.8685 | 0.9156 | 0.9176 |
| LGBM-PoisNB | test | 3 | 0.9500 | 0.9240 | 0.9257 | 0.9517 | 0.9533 |
| TSB | val | 3 | 0.5000 | 0.3782 | 0.3732 | 0.5925 | 0.5880 |
| TSB | val | 3 | 0.8000 | 0.6586 | 0.6644 | 0.8120 | 0.8165 |
| TSB | val | 3 | 0.9000 | 0.7727 | 0.7797 | 0.8905 | 0.8954 |
| TSB | val | 3 | 0.9500 | 0.8396 | 0.8465 | 0.9321 | 0.9362 |
| TSB | test | 3 | 0.5000 | 0.3938 | 0.3881 | 0.5801 | 0.5751 |
| TSB | test | 3 | 0.8000 | 0.6987 | 0.7040 | 0.8152 | 0.8189 |
| TSB | test | 3 | 0.9000 | 0.8154 | 0.8233 | 0.8963 | 0.9017 |
| TSB | test | 3 | 0.9500 | 0.8840 | 0.8903 | 0.9369 | 0.9415 |

Tiêu chí hiệu chuẩn rời rạc (số q trên 7 thỏa khoảng nới ±0.05 và |P(y≤q̂)−q|≤0.05) chỉ khác ở 0 dòng (split,H,model):
| split | H | model | in_bracket_tol_TRUOC | in_bracket_tol_SAU | abs_dev_ok_TRUOC | abs_dev_ok_SAU |
|---|---|---|---|---|---|---|


MLP-NB 3 seed TRƯỚC/SAU (pinball_avg, WAPE, MASE; trung bình ± độ lệch chuẩn):
| phien_ban | split | H | pinball_avg | wape | mase | epochs | kappa |
|---|---|---|---|---|---|---|---|
| SAU | test | 3 | 0.9460 ± 0.0024 | 0.3515 ± 0.0004 | 0.7843 ± 0.0005 | 17/25/20 | 0.5/0.6/0.6 |
| TRUOC | test | 3 | 0.9477 ± 0.0021 | 0.3515 ± 0.0004 | 0.7843 ± 0.0005 | 17/25/20 | 0.5/0.5/0.5 |
| SAU | test | 4 | 1.1433 ± 0.0058 | 0.3170 ± 0.0006 | 0.8031 ± 0.0032 | 30/16/26 | 0.5/0.6/0.6 |
| TRUOC | test | 4 | 1.1449 ± 0.0045 | 0.3170 ± 0.0006 | 0.8031 ± 0.0032 | 30/16/26 | 0.5/0.5/0.5 |
| SAU | val | 3 | 0.9023 ± 0.0008 | 0.3680 ± 0.0029 | 0.7559 ± 0.0043 | 17/25/20 | 0.5/0.6/0.6 |
| TRUOC | val | 3 | 0.9026 ± 0.0005 | 0.3680 ± 0.0029 | 0.7559 ± 0.0043 | 17/25/20 | 0.5/0.5/0.5 |
| SAU | val | 4 | 1.0897 ± 0.0039 | 0.3315 ± 0.0022 | 0.7748 ± 0.0043 | 30/16/26 | 0.5/0.6/0.6 |
| TRUOC | val | 4 | 1.0899 ± 0.0038 | 0.3315 ± 0.0022 | 0.7748 ± 0.0043 | 30/16/26 | 0.5/0.5/0.5 |

- sieu_tham_so.json: sha256 TRƯỚC `c26d657819002fb031ee7405d7c9bdc07685428e67a13cb607b082863f72e208`, SAU `8bc7d7507bf0167e83b1d31f831fb8edb5edf60c23a742c67b32b0f86366765b` (đổi vì kappa của CrostonSBA/TSB và LGBM-PoisNB đổi; lý do: lưới kappa mới). Test chỉ chạy một lần với siêu tham số đã chốt trên val (không chọn kappa theo test).
- Hệ quả: thay đổi nhỏ (pinball_avg thay đổi ở chữ số thập phân thứ 3–4), không đổi thứ hạng mô hình; kappa của WindowAverage, SeasonalNaive và MLP-NB (seed 42, mô hình chính) giữ 0.5; MLP-NB seed 43 và 44 đổi 0.5 → 0.6; CrostonSBA/TSB đổi 0.5 → 0.6 và LGBM-PoisNB 0.75 → 0.8 (các giá trị mới nằm trong lưới cũ hoặc là lân cận của giá trị cũ, nên không có dấu hiệu tối ưu nằm sát cận dưới).
- Test đơn điệu kappa (`test_smaller_kappa_gives_narrower_upper_tail_and_spread`): **không thể kiểm chứng đúng nghĩa đen "phân vị hẹp hơn hoặc bằng ở MỌI q"**: với trung bình cố định, alpha nhỏ hơn làm phân vị thấp/giữa tăng (ví dụ mu=30, alpha=1: Q0.5 từ 14 (kappa 2.0) lên 27 (kappa 0.25)), và ở alpha lớn thậm chí q=0.9 không đơn điệu. Test được viết cho tính chất đúng: phân vị đuôi trên (q ≥ 0.95) không giảm theo kappa và khoảng Q95−Q50 không giảm theo kappa.

### 3.3 Việc 3a – sửa câu sai về tỷ lệ 1–2
Câu cũ "Ở tỷ lệ 1–2 điểm an toàn thường rẻ hơn hoặc ngang" chỉ đúng cho tỷ lệ 1. Tính lại theo số mới (test, k* − q*, KTC 95%):
| L | cost_ratio | diem_an_toan_re_hon (KTC<0) | ngang (KTC chua 0) | phan_vi_re_hon (KTC>0) |
|---|---|---|---|---|
| 2 | 1 | WindowAverage, SeasonalNaive, CrostonSBA, TSB, LGBM-Q, LGBM-PoisNB, MLP-NB | - | - |
| 2 | 2 | SeasonalNaive | - | WindowAverage, CrostonSBA, TSB, LGBM-Q, LGBM-PoisNB, MLP-NB |
| 3 | 1 | WindowAverage, SeasonalNaive, CrostonSBA, TSB, LGBM-Q, LGBM-PoisNB, MLP-NB | - | - |
| 3 | 2 | WindowAverage, SeasonalNaive, TSB | CrostonSBA, LGBM-Q, LGBM-PoisNB, MLP-NB | - |

Tỷ lệ 2, L=2: ví dụ WindowAverage 2.46 [0.91; 4.08] (số trước GĐ7: 2.46 [0.91; 4.08]), MLP-NB 3.08 [1.13; 5.14] — hai ví dụ này KHÔNG đổi sau GĐ7 vì WindowAverage và MLP-NB (seed 42) không bị ảnh hưởng bởi làm tròn LGBM-Q và kappa của chúng giữ 0.5; số của CrostonSBA, TSB, LGBM-Q, LGBM-PoisNB thì đã được tính lại (xem bảng trên). Các câu tương ứng trong `bao_cao_cuoi_cung.md` được viết lại bằng số mới (xem mục 6 báo cáo cuối); `06_bo_sung.md` mục 3 cũng được dựng lại.

### 3.4 Việc 3b – bỏ riêng fut_wd_index và fut_event_index (mã hóa mục tiêu trong mẫu)
Huấn luyện LightGBM-Q bỏ riêng hai cột (30 → 28 đặc trưng), cùng cấu hình:
| split | H | model | mase | pinball_avg | wape |
|---|---|---|---|---|---|
| test | 3 | LGBM-Q[full] | 0.7870 | 0.9354 | 0.3508 |
| test | 3 | LGBM-Q[no_effect_cols] | 0.7876 | 0.9380 | 0.3510 |
| test | 4 | LGBM-Q[full] | 0.8051 | 1.1344 | 0.3166 |
| test | 4 | LGBM-Q[no_effect_cols] | 0.8054 | 1.1325 | 0.3178 |
| val | 3 | LGBM-Q[full] | 0.7672 | 0.9109 | 0.3697 |
| val | 3 | LGBM-Q[no_effect_cols] | 0.7667 | 0.9115 | 0.3695 |
| val | 4 | LGBM-Q[full] | 0.7884 | 1.0922 | 0.3333 |
| val | 4 | LGBM-Q[no_effect_cols] | 0.7885 | 1.0927 | 0.3333 |

Chênh lệch tương đối pinball_avg: -0.17% … +0.27% (|chênh| lớn nhất 0.27%); tác động không đáng kể (|chênh| ≤ 1%, dấu không nhất quán giữa các giai đoạn/H):
| split | H | pinball_full | pinball_no_effect_cols | chenh | tuong_doi |
|---|---|---|---|---|---|
| test | 3 | 0.9354 | 0.9380 | 0.0025 | 0.0027 |
| test | 4 | 1.1344 | 1.1325 | -0.0020 | -0.0017 |
| val | 3 | 0.9109 | 0.9115 | 0.0007 | 0.0007 |
| val | 4 | 1.0922 | 1.0927 | 0.0005 | 0.0005 |

Kết luận: nguy cơ "tin quá mức vào đặc trưng sự kiện hiếm" không thấy thể hiện trong sai số ngoài mẫu (val/test). Hạn chế đã ghi vào `bao_cao_cuoi_cung.md`.

### 3.5 Việc 3c – câu 2: phân vị q* vs điểm an toàn k* theo nhóm tốc độ (test, tỷ lệ 10, L=2; k*, q* chọn trên val RIÊNG trong từng nhóm)
| model | group | n | k*_val | q*_val | chi_phi_diem_thuan(k=1) | chi_phi_diem_an_toan(k*) | chi_phi_phan_vi(q*) | k* - q* [KTC] | q* - diem_thuan [KTC] | k* - diem_thuan [KTC] |
|---|---|---|---|---|---|---|---|---|---|---|
| CrostonSBA | fast | 326 | 1.30 | 0.80 | 614.68 | 548.46 | 547.75 | 0.72 [-3.27; 4.61] | -66.94 [-83.88; -51.31] | -66.22 [-81.04; -52.97] |
| LGBM-PoisNB | fast | 326 | 1.20 | 0.80 | 504.73 | 454.85 | 452.15 | 2.71 [-2.69; 8.61] | -52.58 [-66.02; -40.06] | -49.87 [-59.86; -40.16] |
| LGBM-Q | fast | 326 | 1.20 | 0.80 | 504.73 | 454.85 | 448.17 | 6.68 [2.29; 11.78] | -56.55 [-69.80; -44.50] | -49.87 [-59.86; -40.16] |
| MLP-NB | fast | 326 | 1.20 | 0.80 | 486.50 | 455.58 | 450.33 | 5.26 [2.04; 8.70] | -36.18 [-46.12; -26.61] | -30.92 [-39.25; -22.91] |
| SeasonalNaive | fast | 326 | 1.10 | 0.70 | 562.58 | 530.76 | 529.46 | 1.29 [-1.45; 4.16] | -33.11 [-41.33; -25.78] | -31.82 [-38.02; -26.10] |
| TSB | fast | 326 | 1.30 | 0.80 | 547.38 | 495.33 | 492.60 | 2.73 [-0.33; 6.08] | -54.78 [-68.37; -41.82] | -52.05 [-65.02; -39.63] |
| WindowAverage | fast | 326 | 1.30 | 0.80 | 558.14 | 505.73 | 503.41 | 2.32 [-0.37; 5.12] | -54.73 [-67.96; -41.79] | -52.41 [-65.08; -39.67] |
| CrostonSBA | medium | 321 | 1.40 | 0.80 | 181.26 | 169.52 | 169.54 | -0.02 [-0.76; 0.73] | -11.72 [-15.03; -8.45] | -11.75 [-14.88; -8.48] |
| LGBM-PoisNB | medium | 321 | 1.30 | 0.80 | 164.59 | 155.20 | 155.87 | -0.67 [-1.52; 0.15] | -8.72 [-12.04; -5.50] | -9.39 [-12.31; -6.65] |
| LGBM-Q | medium | 321 | 1.30 | 0.80 | 164.59 | 155.20 | 155.45 | -0.25 [-1.53; 1.05] | -9.13 [-12.59; -5.72] | -9.39 [-12.31; -6.65] |
| MLP-NB | medium | 321 | 1.30 | 0.80 | 161.96 | 154.75 | 154.86 | -0.11 [-0.96; 0.72] | -7.10 [-10.04; -4.18] | -7.21 [-10.03; -4.52] |
| SeasonalNaive | medium | 321 | 1.10 | 0.70 | 190.73 | 182.73 | 188.48 | -5.75 [-6.84; -4.69] | -2.24 [-4.23; -0.14] | -7.99 [-9.60; -6.31] |
| TSB | medium | 321 | 1.30 | 0.80 | 165.59 | 159.65 | 160.29 | -0.64 [-1.72; 0.45] | -5.30 [-8.23; -2.46] | -5.94 [-8.29; -3.60] |
| WindowAverage | medium | 321 | 1.30 | 0.80 | 168.10 | 161.17 | 162.04 | -0.88 [-1.74; 0.01] | -6.06 [-8.77; -3.27] | -6.93 [-9.41; -4.36] |
| CrostonSBA | slow | 176 | 1.30 | 0.80 | 98.70 | 97.31 | 98.94 | -1.63 [-2.86; -0.48] | 0.23 [-1.65; 2.29] | -1.40 [-2.95; 0.23] |
| LGBM-PoisNB | slow | 176 | 1.40 | 0.80 | 92.87 | 95.35 | 94.39 | 0.95 [0.05; 1.93] | 1.52 [-0.78; 3.63] | 2.48 [0.18; 4.68] |
| LGBM-Q | slow | 176 | 1.40 | 0.80 | 92.87 | 95.35 | 95.12 | 0.23 [-1.41; 1.78] | 2.25 [-0.30; 4.65] | 2.48 [0.18; 4.68] |
| MLP-NB | slow | 176 | 1.30 | 0.80 | 93.18 | 93.05 | 92.23 | 0.82 [-0.07; 1.77] | -0.95 [-3.19; 1.10] | -0.13 [-2.11; 1.79] |
| SeasonalNaive | slow | 176 | 0.70 | 0.50 | 110.20 | 106.49 | 108.94 | -2.45 [-3.66; -1.32] | -1.26 [-2.76; 0.36] | -3.71 [-5.44; -1.97] |
| TSB | slow | 176 | 1.30 | 0.70 | 91.84 | 94.19 | 92.66 | 1.52 [-0.16; 3.24] | 0.83 [-0.47; 2.08] | 2.35 [0.62; 4.18] |
| WindowAverage | slow | 176 | 1.30 | 0.80 | 94.32 | 94.79 | 96.70 | -1.91 [-3.01; -0.89] | 2.38 [0.45; 4.50] | 0.47 [-1.24; 2.31] |

Chú giải: k* − q* dương = phân vị rẻ hơn; q* − điểm thuần âm = phân vị rẻ hơn điểm thuần.
- Hàng nhanh: q* − điểm thuần -66.94 … -33.11 (đều rẻ hơn rõ: 7/7 mô hình), nhưng k* − q* chỉ 0.72 … 6.68: phân vị rẻ hơn điểm an toàn rõ ở LGBM-Q, MLP-NB (còn lại không phân biệt được). **Lợi thế của phân vị ở hàng nhanh gần như biến mất khi so với điểm an toàn**: chỉ còn khoảng 1–1.5% chi phí và chỉ ở hai mô hình (LGBM-Q, MLP-NB).
- Hàng trung bình: không phân biệt được hoặc điểm an toàn rẻ hơn (rẻ hơn rõ: SeasonalNaive).
- Hàng chậm: điểm an toàn rẻ hơn hoặc ngang phân vị (rẻ hơn rõ: WindowAverage, SeasonalNaive, CrostonSBA; phân vị rẻ hơn rõ: LGBM-PoisNB).
- **L=3** (bảng dưới; `results/diem_an_toan_theo_toc_do_test.csv`) — kết quả KHÔNG hoàn toàn giống L=2: hàng nhanh: phân vị rẻ hơn điểm an toàn rõ ở MLP-NB, điểm an toàn rẻ hơn rõ ở WindowAverage, SeasonalNaive, CrostonSBA; hàng trung bình: phân vị rẻ hơn rõ ở LGBM-Q, LGBM-PoisNB, điểm an toàn rẻ hơn rõ ở WindowAverage; hàng chậm: phân vị rẻ hơn rõ ở không mô hình nào, điểm an toàn rẻ hơn rõ ở WindowAverage, SeasonalNaive, CrostonSBA. Nghĩa là kết luận "lợi thế phân vị ở hàng nhanh biến mất" chắc ở L=2 (chỉ LGBM-Q và MLP-NB còn lợi thế nhỏ) và mang tính hỗn hợp ở L=3 (MLP-NB còn lợi thế ở hàng nhanh, nhưng ở CrostonSBA, SeasonalNaive, WindowAverage điểm an toàn rẻ hơn).
L=3:
| model | group | n | k*_val | q*_val | chi_phi_diem_thuan(k=1) | chi_phi_diem_an_toan(k*) | chi_phi_phan_vi(q*) | k* - q* [KTC] | q* - diem_thuan [KTC] | k* - diem_thuan [KTC] |
|---|---|---|---|---|---|---|---|---|---|---|
| CrostonSBA | fast | 326 | 1.20 | 0.70 | 641.03 | 597.97 | 604.20 | -6.23 [-8.49; -3.96] | -36.84 [-47.25; -27.48] | -43.07 [-54.00; -32.95] |
| LGBM-PoisNB | fast | 326 | 1.10 | 0.70 | 521.51 | 493.07 | 490.36 | 2.71 [-2.13; 6.50] | -31.15 [-39.79; -22.44] | -28.44 [-35.55; -21.92] |
| LGBM-Q | fast | 326 | 1.10 | 0.70 | 521.51 | 493.07 | 490.72 | 2.35 [-1.38; 5.91] | -30.79 [-38.41; -23.71] | -28.44 [-35.55; -21.92] |
| MLP-NB | fast | 326 | 1.10 | 0.70 | 515.63 | 493.62 | 488.98 | 4.65 [1.95; 7.54] | -26.66 [-33.23; -20.31] | -22.01 [-27.67; -16.56] |
| SeasonalNaive | fast | 326 | 1.10 | 0.60 | 568.35 | 549.93 | 558.85 | -8.92 [-12.41; -5.83] | -9.50 [-13.12; -5.75] | -18.42 [-24.22; -13.24] |
| TSB | fast | 326 | 1.20 | 0.70 | 567.18 | 536.58 | 539.96 | -3.38 [-7.55; 0.43] | -27.21 [-34.83; -19.91] | -30.60 [-40.33; -20.92] |
| WindowAverage | fast | 326 | 1.20 | 0.70 | 575.85 | 544.15 | 551.21 | -7.06 [-12.75; -2.21] | -24.64 [-32.72; -16.36] | -31.70 [-41.66; -21.53] |
| CrostonSBA | medium | 321 | 1.20 | 0.80 | 187.42 | 180.93 | 182.64 | -1.71 [-3.60; 0.03] | -4.79 [-8.24; -1.25] | -6.50 [-8.62; -4.36] |
| LGBM-PoisNB | medium | 321 | 1.30 | 0.70 | 170.98 | 167.58 | 165.93 | 1.65 [0.03; 3.24] | -5.05 [-6.74; -3.28] | -3.40 [-6.43; -0.39] |
| LGBM-Q | medium | 321 | 1.30 | 0.70 | 170.98 | 167.58 | 164.48 | 3.10 [1.46; 4.72] | -6.50 [-8.74; -4.31] | -3.40 [-6.43; -0.39] |
| MLP-NB | medium | 321 | 1.20 | 0.70 | 168.96 | 164.69 | 164.76 | -0.07 [-1.34; 1.15] | -4.20 [-5.97; -2.50] | -4.27 [-6.62; -1.87] |
| SeasonalNaive | medium | 321 | 0.90 | 0.50 | 191.01 | 190.53 | 191.65 | -1.12 [-2.45; 0.19] | 0.64 [-0.90; 2.18] | -0.48 [-1.26; 0.30] |
| TSB | medium | 321 | 1.20 | 0.70 | 171.67 | 170.27 | 169.33 | 0.94 [-0.27; 2.02] | -2.35 [-4.13; -0.68] | -1.40 [-3.49; 0.56] |
| WindowAverage | medium | 321 | 1.20 | 0.70 | 175.30 | 170.71 | 171.98 | -1.27 [-2.35; -0.15] | -3.31 [-5.01; -1.59] | -4.58 [-6.67; -2.52] |
| CrostonSBA | slow | 176 | 1.10 | 0.70 | 101.84 | 101.17 | 103.86 | -2.69 [-3.85; -1.57] | 2.03 [0.62; 3.44] | -0.66 [-1.80; 0.44] |
| LGBM-PoisNB | slow | 176 | 1.10 | 0.70 | 97.38 | 97.38 | 97.29 | 0.09 [-0.73; 0.97] | -0.09 [-1.38; 1.22] | -0.01 [-1.25; 1.24] |
| LGBM-Q | slow | 176 | 1.10 | 0.70 | 97.38 | 97.38 | 97.34 | 0.03 [-1.66; 1.70] | -0.04 [-1.81; 1.72] | -0.01 [-1.25; 1.24] |
| MLP-NB | slow | 176 | 1.10 | 0.70 | 97.12 | 96.36 | 96.44 | -0.09 [-1.03; 0.89] | -0.68 [-2.05; 0.57] | -0.77 [-1.93; 0.36] |
| SeasonalNaive | slow | 176 | 0.80 | 0.50 | 111.17 | 107.86 | 109.22 | -1.36 [-2.61; -0.13] | -1.95 [-3.29; -0.59] | -3.31 [-4.58; -2.04] |
| TSB | slow | 176 | 1.00 | 0.60 | 96.89 | 96.89 | 97.33 | -0.44 [-1.90; 1.00] | 0.44 [-1.00; 1.90] | 0.00 [0.00; 0.00] |
| WindowAverage | slow | 176 | 1.10 | 0.70 | 98.07 | 98.59 | 100.79 | -2.20 [-3.22; -1.19] | 2.72 [1.26; 4.26] | 0.52 [-0.88; 1.94] |


## 4. File sinh ra
`results/mo_phong_ton_kho_{val,test}[_ext].csv`, `q_sao_val.csv`, `q_sao_ext_val.csv`, `bang_chinh_sach_val.csv`, `danh_gia_chinh_sach_val_test.csv`, `danh_gia_chinh_sach_bootstrap.csv`, `chuyen_q_sao_test.csv`, `cau_hoi_nghien_cuu_{val,test}.csv`, `diem_an_toan_test.csv`, `diem_an_toan_theo_toc_do_test.csv`, `danh_gia_du_bao_{val,test}_all.csv`, `sieu_tham_so.json`, `nhat_ky_tinh_chinh.csv`, `calibration_*.csv`, `mlp_cac_seed.csv`, `mlp_duong_nll.csv`, `loai_bo_hieu_ung_cot.csv`, `forecasts/{val,test}.npz`, hình `results/figures/chinh_sach_*.png`. Bản trước GĐ7: `archive/pre_gd7/`.

## 5. Cấu hình
Seed 42. Lưới kappa {0.25, 0.35, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0}; lưới q chính 0.5–0.99, phụ 0.3/0.4; k 0.5–3.0 bước 0.1; bootstrap 2000 lần seed 42. lightgbm 4.7.0, torch 2.15.0+cpu, scipy 1.18.1, numpy 2.5.4, pandas 2.3.3. CPU 12 luồng; các tác vụ chạy song song nên thời gian từng bước dài hơn GĐ6 (run_forecasts val ≈ 654 s, test ≈ 310 s; mlp_seeds seed đầu 736 s).

## 6. Vấn đề / bất thường / giả định
- Tiền đề "kappa chạm cận 0.5" không đúng với lưới cũ (xem 3.2); lưới mới theo yêu cầu thực ra thu hẹp cận dưới (0.1 → 0.25) mà kết quả không đổi nhiều.
- **Đính chính GĐ6**: lời tóm tắt "cả 3 seed MLP-NB đều hơn LGBM-Q trên val" chỉ đúng ở H=3; ở H=4 trên val seed 42 (1.0929) kém hơn LGBM-Q (1.0922), tức hai mô hình gần như ngang trên val ở H=4 (kết quả lẫn lộn giữa các seed). Đã sửa trong `06_bo_sung.md` và `bao_cao_cuoi_cung.md` (val H=3: MLP-NB 0.9014–0.9029 vs LGBM-Q 0.9109 → cả 3 seed tốt hơn LGBM-Q; val H=4: MLP-NB 1.0853–1.0929 vs LGBM-Q 1.0922 → lẫn lộn (một số seed tốt hơn, một số kém hơn); test H=3: MLP-NB 0.9432–0.9477 vs LGBM-Q 0.9354 → cả 3 seed kém hơn LGBM-Q; test H=4: MLP-NB 1.1374–1.1489 vs LGBM-Q 1.1344 → cả 3 seed kém hơn LGBM-Q).
- Sau làm tròn, LGBM-Q và MLP-NB hòa về chi phí ở L=2: kết luận "mô hình tốt nhất" không còn một mô hình duy nhất.
- Điểm thuần dùng ceil (thiên bảo thủ) – xem 3.1; k* hấp thụ.
- k* và q* cùng chọn trên val (lạc quan), chọn theo nhóm tốc độ càng dễ nhiễu (nhóm slow n=176).
- Chỉ 28 ngày; KTC coi chuỗi độc lập; tỷ lệ 1–2 chạm cận lưới k (0.5) và q (0.3).
- Các báo cáo 02, 03, 04 là ảnh chụp trước GĐ7 (số liệu của chúng khác nhẹ ở CrostonSBA/TSB/LGBM-PoisNB và các mục mô phỏng của LGBM-Q); bản dùng thống nhất là 05, 06, 07 và `bao_cao_cuoi_cung.md`.

## 7. Chưa làm
Chưa chạy lại MLP-NB H=4 với 46 epoch; chưa mở lưới q dưới 0.3 và k dưới 0.5; chưa đối chứng Croston/TSB + biến ngoại sinh; chưa cập nhật số liệu trong báo cáo 02–04 (chỉ ghi chú).
