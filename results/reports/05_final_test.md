# Báo cáo GĐ5 – Đánh giá cuối trên TEST

## 1. Đã làm gì
- Chạy `python -m src.run_forecasts --split test`: huấn luyện lại LightGBM/MLP-NB trên train+val (nhãn đến hết val; MLP-NB dùng đúng số epoch đã chốt: H=3 → 17, H=4 → 30), dùng NGUYÊN siêu tham số của val (`results/hparams.json`, sha256 16 ký tự đầu = c26d657819002fb0, trước và sau khi chạy test không đổi). Chỉ số hiệu ứng thứ/sự kiện của đặc trưng chỉ tính đến hết val.
- `python -m src.inventory_sim --split test`, `python -m src.final_eval` (áp dụng q* và chính sách theo nhóm chốt từ val lên test), bootstrap câu hỏi nghiên cứu trên test (`research_questions_test.csv`).
- File mới: `src/final_eval.py`, `tests/test_final_eval.py` (2 test), thêm test `test_test_npz_...` trong `tests/test_outputs.py`; `notebooks/04_final_results.ipynb` (Restart & Run All, 0 lỗi); viết lại `notebooks/03_baseline_stats.ipynb` (bản cũ gọi sai hàm); `README.md`, `requirements.txt` (thêm torch), `results/reports/final_report.md`.
- Test chỉ được chạm một lần theo nghĩa: không có bước chọn/chỉnh nào dùng kết quả test; kết quả test chỉ được đọc để báo cáo.

## 2. Lệnh đã chạy và pytest
```
python -m src.run_forecasts --split test
python -m src.inventory_sim --split test
python -m src.final_eval
python -m pytest -v
```
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\lt\261\PTDLHS\DA\.venv\Scripts\python.exe
cachedir: .pytest_cache
Fugue tests will be initialized with options:
rootdir: D:\lt\261\PTDLHS\DA
plugins: anyio-4.15.1, fugue-0.9.7, platformdirs-4.14.0
collecting ... collected 41 items

tests/test_eda.py::test_load_train_excludes_val PASSED                   [  2%]
tests/test_eda.py::test_active_rows_drops_pre_launch_days PASSED         [  4%]
tests/test_eda.py::test_weekday_table_has_seven_rows_and_mean_two PASSED [  7%]
tests/test_eda.py::test_demand_speed_table_totals PASSED                 [  9%]
tests/test_final_eval.py::test_group_policy_levels_follow_group_choice PASSED [ 12%]
tests/test_final_eval.py::test_case_levels_point_and_quantile PASSED     [ 14%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_daily_target_date_tracks_horizon_offset PASSED [ 17%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_evaluate_validation_summary_matches_by_date_and_id PASSED [ 19%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_join_rows_do_not_duplicate PASSED [ 21%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_n_samples_changes_quantile_output PASSED [ 24%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_pinball_known_answer PASSED [ 26%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_poisson_distribution_scaling PASSED [ 29%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_sample_total_window_does_not_blow_up_for_zero_mean PASSED [ 31%]
tests/test_gd6.py::test_k_grid_is_0p5_to_3p0_step_0p1 PASSED             [ 34%]
tests/test_gd6.py::test_series_cost_matches_hold_plus_ratio_times_lost PASSED [ 36%]
tests/test_gd6.py::test_mase_scale_uses_train_only_and_equals_hand_calculation PASSED [ 39%]
tests/test_gd6.py::test_ext_quantiles_below_main_grid_and_monotone PASSED [ 41%]
tests/test_gd6.py::test_lgbm_ext_quantiles_not_above_first_main_quantile PASSED [ 43%]
tests/test_gd6.py::test_exog_mask_drops_only_price_snap_event_columns PASSED [ 46%]
tests/test_gd6.py::test_mlp_seed_changes_weights_but_same_seed_reproduces PASSED [ 48%]
tests/test_inventory_sim.py::test_constant_demand_matches_hand_calculation PASSED [ 51%]
tests/test_inventory_sim.py::test_inventory_conservation_every_day PASSED [ 53%]
tests/test_inventory_sim.py::test_costs_nonnegative_and_fill_rate_range_and_monotone_in_level PASSED [ 56%]
tests/test_inventory_sim.py::test_reproducible_same_inputs PASSED        [ 58%]
tests/test_outputs.py::test_val_npz_shapes_cover_all_series_and_origins_and_monotone PASSED [ 60%]
tests/test_outputs.py::test_val_eval_csv_schema_and_coverage_matches_npz PASSED [ 63%]
tests/test_outputs.py::test_actual_in_npz_equals_window_sum_of_val_demand PASSED [ 65%]
tests/test_outputs.py::test_md_table_rounds_to_four_decimals PASSED      [ 68%]
tests/test_outputs.py::test_test_npz_origins_inside_test_split_and_shapes PASSED [ 70%]
tests/test_prob_models.py::test_pinball_known_answer PASSED              [ 73%]
tests/test_prob_models.py::test_nb_quantiles_match_scipy_and_cdf_definition PASSED [ 75%]
tests/test_prob_models.py::test_nb_quantiles_monotone_in_q_and_zero_mean PASSED [ 78%]
tests/test_prob_models.py::test_legacy_sampler_mean_not_inflated_when_n_below_one PASSED [ 80%]
tests/test_prob_models.py::test_window_targets_values_and_nan_tail PASSED [ 82%]
tests/test_prob_models.py::test_origins_window_inside_split PASSED       [ 85%]
tests/test_prob_models.py::test_croston_tsb_constant_demand PASSED       [ 87%]
tests/test_prob_models.py::test_no_leak_stat_forecast_and_alpha PASSED   [ 90%]
tests/test_prob_models.py::test_no_leak_features PASSED                  [ 92%]
tests/test_prob_models.py::test_estimate_alpha_known_poisson_is_small PASSED [ 95%]
tests/test_prob_models.py::test_evaluate_forecast_coverage_and_shapes PASSED [ 97%]
tests/test_prob_models.py::test_lgbm_quantiles_monotone_and_shape PASSED [100%]

============================= 41 passed in 7.71s ==============================
```

## 3. Kết quả
### 3.1 Dự báo, H=3, tổng thể (val vs test)
| model | pinball_avg_val | pinball_avg_test | wape_val | wape_test | mase_val | mase_test |
|---|---|---|---|---|---|---|
| WindowAverage | 1.0757 | 1.1316 | 0.4129 | 0.3987 | 0.8125 | 0.8474 |
| SeasonalNaive | 1.4002 | 1.4112 | 0.4741 | 0.4399 | 0.9645 | 0.9882 |
| CrostonSBA | 1.2539 | 1.2659 | 0.4948 | 0.4484 | 0.9502 | 0.9174 |
| TSB | 1.0398 | 1.0799 | 0.4058 | 0.3898 | 0.8009 | 0.8287 |
| LGBM-Q | 0.9109 | 0.9354 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| LGBM-PoisNB | 0.9302 | 0.9487 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| MLP-NB | 0.9025 | 0.9477 | 0.3701 | 0.3512 | 0.7597 | 0.7839 |

H=4:
| model | pinball_avg_val | pinball_avg_test | wape_val | wape_test | mase_val | mase_test |
|---|---|---|---|---|---|---|
| WindowAverage | 1.2918 | 1.3782 | 0.3682 | 0.3575 | 0.8320 | 0.8677 |
| SeasonalNaive | 1.6261 | 1.6239 | 0.4214 | 0.3859 | 0.9792 | 0.9882 |
| CrostonSBA | 1.5197 | 1.5402 | 0.4506 | 0.4060 | 0.9886 | 0.9445 |
| TSB | 1.2377 | 1.3023 | 0.3605 | 0.3474 | 0.8173 | 0.8444 |
| LGBM-Q | 1.0922 | 1.1344 | 0.3333 | 0.3166 | 0.7884 | 0.8051 |
| LGBM-PoisNB | 1.1192 | 1.1501 | 0.3333 | 0.3166 | 0.7884 | 0.8051 |
| MLP-NB | 1.0929 | 1.1489 | 0.3328 | 0.3165 | 0.7746 | 0.8024 |

Coverage P(y≤Q) H=3:
| model | cov_q0.5_val | cov_q0.5_test | cov_q0.9_val | cov_q0.9_test | cov_q0.99_val | cov_q0.99_test |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5850 | 0.5694 | 0.8897 | 0.8906 | 0.9678 | 0.9688 |
| SeasonalNaive | 0.5765 | 0.5518 | 0.8043 | 0.7974 | 0.8844 | 0.8805 |
| CrostonSBA | 0.5445 | 0.5236 | 0.8707 | 0.8738 | 0.9671 | 0.9665 |
| TSB | 0.5925 | 0.5801 | 0.8905 | 0.8963 | 0.9712 | 0.9741 |
| LGBM-Q | 0.5412 | 0.5127 | 0.9063 | 0.9027 | 0.9881 | 0.9874 |
| LGBM-PoisNB | 0.5989 | 0.5893 | 0.9124 | 0.9156 | 0.9820 | 0.9842 |
| MLP-NB | 0.6198 | 0.6004 | 0.9135 | 0.9105 | 0.9874 | 0.9834 |

Coverage test mọi q (H=3):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.5694 | 0.6488 | 0.7303 | 0.8092 | 0.8499 | 0.8906 | 0.9318 | 0.9539 | 0.9688 |
| SeasonalNaive | 0.5518 | 0.5944 | 0.6655 | 0.7356 | 0.7677 | 0.7974 | 0.8354 | 0.8634 | 0.8805 |
| CrostonSBA | 0.5236 | 0.6080 | 0.6943 | 0.7833 | 0.8278 | 0.8738 | 0.9215 | 0.9475 | 0.9665 |
| TSB | 0.5801 | 0.6629 | 0.7398 | 0.8152 | 0.8547 | 0.8963 | 0.9369 | 0.9583 | 0.9741 |
| LGBM-Q | 0.5127 | 0.6069 | 0.7062 | 0.8064 | 0.8553 | 0.9027 | 0.9476 | 0.9720 | 0.9874 |
| LGBM-PoisNB | 0.5893 | 0.6711 | 0.7546 | 0.8386 | 0.8774 | 0.9156 | 0.9517 | 0.9711 | 0.9842 |
| MLP-NB | 0.6004 | 0.6787 | 0.7571 | 0.8359 | 0.8739 | 0.9105 | 0.9493 | 0.9702 | 0.9834 |

Pinball_avg theo speed, val [H=3; không phụ thuộc tỷ lệ chi phí/L/q]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 1.8014 | 0.7119 | 0.3950 |
| SeasonalNaive | 2.2183 | 0.9999 | 0.6150 |
| CrostonSBA | 2.1232 | 0.8135 | 0.4472 |
| TSB | 1.7472 | 0.6812 | 0.3835 |
| LGBM-Q | 1.4905 | 0.6237 | 0.3609 |
| LGBM-PoisNB | 1.5303 | 0.6310 | 0.3642 |
| MLP-NB | 1.4806 | 0.6143 | 0.3574 |

test [H=3; không phụ thuộc tỷ lệ chi phí/L/q]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 1.9478 | 0.6794 | 0.4445 |
| SeasonalNaive | 2.2158 | 0.9902 | 0.6886 |
| CrostonSBA | 2.2226 | 0.7307 | 0.4701 |
| TSB | 1.8477 | 0.6577 | 0.4278 |
| LGBM-Q | 1.5358 | 0.6139 | 0.4098 |
| LGBM-PoisNB | 1.5661 | 0.6194 | 0.4060 |
| MLP-NB | 1.5679 | 0.6154 | 0.4050 |

WAPE theo speed, val [H=3; không phụ thuộc tỷ lệ chi phí/L/q]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 0.3616 | 0.5519 | 0.7006 |
| SeasonalNaive | 0.4048 | 0.6673 | 0.8418 |
| CrostonSBA | 0.4455 | 0.6231 | 0.7930 |
| TSB | 0.3554 | 0.5404 | 0.6964 |
| LGBM-Q | 0.3123 | 0.5220 | 0.7039 |
| LGBM-PoisNB | 0.3123 | 0.5220 | 0.7039 |
| MLP-NB | 0.3150 | 0.5169 | 0.6888 |

test [H=3; không phụ thuộc tỷ lệ chi phí/L/q]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 0.3460 | 0.5409 | 0.7675 |
| SeasonalNaive | 0.3627 | 0.6613 | 0.9250 |
| CrostonSBA | 0.3990 | 0.5800 | 0.8008 |
| TSB | 0.3372 | 0.5336 | 0.7494 |
| LGBM-Q | 0.2916 | 0.5148 | 0.7447 |
| LGBM-PoisNB | 0.2916 | 0.5148 | 0.7447 |
| MLP-NB | 0.2931 | 0.5141 | 0.7320 |


### 3.2 Chính sách mô phỏng (cost_ratio=10): q*(val) áp dụng trên test, cùng mô hình
L=2 (q_star_val, fill rate val/test, tổng chi phí TB/chuỗi val/test, chi phí chính sách điểm trên test, thay đổi tương đối so với điểm trên val/test):
| model | q_star_val | fill_val | fill_test | cost_val | cost_test | cost_point_test | gain_vs_point_val | gain_vs_point_test |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.8000 | 0.8740 | 0.8683 | 271.4933 | 283.2892 | 306.8226 | -0.0674 | -0.0767 |
| SeasonalNaive | 0.7000 | 0.8347 | 0.8377 | 305.1470 | 308.2199 | 320.7995 | -0.0207 | -0.0392 |
| CrostonSBA | 0.8000 | 0.8484 | 0.8476 | 301.1944 | 304.2625 | 335.2916 | -0.0759 | -0.0925 |
| TSB | 0.8000 | 0.8791 | 0.8799 | 269.3244 | 277.9247 | 301.0486 | -0.0729 | -0.0768 |
| LGBM-Q | 0.7000 | 0.8785 | 0.8744 | 255.8554 | 261.4253 | 283.9842 | -0.0733 | -0.0794 |
| LGBM-PoisNB | 0.8000 | 0.9002 | 0.9024 | 256.4374 | 259.9672 | 283.9842 | -0.0712 | -0.0846 |
| MLP-NB | 0.8000 | 0.9062 | 0.9034 | 252.4508 | 258.5030 | 275.8068 | -0.0546 | -0.0627 |

L=3:
| model | q_star_val | fill_val | fill_test | cost_val | cost_test | cost_point_test | gain_vs_point_val | gain_vs_point_test |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.7000 | 0.8408 | 0.8294 | 289.8676 | 306.9745 | 317.4459 | -0.0336 | -0.0330 |
| SeasonalNaive | 0.6000 | 0.8061 | 0.8052 | 313.9903 | 320.0680 | 323.4046 | -0.0065 | -0.0103 |
| CrostonSBA | 0.7000 | 0.8069 | 0.8014 | 326.9465 | 333.4617 | 348.8007 | -0.0335 | -0.0440 |
| TSB | 0.7000 | 0.8473 | 0.8430 | 288.6549 | 301.4374 | 312.3439 | -0.0354 | -0.0349 |
| LGBM-Q | 0.7000 | 0.8806 | 0.8776 | 274.3378 | 279.8408 | 294.0911 | -0.0439 | -0.0485 |
| LGBM-PoisNB | 0.7000 | 0.8632 | 0.8648 | 274.4982 | 279.8323 | 294.0911 | -0.0434 | -0.0485 |
| MLP-NB | 0.7000 | 0.8695 | 0.8633 | 270.0267 | 278.5772 | 290.9210 | -0.0328 | -0.0424 |

Thay đổi chi phí TEST của q*(val) so với chính sách điểm, L=2, theo tỷ lệ chi phí (âm = rẻ hơn):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -0.1091 | -0.0688 | 0.0004 | -0.0767 | -0.2740 | -0.5390 |
| SeasonalNaive | -0.0723 | -0.0530 | -0.0138 | -0.0392 | -0.1906 | -0.4419 |
| CrostonSBA | -0.1188 | -0.0726 | -0.0002 | -0.0925 | -0.2926 | -0.5614 |
| TSB | -0.1106 | -0.0732 | -0.0041 | -0.0768 | -0.2712 | -0.5440 |
| LGBM-Q | -0.0569 | -0.0410 | -0.0079 | -0.0794 | -0.2726 | -0.5507 |
| LGBM-PoisNB | -0.1148 | -0.0767 | -0.0082 | -0.0846 | -0.2766 | -0.5474 |
| MLP-NB | -0.1052 | -0.0749 | -0.0098 | -0.0627 | -0.2352 | -0.5054 |

Mô hình có chi phí test thấp nhất tại q*(val) cho mỗi (L, tỷ lệ) (chỉ để tham khảo – KHÔNG dùng để đổi lựa chọn):
| L | cost_ratio | model | q_star_val | cost_test |
|---|---|---|---|---|
| 2 | 1.0000 | WindowAverage | 0.5000 | 126.1652 |
| 2 | 2.0000 | LGBM-PoisNB | 0.5000 | 146.2916 |
| 2 | 5.0000 | MLP-NB | 0.5000 | 202.0352 |
| 2 | 10.0000 | MLP-NB | 0.8000 | 258.5030 |
| 2 | 20.0000 | LGBM-PoisNB | 0.9000 | 318.9684 |
| 2 | 50.0000 | LGBM-Q | 0.9500 | 409.6598 |
| 3 | 1.0000 | LGBM-PoisNB | 0.5000 | 142.0462 |
| 3 | 2.0000 | LGBM-PoisNB | 0.5000 | 160.6731 |
| 3 | 5.0000 | MLP-NB | 0.5000 | 213.3050 |
| 3 | 10.0000 | MLP-NB | 0.7000 | 278.5772 |
| 3 | 20.0000 | LGBM-PoisNB | 0.8500 | 349.1908 |
| 3 | 50.0000 | LGBM-PoisNB | 0.9500 | 455.6756 |


### 3.3 Chính sách theo nhóm vs toàn cục vs điểm (cost_ratio=10; chốt từ val)
| L | policy | split | fill_rate | stockout_days | avg_inventory | holding_cost | stockout_cost | total_cost |
|---|---|---|---|---|---|---|---|---|
| 2 | A_theo_nhom | val | 0.8963 | 2.7120 | 6.2212 | 174.1932 | 77.6549 | 251.8481 |
| 2 | B_toan_cuc (MLP-NB, q=0.8) | val | 0.9062 | 2.5152 | 6.5066 | 182.1835 | 70.2673 | 252.4508 |
| 2 | C_diem (MLP-NB) | val | 0.8240 | 4.5176 | 4.8282 | 135.1896 | 131.8348 | 267.0243 |
| 2 | A_theo_nhom | test | 0.8918 | 2.7679 | 6.2146 | 174.0097 | 86.8044 | 260.8141 |
| 2 | B_toan_cuc (MLP-NB, q=0.8) | test | 0.9034 | 2.5358 | 6.4662 | 181.0547 | 77.4484 | 258.5030 |
| 2 | C_diem (MLP-NB) | test | 0.8210 | 4.6209 | 4.7235 | 132.2588 | 143.5480 | 275.8068 |
| 3 | A_theo_nhom | val | 0.8716 | 3.2406 | 6.1902 | 173.3256 | 96.1725 | 269.4982 |
| 3 | B_toan_cuc (MLP-NB, q=0.7) | val | 0.8695 | 3.4241 | 6.1531 | 172.2868 | 97.7400 | 270.0267 |
| 3 | C_diem (MLP-NB) | val | 0.8276 | 4.2661 | 5.3586 | 150.0413 | 129.1373 | 279.1786 |
| 3 | A_theo_nhom | test | 0.8640 | 3.2637 | 6.1211 | 171.3900 | 109.0765 | 280.4666 |
| 3 | B_toan_cuc (MLP-NB, q=0.7) | test | 0.8633 | 3.4058 | 6.0336 | 168.9417 | 109.6355 | 278.5772 |
| 3 | C_diem (MLP-NB) | test | 0.8183 | 4.3888 | 5.1852 | 145.1859 | 145.7351 | 290.9210 |

L=2 theo speed [cost_ratio=10; A: q* theo bảng chính sách từng nhóm, B: MLP-NB q=0.8, C: điểm MLP-NB]:
| group | policy | fill_rate_test | fill_rate_val | total_cost_test | total_cost_val |
|---|---|---|---|---|---|
| fast | A_theo_nhom | 0.9032 | 0.9118 | 455.5706 | 437.4417 |
| fast | B_toan_cuc (MLP-NB, q=0.8) | 0.9169 | 0.9236 | 450.3252 | 438.7822 |
| fast | C_diem (MLP-NB) | 0.8400 | 0.8481 | 486.5031 | 463.3988 |
| medium | A_theo_nhom | 0.8699 | 0.8568 | 154.8598 | 155.2461 |
| medium | B_toan_cuc (MLP-NB, q=0.8) | 0.8699 | 0.8568 | 154.8598 | 155.2461 |
| medium | C_diem (MLP-NB) | 0.7670 | 0.7515 | 161.9595 | 165.1433 |
| slow | A_theo_nhom | 0.7756 | 0.7983 | 93.3182 | 84.2670 |
| slow | B_toan_cuc (MLP-NB, q=0.8) | 0.7983 | 0.8169 | 92.2273 | 84.6023 |
| slow | C_diem (MLP-NB) | 0.7015 | 0.7175 | 93.1818 | 89.1023 |

Bootstrap theo chuỗi, chênh lệch chi phí (A theo nhóm − phương án khác):
| L | split | A_minus | mean_diff | ci_lo | ci_hi |
|---|---|---|---|---|---|
| 2 | val | B_toan_cuc (MLP-NB, q=0.8) | -0.6027 | -1.7450 | 0.4922 |
| 2 | test | B_toan_cuc (MLP-NB, q=0.8) | 2.3111 | 0.9258 | 4.0474 |
| 2 | val | C_diem (MLP-NB) | -15.1762 | -17.8513 | -12.4288 |
| 2 | test | C_diem (MLP-NB) | -14.9927 | -18.5970 | -11.9008 |
| 3 | val | B_toan_cuc (MLP-NB, q=0.7) | -0.5286 | -1.6756 | 0.6051 |
| 3 | test | B_toan_cuc (MLP-NB, q=0.7) | 1.8894 | 0.4409 | 3.3767 |
| 3 | val | C_diem (MLP-NB) | -9.6804 | -11.7803 | -7.5792 |
| 3 | test | C_diem (MLP-NB) | -10.4544 | -13.4681 | -7.6205 |


### 3.4 Câu hỏi nghiên cứu trên test (bootstrap theo chuỗi, L=2)
Q1 – chi phí[phân vị q newsvendor] − chi phí[điểm], [KTC 95%], theo cost_ratio:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -15.45 [-16.03; -14.90] | 14.83 [13.16; 16.68] | 26.96 [23.17; 30.85] | -13.56 [-22.41; -5.58] | -125.79 [-150.78; -104.20] | -561.10 [-638.70; -493.37] |
| SeasonalNaive | -11.24 [-12.11; -10.44] | 30.13 [28.25; 32.18] | 61.81 [57.15; 67.08] | 35.30 [27.89; 42.67] | -47.62 [-65.22; -29.53] | -441.91 [-497.13; -389.31] |
| CrostonSBA | -17.19 [-18.20; -16.26] | 14.77 [12.87; 16.81] | 26.03 [21.25; 30.86] | -21.60 [-31.37; -12.32] | -154.43 [-180.13; -131.04] | -663.59 [-746.25; -590.66] |
| TSB | -16.05 [-16.57; -15.54] | 14.52 [12.90; 16.26] | 27.49 [23.98; 30.89] | -11.44 [-19.79; -3.93] | -119.86 [-144.20; -98.47] | -540.85 [-619.55; -471.12] |
| LGBM-Q | -8.12 [-8.83; -7.48] | 22.42 [21.15; 23.71] | 36.03 [33.26; 38.96] | 0.16 [-7.11; 6.62] | -96.97 [-117.09; -79.21] | -491.24 [-555.43; -434.81] |
| LGBM-PoisNB | -16.39 [-17.04; -15.79] | 13.58 [12.21; 14.99] | 28.94 [25.70; 32.33] | -6.60 [-13.93; -0.50] | -102.11 [-122.20; -85.00] | -486.95 [-551.93; -431.27] |
| MLP-NB | -15.42 [-15.99; -14.84] | 13.32 [11.76; 15.03] | 27.75 [24.51; 31.16] | -1.00 [-7.58; 4.83] | -84.80 [-102.42; -69.98] | -430.20 [-484.08; -382.71] |

theo speed [tỷ lệ chi phí 10, L=2, q=0.9 = q newsvendor; hiệu chi phí phân vị − điểm]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | -45.30 [-65.46; -25.59] | 5.04 [0.34; 9.81] | 11.32 [7.66; 15.33] |
| SeasonalNaive | 21.56 [3.95; 38.73] | 40.26 [34.77; 46.86] | 51.70 [44.37; 58.85] |
| CrostonSBA | -53.07 [-76.40; -30.68] | -4.35 [-9.17; 0.55] | 5.20 [1.85; 8.71] |
| TSB | -44.34 [-63.49; -25.89] | 7.28 [2.74; 11.81] | 15.35 [11.83; 19.01] |
| LGBM-Q | -26.94 [-43.25; -11.89] | 14.69 [9.55; 19.70] | 23.85 [19.36; 28.08] |
| LGBM-PoisNB | -28.89 [-45.60; -13.25] | 4.97 [0.39; 9.42] | 13.61 [10.07; 16.99] |
| MLP-NB | -11.71 [-26.86; 2.39] | 4.26 [-0.01; 8.73] | 9.25 [5.68; 12.60] |

Q3 – chi phí MLP-NB − đối thủ (cùng q newsvendor, tỷ lệ 10):
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: MLP-NB - CrostonSBA (cung q_newsvendor) | -38.88 [-52.61; -27.20] | -86.83 [-120.66; -57.53] | -10.70 [-17.69; -4.93] | -1.47 [-5.75; 2.35] |
| Q3: MLP-NB - LGBM-Q (cung q_newsvendor) | -9.34 [-11.57; -6.90] | -3.00 [-7.95; 2.75] | -13.06 [-14.74; -11.47] | -14.29 [-16.59; -12.20] |
| Q3: MLP-NB - TSB (cung q_newsvendor) | -14.80 [-19.34; -10.73] | -28.24 [-38.95; -18.71] | -6.65 [-9.17; -4.25] | -4.75 [-7.43; -2.16] |

Q3 – pinball_avg MLP-NB − đối thủ (H=3):
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: pinball_avg MLP-NB - CrostonSBA (H=3) | -0.3182 [-0.4210; -0.2305] | -0.6548 [-0.9119; -0.4414] | -0.1152 [-0.1561; -0.0818] | -0.0651 [-0.0940; -0.0405] |
| Q3: pinball_avg MLP-NB - LGBM-Q (H=3) | 0.0123 [0.0015; 0.0253] | 0.0320 [0.0059; 0.0646] | 0.0016 [-0.0039; 0.0071] | -0.0048 [-0.0097; 0.0001] |
| Q3: pinball_avg MLP-NB - TSB (H=3) | -0.1322 [-0.1695; -0.0996] | -0.2798 [-0.3727; -0.2000] | -0.0423 [-0.0555; -0.0297] | -0.0228 [-0.0322; -0.0138] |


## 4. File sinh ra
`results/forecasts/test.npz`, `results/forecast_eval_test_all.csv`, `results/inventory_sim_test.csv`, `results/policy_eval_val_test.csv`, `results/policy_eval_bootstrap.csv`, `results/test_qstar_transfer.csv`, `results/research_questions_test.csv`, `results/timing_test.json`, `results/run_test.log`, `notebooks/04_final_results.ipynb`, `results/figures/final_*.png`.

## 5. Cấu hình
Seed 42. Thời gian chạy test: tổng 227.9 s (LightGBM fit H=3 44.0 s, H=4 42.6 s; MLP fit H=3 56.5 s, H=4 69.5 s; CPU). lightgbm 4.7.0, torch 2.15.0+cpu, scipy 1.18.1, numpy 2.5.4, pandas 2.3.3.

## 6. Quan sát chính và vấn đề
- **Xếp hạng giữa các mô hình học máy đổi chiều từ val sang test**: val pinball_avg (H=3) MLP-NB 0.9025 < LGBM-Q 0.9109; test LGBM-Q 0.9354 < MLP-NB 0.9477 (hiệu MLP-NB − LGBM-Q trên test +0.0123, KTC [0.0015; 0.0253]). Cả hai vẫn tốt hơn TSB (1.0799), WindowAverage (1.1316), CrostonSBA (1.2659), SeasonalNaive (1.4112) trên test. Nghĩa là kết luận "MLP-NB tốt nhất" trên val KHÔNG ổn định; kết luận "mô hình dùng biến ngoại sinh (LightGBM/MLP) hơn Croston/TSB" thì ổn định.
- **Chính sách theo nhóm không tốt hơn chính sách toàn cục trên test**: L=2: theo nhóm − toàn cục (MLP-NB, q=0.8) = +2.311 [0.926; 4.047] (tức theo nhóm ĐẮT hơn, KTC dương), trong khi trên val sự khác biệt là −0.603 [−1.745; 0.492]. Lựa chọn theo nhóm (9 nhóm × 7 mô hình × 9 q) bị overfit validation. L=3: +1.889 [0.441; 3.377]. Cả hai đều rẻ hơn chính sách điểm (L=2: −14.993 [−18.597; −11.901]).
- Hiệu chuẩn trên test gần với val; SeasonalNaive vẫn thiếu phủ ở đuôi (0.880 tại q=0.99).
- Q* chạm cận dưới lưới ở tỷ lệ ≤5 (xem GĐ4); không thay đổi theo test.
- Cảnh báo chung: chỉ 28 ngày test, KTC bootstrap coi các chuỗi độc lập.

## 7. Chưa làm
Không có đánh giá lặp theo nhiều giai đoạn (rolling multi-split), không có DeepAR/N-HiTS/TFT/Chronos-2, không có WRMSSE (lý do ở báo cáo 02).


> **Cập nhật GĐ6**: kết luận câu 1 (phân vị vs điểm) và lời giải thích câu 3 (biến ngoại sinh) đã được sửa/bổ sung ở `06_bo_sung.md` (điểm có hệ số an toàn, ablation ngoại sinh, 3 seed MLP-NB, lưới q mở rộng, hiệu chuẩn rời rạc).
