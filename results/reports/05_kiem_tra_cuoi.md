# Báo cáo GĐ5 – Đánh giá cuối trên TEST

## 1. Đã làm gì
- Chạy `python -m src.chay_du_bao --split test`: huấn luyện lại LightGBM/MLP-NB trên train+val (nhãn đến hết val; MLP-NB dùng đúng số epoch đã chốt: H=3 → 17, H=4 → 30), dùng NGUYÊN siêu tham số của val (`results/sieu_tham_so.json`, sha256 16 ký tự đầu = 8bc7d7507bf0167e, không đổi giữa lần chạy val và test). Chỉ số hiệu ứng thứ/sự kiện của đặc trưng chỉ tính đến hết val.
- `python -m src.mo_phong_ton_kho --split test`, `python -m src.danh_gia_cuoi` (áp dụng q* và chính sách theo nhóm chốt từ val lên test), bootstrap câu hỏi nghiên cứu trên test (`cau_hoi_nghien_cuu_test.csv`).
- File mới: `src/danh_gia_cuoi.py`, `tests/kiem_thu_danh_gia_cuoi.py` (2 test), thêm test `test_test_npz_...` trong `tests/kiem_thu_dau_ra.py`; `notebooks/04_ket_qua_cuoi.ipynb` (Restart & Run All, 0 lỗi); viết lại `notebooks/03_thong_ke_co_so.ipynb` (bản cũ gọi sai hàm); `README.md`, `requirements.txt` (thêm torch), `results/reports/bao_cao_cuoi_cung.md`.
- Test chỉ được chạm một lần theo nghĩa: không có bước chọn/chỉnh nào dùng kết quả test; kết quả test chỉ được đọc để báo cáo.

## 2. Lệnh đã chạy và pytest
```
python -m src.chay_du_bao --split test
python -m src.mo_phong_ton_kho --split test
python -m src.danh_gia_cuoi
python -m pytest -v
```
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

============================= 56 passed in 15.31s =============================
```

## 3. Kết quả
### 3.1 Dự báo, H=3, tổng thể (val vs test)
| model | pinball_avg_val | pinball_avg_test | wape_val | wape_test | mase_val | mase_test |
|---|---|---|---|---|---|---|
| WindowAverage | 1.0757 | 1.1316 | 0.4129 | 0.3987 | 0.8125 | 0.8474 |
| SeasonalNaive | 1.4002 | 1.4112 | 0.4741 | 0.4399 | 0.9645 | 0.9882 |
| CrostonSBA | 1.2526 | 1.2631 | 0.4948 | 0.4484 | 0.9502 | 0.9174 |
| TSB | 1.0382 | 1.0772 | 0.4058 | 0.3898 | 0.8009 | 0.8287 |
| LGBM-Q | 0.9109 | 0.9354 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| LGBM-PoisNB | 0.9300 | 0.9490 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| MLP-NB | 0.9025 | 0.9477 | 0.3701 | 0.3512 | 0.7597 | 0.7839 |

H=4:
| model | pinball_avg_val | pinball_avg_test | wape_val | wape_test | mase_val | mase_test |
|---|---|---|---|---|---|---|
| WindowAverage | 1.2918 | 1.3782 | 0.3682 | 0.3575 | 0.8320 | 0.8677 |
| SeasonalNaive | 1.6261 | 1.6239 | 0.4214 | 0.3859 | 0.9792 | 0.9882 |
| CrostonSBA | 1.5195 | 1.5381 | 0.4506 | 0.4060 | 0.9886 | 0.9445 |
| TSB | 1.2372 | 1.2991 | 0.3605 | 0.3474 | 0.8173 | 0.8444 |
| LGBM-Q | 1.0922 | 1.1344 | 0.3333 | 0.3166 | 0.7884 | 0.8051 |
| LGBM-PoisNB | 1.1192 | 1.1506 | 0.3333 | 0.3166 | 0.7884 | 0.8051 |
| MLP-NB | 1.0929 | 1.1489 | 0.3328 | 0.3165 | 0.7746 | 0.8024 |

Coverage P(y≤Q) H=3:
| model | cov_q0.5_val | cov_q0.5_test | cov_q0.9_val | cov_q0.9_test | cov_q0.99_val | cov_q0.99_test |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5850 | 0.5694 | 0.8897 | 0.8906 | 0.9678 | 0.9688 |
| SeasonalNaive | 0.5765 | 0.5518 | 0.8043 | 0.7974 | 0.8844 | 0.8805 |
| CrostonSBA | 0.5384 | 0.5165 | 0.8774 | 0.8804 | 0.9702 | 0.9691 |
| TSB | 0.5880 | 0.5751 | 0.8954 | 0.9017 | 0.9734 | 0.9769 |
| LGBM-Q | 0.5412 | 0.5127 | 0.9063 | 0.9027 | 0.9881 | 0.9874 |
| LGBM-PoisNB | 0.5978 | 0.5872 | 0.9140 | 0.9176 | 0.9827 | 0.9844 |
| MLP-NB | 0.6198 | 0.6004 | 0.9135 | 0.9105 | 0.9874 | 0.9834 |

Coverage test mọi q (H=3):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.5694 | 0.6488 | 0.7303 | 0.8092 | 0.8499 | 0.8906 | 0.9318 | 0.9539 | 0.9688 |
| SeasonalNaive | 0.5518 | 0.5944 | 0.6655 | 0.7356 | 0.7677 | 0.7974 | 0.8354 | 0.8634 | 0.8805 |
| CrostonSBA | 0.5165 | 0.6050 | 0.6960 | 0.7882 | 0.8346 | 0.8804 | 0.9278 | 0.9522 | 0.9691 |
| TSB | 0.5751 | 0.6607 | 0.7407 | 0.8189 | 0.8598 | 0.9017 | 0.9415 | 0.9617 | 0.9769 |
| LGBM-Q | 0.5127 | 0.6069 | 0.7062 | 0.8064 | 0.8553 | 0.9027 | 0.9476 | 0.9720 | 0.9874 |
| LGBM-PoisNB | 0.5872 | 0.6704 | 0.7552 | 0.8401 | 0.8790 | 0.9176 | 0.9533 | 0.9724 | 0.9844 |
| MLP-NB | 0.6004 | 0.6787 | 0.7571 | 0.8359 | 0.8739 | 0.9105 | 0.9493 | 0.9702 | 0.9834 |

Pinball_avg theo speed, val:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 1.8014 | 0.7119 | 0.3950 |
| SeasonalNaive | 2.2183 | 0.9999 | 0.6150 |
| CrostonSBA | 2.1225 | 0.8112 | 0.4463 |
| TSB | 1.7432 | 0.6811 | 0.3839 |
| LGBM-Q | 1.4905 | 0.6237 | 0.3609 |
| LGBM-PoisNB | 1.5302 | 0.6307 | 0.3641 |
| MLP-NB | 1.4806 | 0.6143 | 0.3574 |

test:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 1.9478 | 0.6794 | 0.4445 |
| SeasonalNaive | 2.2158 | 0.9902 | 0.6886 |
| CrostonSBA | 2.2152 | 0.7315 | 0.4693 |
| TSB | 1.8410 | 0.6576 | 0.4278 |
| LGBM-Q | 1.5358 | 0.6139 | 0.4098 |
| LGBM-PoisNB | 1.5664 | 0.6196 | 0.4063 |
| MLP-NB | 1.5679 | 0.6154 | 0.4050 |

WAPE theo speed, val:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 0.3616 | 0.5519 | 0.7006 |
| SeasonalNaive | 0.4048 | 0.6673 | 0.8418 |
| CrostonSBA | 0.4455 | 0.6231 | 0.7930 |
| TSB | 0.3554 | 0.5404 | 0.6964 |
| LGBM-Q | 0.3123 | 0.5220 | 0.7039 |
| LGBM-PoisNB | 0.3123 | 0.5220 | 0.7039 |
| MLP-NB | 0.3150 | 0.5169 | 0.6888 |

test:
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
| CrostonSBA | 0.8000 | 0.8529 | 0.8522 | 301.4739 | 304.2539 | 335.2916 | -0.0751 | -0.0926 |
| TSB | 0.8000 | 0.8827 | 0.8835 | 269.6221 | 278.0923 | 301.0486 | -0.0719 | -0.0763 |
| LGBM-Q | 0.8000 | 0.9069 | 0.9028 | 254.4751 | 258.5018 | 283.9842 | -0.0783 | -0.0897 |
| LGBM-PoisNB | 0.8000 | 0.9014 | 0.9035 | 256.5772 | 260.0826 | 283.9842 | -0.0707 | -0.0842 |
| MLP-NB | 0.8000 | 0.9062 | 0.9034 | 252.4508 | 258.5030 | 275.8068 | -0.0546 | -0.0627 |

L=3:
| model | q_star_val | fill_val | fill_test | cost_val | cost_test | cost_point_test | gain_vs_point_val | gain_vs_point_test |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.7000 | 0.8408 | 0.8294 | 289.8676 | 306.9745 | 317.4459 | -0.0336 | -0.0330 |
| SeasonalNaive | 0.6000 | 0.8061 | 0.8052 | 313.9903 | 320.0680 | 323.4046 | -0.0065 | -0.0103 |
| CrostonSBA | 0.7000 | 0.8091 | 0.8035 | 326.5699 | 333.0826 | 348.8007 | -0.0346 | -0.0451 |
| TSB | 0.7000 | 0.8494 | 0.8451 | 288.3499 | 300.9453 | 312.3439 | -0.0364 | -0.0365 |
| LGBM-Q | 0.7000 | 0.8663 | 0.8642 | 273.0911 | 279.3499 | 294.0911 | -0.0483 | -0.0501 |
| LGBM-PoisNB | 0.7000 | 0.8637 | 0.8653 | 274.4885 | 279.7606 | 294.0911 | -0.0434 | -0.0487 |
| MLP-NB | 0.7000 | 0.8695 | 0.8633 | 270.0267 | 278.5772 | 290.9210 | -0.0328 | -0.0424 |

Thay đổi chi phí TEST của q*(val) so với chính sách điểm, L=2, theo tỷ lệ chi phí (âm = rẻ hơn):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -0.1091 | -0.0688 | 0.0004 | -0.0767 | -0.2740 | -0.5390 |
| SeasonalNaive | -0.0723 | -0.0530 | -0.0138 | -0.0392 | -0.1906 | -0.4419 |
| CrostonSBA | -0.1275 | -0.0774 | 0.0004 | -0.0926 | -0.2937 | -0.5586 |
| TSB | -0.1163 | -0.0764 | -0.0039 | -0.0763 | -0.2727 | -0.5405 |
| LGBM-Q | -0.1177 | -0.0829 | -0.0102 | -0.0897 | -0.2666 | -0.5500 |
| LGBM-PoisNB | -0.1169 | -0.0779 | -0.0086 | -0.0842 | -0.2763 | -0.5476 |
| MLP-NB | -0.1052 | -0.0749 | -0.0098 | -0.0627 | -0.2352 | -0.5054 |

Mô hình có chi phí test thấp nhất tại q*(val) cho mỗi (L, tỷ lệ) (chỉ để tham khảo – KHÔNG dùng để đổi lựa chọn):
| L | cost_ratio | model | q_star_val | cost_test |
|---|---|---|---|---|
| 2 | 1.0000 | LGBM-Q | 0.5000 | 125.9441 |
| 2 | 2.0000 | LGBM-Q | 0.5000 | 145.3135 |
| 2 | 5.0000 | MLP-NB | 0.5000 | 202.0352 |
| 2 | 10.0000 | LGBM-Q | 0.8000 | 258.5018 |
| 2 | 20.0000 | LGBM-PoisNB | 0.9000 | 319.0741 |
| 2 | 50.0000 | LGBM-Q | 0.9500 | 410.2260 |
| 3 | 1.0000 | LGBM-PoisNB | 0.5000 | 141.6367 |
| 3 | 2.0000 | LGBM-Q | 0.5000 | 160.1373 |
| 3 | 5.0000 | MLP-NB | 0.5000 | 213.3050 |
| 3 | 10.0000 | MLP-NB | 0.7000 | 278.5772 |
| 3 | 20.0000 | LGBM-Q | 0.8500 | 348.4313 |
| 3 | 50.0000 | LGBM-Q | 0.9500 | 455.2746 |


### 3.3 Chính sách theo nhóm vs toàn cục vs điểm (cost_ratio=10; chốt từ val)
| L | policy | split | fill_rate | stockout_days | avg_inventory | holding_cost | stockout_cost | total_cost |
|---|---|---|---|---|---|---|---|---|
| 2 | A_theo_nhom | val | 0.8991 | 2.6634 | 6.2960 | 176.2880 | 75.5893 | 251.8773 |
| 2 | B_toan_cuc (MLP-NB, q=0.8) | val | 0.9062 | 2.5152 | 6.5066 | 182.1835 | 70.2673 | 252.4508 |
| 2 | C_diem (MLP-NB) | val | 0.8240 | 4.5176 | 4.8282 | 135.1896 | 131.8348 | 267.0243 |
| 2 | A_theo_nhom | test | 0.8948 | 2.7132 | 6.2469 | 174.9125 | 84.3621 | 259.2746 |
| 2 | B_toan_cuc (MLP-NB, q=0.8) | test | 0.9034 | 2.5358 | 6.4662 | 181.0547 | 77.4484 | 258.5030 |
| 2 | C_diem (MLP-NB) | test | 0.8210 | 4.6209 | 4.7235 | 132.2588 | 143.5480 | 275.8068 |
| 3 | A_theo_nhom | val | 0.8707 | 3.2576 | 6.1720 | 172.8153 | 96.8773 | 269.6926 |
| 3 | B_toan_cuc (MLP-NB, q=0.7) | val | 0.8695 | 3.4241 | 6.1531 | 172.2868 | 97.7400 | 270.0267 |
| 3 | C_diem (MLP-NB) | val | 0.8276 | 4.2661 | 5.3586 | 150.0413 | 129.1373 | 279.1786 |
| 3 | A_theo_nhom | test | 0.8632 | 3.2807 | 6.0981 | 170.7473 | 109.7084 | 280.4557 |
| 3 | B_toan_cuc (MLP-NB, q=0.7) | test | 0.8633 | 3.4058 | 6.0336 | 168.9417 | 109.6355 | 278.5772 |
| 3 | C_diem (MLP-NB) | test | 0.8183 | 4.3888 | 5.1852 | 145.1859 | 145.7351 | 290.9210 |

L=2 theo speed:
| group | policy | fill_rate_test | fill_rate_val | total_cost_test | total_cost_val |
|---|---|---|---|---|---|
| fast | A_theo_nhom | 0.9071 | 0.9155 | 451.7362 | 437.4141 |
| fast | B_toan_cuc (MLP-NB, q=0.8) | 0.9169 | 0.9236 | 450.3252 | 438.7822 |
| fast | C_diem (MLP-NB) | 0.8400 | 0.8481 | 486.5031 | 463.3988 |
| medium | A_theo_nhom | 0.8699 | 0.8568 | 154.8598 | 155.2461 |
| medium | B_toan_cuc (MLP-NB, q=0.8) | 0.8699 | 0.8568 | 154.8598 | 155.2461 |
| medium | C_diem (MLP-NB) | 0.7670 | 0.7515 | 161.9595 | 165.1433 |
| slow | A_theo_nhom | 0.7753 | 0.7965 | 93.2216 | 84.4545 |
| slow | B_toan_cuc (MLP-NB, q=0.8) | 0.7983 | 0.8169 | 92.2273 | 84.6023 |
| slow | C_diem (MLP-NB) | 0.7015 | 0.7175 | 93.1818 | 89.1023 |

Bootstrap theo chuỗi, chênh lệch chi phí (A theo nhóm − phương án khác):
| L | split | A_minus | mean_diff | ci_lo | ci_hi |
|---|---|---|---|---|---|
| 2 | val | B_toan_cuc (MLP-NB, q=0.8) | -0.5735 | -1.7258 | 0.5044 |
| 2 | test | B_toan_cuc (MLP-NB, q=0.8) | 0.7716 | -1.0705 | 2.6975 |
| 2 | val | C_diem (MLP-NB) | -15.1470 | -18.1034 | -12.1394 |
| 2 | test | C_diem (MLP-NB) | -16.5322 | -20.8324 | -12.8661 |
| 3 | val | B_toan_cuc (MLP-NB, q=0.7) | -0.3341 | -1.4752 | 0.7900 |
| 3 | test | B_toan_cuc (MLP-NB, q=0.7) | 1.8785 | 0.4080 | 3.3635 |
| 3 | val | C_diem (MLP-NB) | -9.4860 | -11.5227 | -7.4361 |
| 3 | test | C_diem (MLP-NB) | -10.4654 | -13.5099 | -7.6581 |


### 3.4 Câu hỏi nghiên cứu trên test (bootstrap theo chuỗi, L=2)
Q1 – chi phí[phân vị q newsvendor] − chi phí[điểm], [KTC 95%], theo cost_ratio:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -15.45 [-16.03; -14.90] | 14.83 [13.16; 16.68] | 26.96 [23.17; 30.85] | -13.56 [-22.41; -5.58] | -125.79 [-150.78; -104.20] | -561.10 [-638.70; -493.37] |
| SeasonalNaive | -11.24 [-12.11; -10.44] | 30.13 [28.25; 32.18] | 61.81 [57.15; 67.08] | 35.30 [27.89; 42.67] | -47.62 [-65.22; -29.53] | -441.91 [-497.13; -389.31] |
| CrostonSBA | -18.45 [-19.56; -17.39] | 15.40 [13.40; 17.55] | 29.28 [24.13; 34.52] | -17.54 [-27.57; -7.86] | -147.91 [-173.50; -124.40] | -660.25 [-745.09; -586.25] |
| TSB | -16.88 [-17.47; -16.32] | 15.18 [13.43; 17.07] | 30.24 [26.53; 33.91] | -8.38 [-17.08; -0.61] | -115.58 [-140.50; -93.55] | -537.32 [-618.04; -466.32] |
| LGBM-Q | -16.81 [-17.61; -16.12] | 13.40 [12.13; 14.71] | 27.34 [24.57; 30.17] | -7.77 [-14.81; -1.41] | -104.94 [-124.96; -87.48] | -497.65 [-561.19; -441.81] |
| LGBM-PoisNB | -16.68 [-17.37; -16.06] | 13.83 [12.42; 15.31] | 29.97 [26.61; 33.47] | -5.36 [-12.62; 0.81] | -99.72 [-119.64; -82.52] | -484.08 [-549.00; -428.01] |
| MLP-NB | -15.42 [-15.99; -14.84] | 13.32 [11.76; 15.03] | 27.75 [24.51; 31.16] | -1.00 [-7.58; 4.83] | -84.80 [-102.42; -69.98] | -430.20 [-484.08; -382.71] |

theo speed, cost_ratio=10:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | -45.30 [-65.46; -25.59] | 5.04 [0.34; 9.81] | 11.32 [7.66; 15.33] |
| SeasonalNaive | 21.56 [3.95; 38.73] | 40.26 [34.77; 46.86] | 51.70 [44.37; 58.85] |
| CrostonSBA | -45.47 [-69.61; -21.75] | -2.25 [-7.29; 2.88] | 6.31 [2.89; 9.85] |
| TSB | -39.41 [-59.58; -20.18] | 9.41 [4.68; 14.06] | 16.64 [12.95; 20.38] |
| LGBM-Q | -33.88 [-49.71; -18.96] | 6.30 [1.51; 11.24] | 14.95 [10.98; 18.71] |
| LGBM-PoisNB | -26.85 [-43.60; -11.23] | 5.90 [1.29; 10.42] | 13.91 [10.29; 17.31] |
| MLP-NB | -11.71 [-26.86; 2.39] | 4.26 [-0.01; 8.73] | 9.25 [5.68; 12.60] |

Q3 – chi phí MLP-NB − đối thủ (cùng q newsvendor, tỷ lệ 10):
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: MLP-NB - CrostonSBA (cung q_newsvendor) | -42.95 [-57.26; -30.79] | -94.43 [-129.14; -63.92] | -12.80 [-19.80; -6.96] | -2.59 [-6.88; 1.27] |
| Q3: MLP-NB - LGBM-Q (cung q_newsvendor) | -1.41 [-3.53; 1.04] | 3.95 [-0.92; 9.72] | -4.67 [-6.12; -3.27] | -5.39 [-7.57; -3.34] |
| Q3: MLP-NB - TSB (cung q_newsvendor) | -17.86 [-22.63; -13.69] | -33.17 [-44.23; -23.48] | -8.78 [-11.49; -6.15] | -6.05 [-8.88; -3.42] |

Q3 – pinball_avg MLP-NB − đối thủ (H=3):
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: pinball_avg MLP-NB - CrostonSBA (H=3) | -0.3154 [-0.4172; -0.2290] | -0.6473 [-0.9000; -0.4365] | -0.1161 [-0.1578; -0.0825] | -0.0643 [-0.0928; -0.0401] |
| Q3: pinball_avg MLP-NB - LGBM-Q (H=3) | 0.0123 [0.0015; 0.0253] | 0.0320 [0.0059; 0.0646] | 0.0016 [-0.0039; 0.0071] | -0.0048 [-0.0097; 0.0001] |
| Q3: pinball_avg MLP-NB - TSB (H=3) | -0.1295 [-0.1670; -0.0968] | -0.2731 [-0.3659; -0.1934] | -0.0422 [-0.0554; -0.0297] | -0.0228 [-0.0326; -0.0134] |


## 4. File sinh ra
`results/forecasts/test.npz`, `results/danh_gia_du_bao_test_all.csv`, `results/mo_phong_ton_kho_test.csv`, `results/danh_gia_chinh_sach_val_test.csv`, `results/danh_gia_chinh_sach_bootstrap.csv`, `results/chuyen_q_sao_test.csv`, `results/cau_hoi_nghien_cuu_test.csv`, `results/thoi_gian_test.json`, `results/chay_test.log`, `notebooks/04_ket_qua_cuoi.ipynb`, `results/figures/cuoi_*.png`.

## 5. Cấu hình
Seed 42. Thời gian chạy test: tổng 309.7 s (LightGBM fit H=3 74.7 s, H=4 79.9 s; MLP fit H=3 52.5 s, H=4 84.7 s; CPU). lightgbm 4.7.0, torch 2.15.0+cpu, scipy 1.18.1, numpy 2.5.4, pandas 2.3.3.

## 6. Quan sát chính và vấn đề
- **Xếp hạng giữa các mô hình học máy đổi chiều từ val sang test**: val pinball_avg (H=3) MLP-NB 0.9025 < LGBM-Q 0.9109; test LGBM-Q 0.9354 < MLP-NB 0.9477 (hiệu MLP-NB − LGBM-Q trên test +0.0123 [0.0015; 0.0253]). Cả hai vẫn tốt hơn TSB (1.0772), WindowAverage (1.1316), CrostonSBA (1.2631), SeasonalNaive (1.4112) trên test. Kết luận "mô hình dùng đặc trưng ngoài lịch sử doanh số thô (LightGBM/MLP) hơn Croston/TSB" ổn định; "MLP-NB tốt nhất" thì không (xem `06`, `07`: 3 seed, ablation).
- **Chính sách theo nhóm vs toàn cục (test, cost_ratio=10)**: L=2: theo nhóm − toàn cục (MLP-NB, q=0.8) = +0.772 [-1.070; 2.698] (KTC chứa 0: không phân biệt được); trên val -0.574 [-1.726; 0.504]. L=3 (MLP-NB, q=0.7): +1.878 [0.408; 3.364] (dương: theo nhóm đắt hơn); trên val -0.334 [-1.475; 0.790]. Cả hai chính sách rẻ hơn chính sách điểm thuần (L=2: -16.532 [-20.832; -12.866]). Tức lựa chọn theo nhóm không tốt hơn lựa chọn toàn cục trên test, và ở L=3 còn đắt hơn có ý nghĩa thống kê.
- Hiệu chuẩn trên test gần với val; SeasonalNaive vẫn thiếu phủ ở đuôi (0.880 tại q=0.99).
- q* chạm cận dưới lưới chính (0.5) ở tỷ lệ ≤5; lưới mở rộng ở `06`, `07`.
- Cảnh báo chung: chỉ 28 ngày test, KTC bootstrap coi các chuỗi độc lập.
- **Cập nhật GĐ7**: báo cáo này được dựng lại từ kết quả sau khi (i) làm tròn phân vị LGBM-Q về số nguyên trong mô phỏng và (ii) mở lưới kappa; chi tiết thay đổi ở `07_sua_loi.md`.

## 7. Chưa làm
Không có đánh giá lặp theo nhiều giai đoạn (rolling multi-split), không có DeepAR/N-HiTS/TFT/Chronos-2, không có WRMSSE (lý do ở báo cáo 02).
