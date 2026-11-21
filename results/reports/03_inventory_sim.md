# Báo cáo GĐ3 – Mô phỏng chính sách tồn kho (validation)

> **Lưu ý GĐ7**: báo cáo này là ảnh chụp tại GĐ3, khi mô phỏng dùng `ceil` cho mọi phân vị. Từ GĐ7 phân vị LGBM-Q được làm tròn về số nguyên gần nhất (NB giữ nguyên) nên các số mô phỏng của LGBM-Q (và nhẹ ở CrostonSBA/TSB/LGBM-PoisNB do kappa) khác bản này; xem `07_sua_loi.md` mục 3.1. Bản cũ ở `archive/pre_gd7/`.

## 1. Đã làm gì
- `src/inventory_sim.py`: `simulate_order_up_to` (mô phỏng song song nhiều kịch bản, có trace theo ngày), `aggregate` (gộp theo nhóm × cost_ratio), `run_inventory_sim` (mọi mô hình × q × L + chính sách điểm), CLI `python -m src.inventory_sim --split val`.
- `tests/test_inventory_sim.py` (4 test). `configs/config.yaml`: thêm tham số chi phí (mục dưới).
- Kết quả: `results/inventory_sim_val.csv` (14280 dòng).

## 2. Giả định và quy ước mô phỏng (áp dụng GIỐNG HỆT cho mọi mô hình)
Tham số trong config: holding_cost=1.0, stockout_cost_ratio=10, cost_ratios=[1, 2, 5, 10, 20, 50], lead_times=[2, 3], review_period=1, lost_sales=True, initial_inventory_rule=order_up_to_at_start.
- **Chi phí là tương đối, KHÔNG phải chi phí thực**: giữ 1 đơn vị-ngày = 1.0; thiếu 1 đơn vị = cost_ratio × 1.0. Cơ sở cost_ratio = 10 là giả định (tương ứng mức phục vụ newsvendor ≈ 10/11 ≈ 0.91), được quét ở GĐ4 (1,2,5,10,20,50). Không có dữ liệu chi phí thật của M5.
- Gốc t0 = 2016-03-27 (cuối train). Ngày d=1..28 ứng với 2016-03-28..2016-04-24 (nhu cầu thực). Cuối ngày t0+d đặt hàng (d=0..n−1, n=28−H+1) với S = ceil(phân vị q của D_H(t0+d)); đặt max(0, S − (tồn tay + hàng đang về)); hàng về cuối ngày t0+d+L, dùng được từ ngày t0+d+H. Orders cho gốc d>n−1 không đặt vì cửa sổ mục tiêu vượt khỏi split (và hàng về cũng quá muộn để phục vụ nhu cầu trong 28 ngày).
- Thiếu hàng bị mất (lost sales, không backorder). Tồn kho cuối ngày = sau bán và sau nhận hàng cuối ngày; chi phí giữ tính trên tồn này. Tồn đầu = S của gốc đầu tiên, không có hàng đang về (order_up_to_at_start).
- Fill rate = tổng bán/tổng cầu của nhóm (gộp theo khối lượng). Số ngày hết hàng = số ngày có thiếu hàng (lost > 0), trung bình theo chuỗi. Mọi chi phí/tồn kho là trung bình theo chuỗi trên 28 ngày.
- Chính sách điểm (đối chứng câu hỏi 1): S = ceil(trung bình dự báo của D_H) của cùng mô hình. LGBM-Q và LGBM-PoisNB dùng chung mean Poisson ⇒ chính sách điểm của hai mô hình này GIỐNG HỆT nhau (số liệu trùng).
- Dự báo dùng cho mô phỏng là dự báo val của GĐ2 (siêu tham số đã chọn trên val ⇒ có lạc quan nhẹ).

## 3. Lệnh đã chạy và pytest
`python -m src.inventory_sim --split val` (thời gian: 1.5 s), `python -m pytest -v`:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\lt\261\PTDLHS\DA\.venv\Scripts\python.exe
cachedir: .pytest_cache
Fugue tests will be initialized with options:
rootdir: D:\lt\261\PTDLHS\DA
plugins: anyio-4.15.1, fugue-0.9.7, platformdirs-4.14.0
collecting ... collected 31 items

tests/test_eda.py::test_load_train_excludes_val PASSED                   [  3%]
tests/test_eda.py::test_active_rows_drops_pre_launch_days PASSED         [  6%]
tests/test_eda.py::test_weekday_table_has_seven_rows_and_mean_two PASSED [  9%]
tests/test_eda.py::test_demand_speed_table_totals PASSED                 [ 12%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_daily_target_date_tracks_horizon_offset PASSED [ 16%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_evaluate_validation_summary_matches_by_date_and_id PASSED [ 19%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_join_rows_do_not_duplicate PASSED [ 22%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_n_samples_changes_quantile_output PASSED [ 25%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_pinball_known_answer PASSED [ 29%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_poisson_distribution_scaling PASSED [ 32%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_sample_total_window_does_not_blow_up_for_zero_mean PASSED [ 35%]
tests/test_inventory_sim.py::test_constant_demand_matches_hand_calculation PASSED [ 38%]
tests/test_inventory_sim.py::test_inventory_conservation_every_day PASSED [ 41%]
tests/test_inventory_sim.py::test_costs_nonnegative_and_fill_rate_range_and_monotone_in_level PASSED [ 45%]
tests/test_inventory_sim.py::test_reproducible_same_inputs PASSED        [ 48%]
tests/test_outputs.py::test_val_npz_shapes_cover_all_series_and_origins_and_monotone PASSED [ 51%]
tests/test_outputs.py::test_val_eval_csv_schema_and_coverage_matches_npz PASSED [ 54%]
tests/test_outputs.py::test_actual_in_npz_equals_window_sum_of_val_demand PASSED [ 58%]
tests/test_outputs.py::test_md_table_rounds_to_four_decimals PASSED      [ 61%]
tests/test_prob_models.py::test_pinball_known_answer PASSED              [ 64%]
tests/test_prob_models.py::test_nb_quantiles_match_scipy_and_cdf_definition PASSED [ 67%]
tests/test_prob_models.py::test_nb_quantiles_monotone_in_q_and_zero_mean PASSED [ 70%]
tests/test_prob_models.py::test_legacy_sampler_mean_not_inflated_when_n_below_one PASSED [ 74%]
tests/test_prob_models.py::test_window_targets_values_and_nan_tail PASSED [ 77%]
tests/test_prob_models.py::test_origins_window_inside_split PASSED       [ 80%]
tests/test_prob_models.py::test_croston_tsb_constant_demand PASSED       [ 83%]
tests/test_prob_models.py::test_no_leak_stat_forecast_and_alpha PASSED   [ 87%]
tests/test_prob_models.py::test_no_leak_features PASSED                  [ 90%]
tests/test_prob_models.py::test_estimate_alpha_known_poisson_is_small PASSED [ 93%]
tests/test_prob_models.py::test_evaluate_forecast_coverage_and_shapes PASSED [ 96%]
tests/test_prob_models.py::test_lgbm_quantiles_monotone_and_shape PASSED [100%]

============================= 31 passed in 3.58s ==============================
```

## 4. Kết quả chính (copy từ `results/inventory_sim_val.csv`, làm tròn 4 chữ số)
### L=2 (H=3), cost_ratio=10, overall (823 chuỗi)
Fill rate (tổng bán / tổng cầu):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.7214 | 0.7736 | 0.8249 | 0.8740 | 0.8990 | 0.9244 | 0.9502 | 0.9641 | 0.9745 | 0.7756 |
| SeasonalNaive | 0.7468 | 0.7870 | 0.8347 | 0.8777 | 0.8996 | 0.9210 | 0.9431 | 0.9561 | 0.9649 | 0.7702 |
| CrostonSBA | 0.6725 | 0.7309 | 0.7883 | 0.8484 | 0.8786 | 0.9103 | 0.9443 | 0.9638 | 0.9777 | 0.7365 |
| TSB | 0.7316 | 0.7821 | 0.8304 | 0.8791 | 0.9030 | 0.9282 | 0.9544 | 0.9692 | 0.9795 | 0.7825 |
| LGBM-Q | 0.7863 | 0.8349 | 0.8785 | 0.9188 | 0.9399 | 0.9579 | 0.9759 | 0.9865 | 0.9933 | 0.8037 |
| LGBM-PoisNB | 0.7527 | 0.8045 | 0.8533 | 0.9002 | 0.9224 | 0.9450 | 0.9670 | 0.9782 | 0.9861 | 0.8037 |
| MLP-NB | 0.7810 | 0.8247 | 0.8661 | 0.9062 | 0.9257 | 0.9462 | 0.9685 | 0.9815 | 0.9904 | 0.8240 |

Số ngày hết hàng TB/chuỗi (trên 28 ngày):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 6.7473 | 5.5735 | 4.3281 | 3.1021 | 2.5322 | 1.8688 | 1.1798 | 0.7837 | 0.5383 | 5.2296 |
| SeasonalNaive | 6.0717 | 5.2564 | 3.9757 | 2.9417 | 2.3961 | 1.8712 | 1.3098 | 0.9611 | 0.7582 | 5.5772 |
| CrostonSBA | 7.6561 | 6.3402 | 5.1069 | 3.7217 | 3.0170 | 2.2880 | 1.4022 | 0.9174 | 0.5334 | 5.9162 |
| TSB | 6.3524 | 5.1908 | 4.0510 | 2.8821 | 2.3171 | 1.6914 | 1.0620 | 0.7011 | 0.4301 | 4.9283 |
| LGBM-Q | 5.2892 | 4.1142 | 3.0255 | 1.9927 | 1.4739 | 0.9866 | 0.5358 | 0.2928 | 0.1264 | 4.8129 |
| LGBM-PoisNB | 6.2369 | 5.0401 | 3.8408 | 2.5990 | 2.0061 | 1.3925 | 0.7740 | 0.4678 | 0.2710 | 4.8129 |
| MLP-NB | 5.7959 | 4.7278 | 3.6416 | 2.5152 | 1.9611 | 1.3913 | 0.7801 | 0.4192 | 0.2041 | 4.5176 |

Tồn kho cuối ngày trung bình (đơn vị/chuỗi):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 3.6958 | 4.3696 | 5.2030 | 6.3239 | 7.0984 | 8.1727 | 9.9524 | 11.6822 | 13.9041 | 4.3929 |
| SeasonalNaive | 4.5074 | 5.3081 | 6.4748 | 7.9924 | 9.0537 | 10.5362 | 13.0551 | 15.5673 | 18.8216 | 4.9798 |
| CrostonSBA | 3.7607 | 4.5064 | 5.4370 | 6.7021 | 7.5836 | 8.8042 | 10.8520 | 12.8600 | 15.4214 | 4.5915 |
| TSB | 3.8160 | 4.4840 | 5.2958 | 6.3853 | 7.1358 | 8.1647 | 9.8710 | 11.5313 | 13.6563 | 4.5556 |
| LGBM-Q | 4.3025 | 5.0226 | 5.8878 | 7.0631 | 8.0433 | 9.2526 | 11.0378 | 13.0822 | 15.6608 | 4.6087 |
| LGBM-PoisNB | 3.8672 | 4.5282 | 5.3527 | 6.4888 | 7.2821 | 8.3526 | 10.1306 | 11.8670 | 14.0716 | 4.6087 |
| MLP-NB | 4.1426 | 4.7580 | 5.5084 | 6.5066 | 7.2060 | 8.1956 | 9.9565 | 11.7738 | 14.2716 | 4.8282 |

Chi phí giữ TB/chuỗi:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 103.4836 | 122.3487 | 145.6841 | 177.0705 | 198.7558 | 228.8360 | 278.6671 | 327.1009 | 389.3159 | 123.0024 |
| SeasonalNaive | 126.2066 | 148.6258 | 181.2953 | 223.7861 | 253.5043 | 295.0146 | 365.5419 | 435.8858 | 527.0036 | 139.4338 |
| CrostonSBA | 105.3001 | 126.1798 | 152.2357 | 187.6586 | 212.3414 | 246.5188 | 303.8554 | 360.0802 | 431.7995 | 128.5614 |
| TSB | 106.8481 | 125.5516 | 148.2831 | 178.7898 | 199.8032 | 228.6112 | 276.3876 | 322.8761 | 382.3767 | 127.5577 |
| LGBM-Q | 120.4702 | 140.6330 | 164.8591 | 197.7667 | 225.2114 | 259.0729 | 309.0571 | 366.3026 | 438.5018 | 129.0450 |
| LGBM-PoisNB | 108.2819 | 126.7886 | 149.8761 | 181.6865 | 203.8979 | 233.8736 | 283.6574 | 332.2746 | 394.0049 | 129.0450 |
| MLP-NB | 115.9915 | 133.2248 | 154.2345 | 182.1835 | 201.7667 | 229.4763 | 278.7825 | 329.6659 | 399.6051 | 135.1896 |

Chi phí thiếu TB/chuỗi:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 208.6999 | 169.5869 | 131.1422 | 94.4228 | 75.6744 | 56.5978 | 37.3269 | 26.9016 | 19.1252 | 168.1166 |
| SeasonalNaive | 189.7084 | 159.5626 | 123.8518 | 91.5796 | 75.2126 | 59.1981 | 42.6245 | 32.9040 | 26.2576 | 172.1507 |
| CrostonSBA | 245.3220 | 201.5674 | 158.5662 | 113.5358 | 90.9721 | 67.1810 | 41.7497 | 27.1203 | 16.7193 | 197.3876 |
| TSB | 201.0936 | 163.2199 | 127.0352 | 90.5346 | 72.6367 | 53.8032 | 34.1798 | 23.0498 | 15.3706 | 162.9526 |
| LGBM-Q | 160.0729 | 123.6695 | 90.9964 | 60.8505 | 45.0182 | 31.5674 | 18.0316 | 10.0851 | 4.9818 | 147.0474 |
| LGBM-PoisNB | 185.2491 | 146.4763 | 109.8663 | 74.7509 | 58.1166 | 41.1908 | 24.7145 | 16.3670 | 10.4496 | 147.0474 |
| MLP-NB | 164.0705 | 131.3366 | 100.3402 | 70.2673 | 55.6865 | 40.3038 | 23.5601 | 13.8761 | 7.1932 | 131.8348 |

Tổng chi phí TB/chuỗi:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 312.1835 | 291.9356 | 276.8262 | 271.4933 | 274.4301 | 285.4338 | 315.9939 | 354.0024 | 408.4411 | 291.1191 |
| SeasonalNaive | 315.9149 | 308.1883 | 305.1470 | 315.3657 | 328.7169 | 354.2126 | 408.1665 | 468.7898 | 553.2612 | 311.5844 |
| CrostonSBA | 350.6221 | 327.7473 | 310.8019 | 301.1944 | 303.3135 | 313.6999 | 345.6051 | 387.2005 | 448.5188 | 325.9490 |
| TSB | 307.9417 | 288.7716 | 275.3183 | 269.3244 | 272.4399 | 282.4143 | 310.5674 | 345.9259 | 397.7473 | 290.5103 |
| LGBM-Q | 280.5431 | 264.3026 | 255.8554 | 258.6173 | 270.2296 | 290.6403 | 327.0887 | 376.3876 | 443.4836 | 276.0923 |
| LGBM-PoisNB | 293.5310 | 273.2649 | 259.7424 | 256.4374 | 262.0146 | 275.0644 | 308.3718 | 348.6416 | 404.4544 | 276.0923 |
| MLP-NB | 280.0620 | 264.5614 | 254.5747 | 252.4508 | 257.4532 | 269.7801 | 302.3426 | 343.5419 | 406.7983 | 267.0243 |

q* tối thiểu hóa tổng chi phí (so với chính sách điểm):
| index | q* (min total_cost) | cost_q* | fill_q* | avg_inv_q* | cost_point | fill_point | cost_q*/cost_point - 1 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.8000 | 271.4933 | 0.8740 | 6.3239 | 291.1191 | 0.7756 | -0.0674 |
| SeasonalNaive | 0.7000 | 305.1470 | 0.8347 | 6.4748 | 311.5844 | 0.7702 | -0.0207 |
| CrostonSBA | 0.8000 | 301.1944 | 0.8484 | 6.7021 | 325.9490 | 0.7365 | -0.0759 |
| TSB | 0.8000 | 269.3244 | 0.8791 | 6.3853 | 290.5103 | 0.7825 | -0.0729 |
| LGBM-Q | 0.7000 | 255.8554 | 0.8785 | 5.8878 | 276.0923 | 0.8037 | -0.0733 |
| LGBM-PoisNB | 0.8000 | 256.4374 | 0.9002 | 6.4888 | 276.0923 | 0.8037 | -0.0712 |
| MLP-NB | 0.8000 | 252.4508 | 0.9062 | 6.5066 | 267.0243 | 0.8240 | -0.0546 |

### L=3 (H=4), cost_ratio=10, overall (823 chuỗi)
Fill rate (tổng bán / tổng cầu):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.7440 | 0.7938 | 0.8408 | 0.8865 | 0.9085 | 0.9302 | 0.9529 | 0.9656 | 0.9747 | 0.7877 |
| SeasonalNaive | 0.7677 | 0.8061 | 0.8488 | 0.8881 | 0.9077 | 0.9267 | 0.9473 | 0.9590 | 0.9675 | 0.7877 |
| CrostonSBA | 0.6949 | 0.7515 | 0.8069 | 0.8625 | 0.8907 | 0.9192 | 0.9492 | 0.9665 | 0.9789 | 0.7500 |
| TSB | 0.7554 | 0.8025 | 0.8473 | 0.8919 | 0.9136 | 0.9353 | 0.9582 | 0.9708 | 0.9797 | 0.7973 |
| LGBM-Q | 0.7975 | 0.8405 | 0.8806 | 0.9205 | 0.9403 | 0.9603 | 0.9767 | 0.9863 | 0.9933 | 0.8112 |
| LGBM-PoisNB | 0.7694 | 0.8167 | 0.8632 | 0.9060 | 0.9266 | 0.9468 | 0.9667 | 0.9776 | 0.9849 | 0.8112 |
| MLP-NB | 0.7922 | 0.8324 | 0.8695 | 0.9056 | 0.9237 | 0.9431 | 0.9646 | 0.9781 | 0.9878 | 0.8276 |

Số ngày hết hàng TB/chuỗi (trên 28 ngày):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 6.1531 | 4.9866 | 3.9478 | 2.8226 | 2.2479 | 1.6780 | 1.0790 | 0.7072 | 0.4885 | 4.9332 |
| SeasonalNaive | 5.4957 | 4.6610 | 3.6112 | 2.6610 | 2.1640 | 1.7363 | 1.2017 | 0.8955 | 0.7047 | 5.0255 |
| CrostonSBA | 6.9988 | 5.7801 | 4.6075 | 3.3937 | 2.7582 | 2.0486 | 1.2637 | 0.8190 | 0.5091 | 5.5504 |
| TSB | 5.7132 | 4.6075 | 3.5808 | 2.5577 | 2.0595 | 1.5298 | 0.9453 | 0.6233 | 0.3985 | 4.5152 |
| LGBM-Q | 4.8226 | 3.8335 | 2.8736 | 1.8736 | 1.4046 | 0.8967 | 0.4763 | 0.2661 | 0.1252 | 4.4957 |
| LGBM-PoisNB | 5.6938 | 4.5674 | 3.4557 | 2.3791 | 1.8153 | 1.2953 | 0.7497 | 0.4617 | 0.2673 | 4.4957 |
| MLP-NB | 5.3706 | 4.3366 | 3.4241 | 2.4180 | 1.9064 | 1.3876 | 0.7728 | 0.4593 | 0.2224 | 4.2661 |

Tồn kho cuối ngày trung bình (đơn vị/chuỗi):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 4.3676 | 5.1467 | 6.0923 | 7.4032 | 8.3197 | 9.5903 | 11.7010 | 13.7392 | 16.3699 | 5.0322 |
| SeasonalNaive | 5.1245 | 6.0265 | 7.2845 | 8.9481 | 10.0980 | 11.7270 | 14.4997 | 17.2501 | 20.8286 | 5.6076 |
| CrostonSBA | 4.5463 | 5.4091 | 6.5109 | 8.0181 | 9.0688 | 10.5430 | 13.0371 | 15.4446 | 18.5468 | 5.3923 |
| TSB | 4.5371 | 5.2839 | 6.2247 | 7.4793 | 8.3443 | 9.5536 | 11.5532 | 13.5038 | 16.0098 | 5.2646 |
| LGBM-Q | 4.9131 | 5.6505 | 6.6030 | 7.9858 | 9.0777 | 10.5374 | 12.7300 | 14.8580 | 17.8308 | 5.1961 |
| LGBM-PoisNB | 4.4711 | 5.2139 | 6.1440 | 7.4436 | 8.3266 | 9.5652 | 11.6063 | 13.5857 | 16.0843 | 5.1961 |
| MLP-NB | 4.6869 | 5.3535 | 6.1531 | 7.2364 | 7.9937 | 9.0831 | 11.0657 | 13.1636 | 16.1581 | 5.3586 |

Chi phí giữ TB/chuỗi:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 122.2940 | 144.1081 | 170.5844 | 207.2904 | 232.9526 | 268.5298 | 327.6270 | 384.6962 | 458.3584 | 140.9004 |
| SeasonalNaive | 143.4848 | 168.7412 | 203.9660 | 250.5480 | 282.7448 | 328.3572 | 405.9903 | 483.0024 | 583.2017 | 157.0134 |
| CrostonSBA | 127.2965 | 151.4557 | 182.3050 | 224.5055 | 253.9271 | 295.2029 | 365.0377 | 432.4496 | 519.3111 | 150.9842 |
| TSB | 127.0389 | 147.9502 | 174.2928 | 209.4192 | 233.6403 | 267.5018 | 323.4885 | 378.1069 | 448.2746 | 147.4083 |
| LGBM-Q | 137.5674 | 158.2151 | 184.8846 | 223.6027 | 254.1762 | 295.0462 | 356.4386 | 416.0243 | 499.2625 | 145.4897 |
| LGBM-PoisNB | 125.1896 | 145.9878 | 172.0316 | 208.4204 | 233.1446 | 267.8262 | 324.9757 | 380.3998 | 450.3597 | 145.4897 |
| MLP-NB | 131.2321 | 149.8979 | 172.2868 | 202.6197 | 223.8238 | 254.3269 | 309.8396 | 368.5796 | 452.4265 | 150.0413 |

Chi phí thiếu TB/chuỗi:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 191.7740 | 154.4957 | 119.2831 | 85.0304 | 68.5541 | 52.3208 | 35.2977 | 25.7351 | 18.9186 | 159.0401 |
| SeasonalNaive | 174.0340 | 145.2491 | 113.2807 | 83.7910 | 69.1130 | 54.9332 | 39.4775 | 30.7047 | 24.3742 | 159.0401 |
| CrostonSBA | 228.5298 | 186.1725 | 144.6416 | 103.0377 | 81.8591 | 60.4982 | 38.0923 | 25.0790 | 15.8202 | 187.2783 |
| TSB | 183.2199 | 147.9222 | 114.3621 | 80.9599 | 64.7388 | 48.4933 | 31.2880 | 21.9077 | 15.2369 | 151.8469 |
| LGBM-Q | 151.6889 | 119.5018 | 89.4532 | 59.5747 | 44.6902 | 29.7327 | 17.4605 | 10.2552 | 4.9939 | 141.4581 |
| LGBM-PoisNB | 172.7096 | 137.2783 | 102.4666 | 70.4010 | 54.9939 | 39.8663 | 24.9210 | 16.8044 | 11.3123 | 141.4581 |
| MLP-NB | 155.6379 | 125.5286 | 97.7400 | 70.7412 | 57.1689 | 42.6245 | 26.4885 | 16.3791 | 9.1616 | 129.1373 |

Tổng chi phí TB/chuỗi:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 314.0680 | 298.6039 | 289.8676 | 292.3208 | 301.5067 | 320.8505 | 362.9247 | 410.4313 | 477.2770 | 299.9405 |
| SeasonalNaive | 317.5188 | 313.9903 | 317.2467 | 334.3390 | 351.8578 | 383.2904 | 445.4678 | 513.7072 | 607.5759 | 316.0535 |
| CrostonSBA | 355.8262 | 337.6282 | 326.9465 | 327.5431 | 335.7861 | 355.7011 | 403.1300 | 457.5286 | 535.1312 | 338.2625 |
| TSB | 310.2588 | 295.8724 | 288.6549 | 290.3791 | 298.3791 | 315.9951 | 354.7764 | 400.0146 | 463.5115 | 299.2552 |
| LGBM-Q | 289.2564 | 277.7169 | 274.3378 | 283.1774 | 298.8663 | 324.7789 | 373.8991 | 426.2795 | 504.2564 | 286.9478 |
| LGBM-PoisNB | 297.8991 | 283.2661 | 274.4982 | 278.8214 | 288.1385 | 307.6926 | 349.8967 | 397.2041 | 461.6719 | 286.9478 |
| MLP-NB | 286.8700 | 275.4265 | 270.0267 | 273.3609 | 280.9927 | 296.9514 | 336.3281 | 384.9587 | 461.5881 | 279.1786 |

q* tối thiểu hóa tổng chi phí (so với chính sách điểm):
| index | q* (min total_cost) | cost_q* | fill_q* | avg_inv_q* | cost_point | fill_point | cost_q*/cost_point - 1 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.7000 | 289.8676 | 0.8408 | 6.0923 | 299.9405 | 0.7877 | -0.0336 |
| SeasonalNaive | 0.6000 | 313.9903 | 0.8061 | 6.0265 | 316.0535 | 0.7877 | -0.0065 |
| CrostonSBA | 0.7000 | 326.9465 | 0.8069 | 6.5109 | 338.2625 | 0.7500 | -0.0335 |
| TSB | 0.7000 | 288.6549 | 0.8473 | 6.2247 | 299.2552 | 0.7973 | -0.0354 |
| LGBM-Q | 0.7000 | 274.3378 | 0.8806 | 6.6030 | 286.9478 | 0.8112 | -0.0439 |
| LGBM-PoisNB | 0.7000 | 274.4982 | 0.8632 | 6.1440 | 286.9478 | 0.8112 | -0.0434 |
| MLP-NB | 0.7000 | 270.0267 | 0.8695 | 6.1531 | 279.1786 | 0.8276 | -0.0328 |


### Tách nhanh/chậm (L=2)
L=2, nhóm speed=fast, tổng chi phí TB/chuỗi (cost_ratio=10; cột = mức q của chính sách phân vị, cột 'point' = chính sách điểm S=ceil(mean)):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 543.5767 | 508.2423 | 483.3589 | 474.6994 | 480.0337 | 498.9202 | 550.7055 | 613.5920 | 699.8773 | 515.1472 |
| SeasonalNaive | 555.0123 | 535.3160 | 527.1043 | 537.1902 | 553.0982 | 586.0828 | 660.6196 | 743.1012 | 858.5675 | 544.5092 |
| CrostonSBA | 616.3098 | 576.7638 | 547.0460 | 531.9632 | 536.0184 | 556.3190 | 615.8466 | 689.9448 | 798.5000 | 581.7362 |
| TSB | 542.7853 | 509.7147 | 485.4663 | 473.6626 | 476.6104 | 491.5092 | 533.6994 | 588.2301 | 666.1319 | 516.9724 |
| LGBM-Q | 488.6350 | 459.6840 | 443.1166 | 445.5031 | 465.0123 | 499.3926 | 554.8098 | 637.7853 | 743.2301 | 481.3160 |
| LGBM-PoisNB | 505.3865 | 471.9018 | 450.0245 | 445.4509 | 454.6687 | 475.7147 | 529.4509 | 592.8282 | 678.9908 | 481.3160 |
| MLP-NB | 480.4908 | 455.8773 | 441.1380 | 438.7822 | 447.8221 | 468.2301 | 521.6196 | 589.8773 | 695.1840 | 463.3988 |

L=2, nhóm speed=fast, fill rate (cost_ratio không ảnh hưởng fill rate; cột = q, 'point' = chính sách điểm):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.7616 | 0.8082 | 0.8524 | 0.8946 | 0.9158 | 0.9369 | 0.9577 | 0.9685 | 0.9773 | 0.7982 |
| SeasonalNaive | 0.7744 | 0.8146 | 0.8532 | 0.8907 | 0.9103 | 0.9299 | 0.9491 | 0.9607 | 0.9696 | 0.7945 |
| CrostonSBA | 0.7161 | 0.7690 | 0.8210 | 0.8741 | 0.9010 | 0.9283 | 0.9564 | 0.9725 | 0.9835 | 0.7617 |
| TSB | 0.7649 | 0.8089 | 0.8512 | 0.8935 | 0.9146 | 0.9362 | 0.9590 | 0.9717 | 0.9809 | 0.7994 |
| LGBM-Q | 0.8128 | 0.8549 | 0.8926 | 0.9288 | 0.9476 | 0.9622 | 0.9782 | 0.9878 | 0.9937 | 0.8270 |
| LGBM-PoisNB | 0.7937 | 0.8378 | 0.8790 | 0.9172 | 0.9350 | 0.9533 | 0.9710 | 0.9797 | 0.9861 | 0.8270 |
| MLP-NB | 0.8210 | 0.8575 | 0.8913 | 0.9236 | 0.9389 | 0.9552 | 0.9730 | 0.9834 | 0.9908 | 0.8481 |

L=2, nhóm speed=slow, tổng chi phí TB/chuỗi (cost_ratio=10; cột = mức q của chính sách phân vị, cột 'point' = chính sách điểm S=ceil(mean)):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 102.1307 | 94.4318 | 89.0625 | 87.1420 | 88.4034 | 93.1136 | 105.5000 | 122.3352 | 148.2159 | 89.4432 |
| SeasonalNaive | 98.2841 | 99.3693 | 102.7159 | 111.1989 | 120.8011 | 137.4261 | 166.9545 | 200.6477 | 246.9489 | 99.0795 |
| CrostonSBA | 111.4148 | 104.2557 | 98.9261 | 94.8693 | 95.7557 | 99.0227 | 109.7784 | 125.2273 | 147.6932 | 97.7045 |
| TSB | 96.2898 | 90.2386 | 86.2102 | 86.1989 | 89.3409 | 95.5284 | 111.8068 | 130.4261 | 158.7784 | 88.8295 |
| LGBM-Q | 92.4205 | 89.2784 | 87.4489 | 90.9205 | 96.1250 | 104.8807 | 125.6705 | 147.8636 | 180.6932 | 91.3580 |
| LGBM-PoisNB | 99.1136 | 91.7841 | 87.5284 | 85.4205 | 89.4830 | 95.6250 | 110.1364 | 129.5455 | 157.1534 | 91.3580 |
| MLP-NB | 95.4830 | 89.2955 | 85.2898 | 84.6023 | 86.3409 | 91.8750 | 107.3409 | 125.6705 | 152.6534 | 89.1023 |

L=2, nhóm speed=slow, fill rate (cost_ratio không ảnh hưởng fill rate; cột = q, 'point' = chính sách điểm):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 | point(ceil mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.4935 | 0.5872 | 0.6802 | 0.7688 | 0.8155 | 0.8647 | 0.9210 | 0.9476 | 0.9641 | 0.6716 |
| SeasonalNaive | 0.6116 | 0.6497 | 0.7376 | 0.8108 | 0.8439 | 0.8719 | 0.9042 | 0.9225 | 0.9286 | 0.6508 |
| CrostonSBA | 0.4300 | 0.5197 | 0.6167 | 0.7182 | 0.7717 | 0.8295 | 0.8952 | 0.9296 | 0.9573 | 0.6271 |
| TSB | 0.5546 | 0.6425 | 0.7297 | 0.8180 | 0.8536 | 0.8999 | 0.9408 | 0.9641 | 0.9767 | 0.7182 |
| LGBM-Q | 0.6597 | 0.7365 | 0.8173 | 0.8769 | 0.9103 | 0.9462 | 0.9727 | 0.9867 | 0.9961 | 0.6978 |
| LGBM-PoisNB | 0.5319 | 0.6256 | 0.7139 | 0.8173 | 0.8622 | 0.9092 | 0.9533 | 0.9749 | 0.9910 | 0.6978 |
| MLP-NB | 0.5675 | 0.6508 | 0.7326 | 0.8169 | 0.8611 | 0.9052 | 0.9501 | 0.9745 | 0.9899 | 0.7175 |


(Các nhóm medium, demand_type, demand_type×speed và mọi cost_ratio: xem CSV; phân tích q* ở GĐ4.)

## 5. File sinh ra
`results/inventory_sim_val.csv` (cột: model, policy ∈ {quantile, point}, q, L, H, group_type, group, n_series, cost_ratio, fill_rate, stockout_days, avg_inventory, lost_units, holding_cost, stockout_cost, total_cost).

## 6. Cấu hình
Seed 42 (mô phỏng xác định, không có bước ngẫu nhiên; test tái lập so sánh hai lần chạy cùng đầu vào).

## 7. Vấn đề / giả định chưa chắc chắn
- Chỉ 28 ngày mô phỏng, tồn đầu = S ở gốc đầu tiên nên giai đoạn khởi động chiếm phần lớn; chi phí giữ/tồn kho phụ thuộc vào quy tắc khởi tạo này. Ngày cuối không có hàng về mới (đặt hàng cuối cùng ở gốc 28−H) nên tồn cuối giảm.
- Fill rate thấp hơn mức q danh nghĩa (q là mức phục vụ theo chu kỳ P(D_H ≤ S), còn fill rate tính theo khối lượng; ngoài ra phân vị rời rạc và đặt hàng cuối ngày với nhu cầu có tương quan chuỗi làm hai khái niệm lệch nhau).
- Các chỉ số gộp theo nhóm là trung bình theo chuỗi nên hàng nhanh (nhu cầu lớn) chi phối chi phí tuyệt đối; đọc so sánh giữa mô hình trong cùng nhóm, không so giữa nhóm.
- Không kiểm định thống kê; 823 chuỗi × 1 lần mô phỏng, không lặp (nhu cầu là thực nên không có ngẫu nhiên).

## 8. Chưa làm
Quét cost_ratio × L, tìm q*, biểu đồ, bảng chính sách theo nhóm và trả lời câu hỏi nghiên cứu: GĐ4 (CSV đã chứa mọi cost_ratio nên GĐ4 chỉ cần phân tích).
