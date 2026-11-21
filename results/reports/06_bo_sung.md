# Báo cáo GĐ6 – Bổ sung trước khi chốt đồ án

## 1. Đã làm gì (file tạo/sửa)
- **Mới**: `src/safety_point.py` (chính sách điểm có hệ số an toàn + q* lưới mở rộng + bootstrap), `src/calibration.py` (bảng hiệu chuẩn rời rạc), `src/ablation.py` (bỏ biến ngoại sinh), `src/mlp_seeds.py` (3 seed + hội tụ), `src/gd6_report_tables.py`, `tests/test_gd6.py` (7 test). Kết quả: `results/safety_point_test.csv`, `q_star_ext_val.csv`, `inventory_sim_{val,test}_ext.csv`, `calibration_table.csv`, `calibration_summary.csv`, `ablation_exog.csv`, `mlp_seeds.csv`, `mlp_nll_curves.csv`, hình `results/figures/mlp_nll_curves.png`.
- **Sửa (không đổi kết quả chính)**: `src/ml_models.py` (`fit_lgbm(extra_q=...)`, `fit_mlp_nb(seed=, max_epochs=)`, `Qext`), `src/prob_models.py` (`EXT_QGRID`, khóa `Qext`), `src/run_forecasts.py` (lưu `Qext` vào npz), `src/inventory_sim.py` (`--ext`), `configs/config.yaml` (`quantile_grid_ext`, `safety_k_grid`, `mlp_seeds`, `mlp_convergence_epochs`).
- **Chạy lại** `run_forecasts --split val/test` để sinh `Qext`: `hparams.json` GIỐNG HỆT (sha256 `c26d657819002fb0`), `forecast_eval_{val,test}_all.csv` khác bản trước tối đa 1.1e-16 (sai số dấu phẩy động), `inventory_sim_{val,test}.csv` giống hệt (so với bản lưu ở `archive/pre_gd6/`). Vì vậy mọi kết quả ở báo cáo 02–05 giữ nguyên.
- **Mục 1 (yêu cầu dán pytest vào 05)**: `05_final_test.md` đã có output `pytest -v` ở mục 2 từ GĐ5 (34 test); đã thay bằng output mới nhất ở cuối GĐ6 (mục 2 dưới đây là cùng output).

## 2. Lệnh đã chạy và pytest
```
python -m src.run_forecasts --split val ; python -m src.run_forecasts --split test
python -m src.inventory_sim --split val --ext ; python -m src.inventory_sim --split test --ext
python -m src.safety_point ; python -m src.calibration ; python -m src.ablation ; python -m src.mlp_seeds
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

## 3. Câu hỏi 1 – chính sách điểm có hệ số an toàn (mục 2 của yêu cầu)
**Thiết kế.** Chính sách điểm an toàn: S = ceil(k × mean cửa sổ của cùng mô hình), k ∈ {0.5, 0.6, …, 3.0} (26 giá trị). Với từng (mô hình, tỷ lệ chi phí, L): k chọn trên VAL (k làm tối thiểu tổng chi phí TB/chuỗi tổng thể trên val; hòa → k nhỏ nhất); q* cũng chọn trên val trên lưới q mở rộng {0.3, 0.4, 0.5, …, 0.99}; cả hai đánh giá trên TEST. "Điểm thuần" là k=1. Các hiệu là chi phí tổng TB/chuỗi (28 ngày, đơn vị tương đối, chênh ÂM = vế đầu rẻ hơn), KTC 95% bootstrap theo chuỗi (2000 lần, seed 42, ghép cặp). Nhóm: toàn bộ 823 chuỗi.
k chọn trên val, L=2 (cột = tỷ lệ chi phí thiếu/giữ):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5000 | 0.5000 | 1.0000 | 1.3000 | 1.5000 | 1.9000 |
| SeasonalNaive | 0.5000 | 0.5000 | 0.8000 | 1.1000 | 1.4000 | 1.7000 |
| CrostonSBA | 0.5000 | 0.5000 | 1.0000 | 1.3000 | 1.7000 | 2.1000 |
| TSB | 0.5000 | 0.5000 | 1.0000 | 1.3000 | 1.5000 | 1.9000 |
| LGBM-Q | 0.5000 | 0.5000 | 1.0000 | 1.2000 | 1.5000 | 1.8000 |
| LGBM-PoisNB | 0.5000 | 0.5000 | 1.0000 | 1.2000 | 1.5000 | 1.8000 |
| MLP-NB | 0.5000 | 0.5000 | 0.9000 | 1.2000 | 1.4000 | 1.7000 |

q* chọn trên val (lưới mở rộng), L=2:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | 0.3000 | 0.3000 | 0.6000 | 0.8000 | 0.9000 | 0.9750 |
| SeasonalNaive | 0.3000 | 0.3000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |
| CrostonSBA | 0.3000 | 0.3000 | 0.6000 | 0.8000 | 0.9000 | 0.9750 |
| TSB | 0.3000 | 0.3000 | 0.6000 | 0.8000 | 0.9000 | 0.9750 |
| LGBM-Q | 0.3000 | 0.3000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |
| LGBM-PoisNB | 0.3000 | 0.3000 | 0.6000 | 0.8000 | 0.9000 | 0.9500 |
| MLP-NB | 0.3000 | 0.3000 | 0.5000 | 0.8000 | 0.9000 | 0.9500 |

Test, L=2, tỷ lệ 10 – chi phí TB/chuỗi:
| model | k_val | q_star_val | cost_val_point | cost_val_k | cost_val_q | cost_test_point | cost_test_k | cost_test_q |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 1.3000 | 0.8000 | 291.1191 | 272.0267 | 271.4933 | 306.8226 | 283.4569 | 283.2892 |
| SeasonalNaive | 1.1000 | 0.7000 | 311.5844 | 301.9174 | 305.1470 | 320.7995 | 305.6343 | 308.2199 |
| CrostonSBA | 1.3000 | 0.8000 | 325.9490 | 300.7631 | 301.1944 | 335.2916 | 304.2211 | 304.2625 |
| TSB | 1.3000 | 0.8000 | 290.5103 | 270.5614 | 269.3244 | 301.0486 | 278.6160 | 277.9247 |
| LGBM-Q | 1.2000 | 0.7000 | 276.0923 | 257.7740 | 255.8554 | 283.9842 | 261.2734 | 261.4253 |
| LGBM-PoisNB | 1.2000 | 0.8000 | 276.0923 | 257.7740 | 256.4374 | 283.9842 | 261.2734 | 259.9672 |
| MLP-NB | 1.2000 | 0.8000 | 267.0243 | 255.9186 | 252.4508 | 275.8068 | 260.6087 | 258.5030 |

**Test, L=2: chi phí[điểm an toàn k] − chi phí[phân vị q*] [KTC 95%]** (dương = phân vị rẻ hơn):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -7.48 [-9.26; -5.81] | 2.46 [0.91; 4.08] | -0.09 [-1.22; 0.84] | 0.17 [-0.97; 1.40] | 10.04 [5.80; 14.82] | 36.70 [26.27; 48.45] |
| SeasonalNaive | -14.61 [-16.59; -12.68] | -2.92 [-4.80; -0.91] | 5.00 [2.65; 7.63] | -2.59 [-3.81; -1.34] | -3.29 [-6.54; 0.06] | -5.36 [-18.21; 5.58] |
| CrostonSBA | -4.91 [-6.49; -3.49] | 2.69 [1.26; 4.21] | 0.04 [-0.58; 0.65] | -0.04 [-1.42; 1.30] | 6.28 [2.08; 10.53] | 43.35 [30.88; 57.46] |
| TSB | -8.10 [-10.01; -6.35] | 2.74 [1.20; 4.26] | 0.88 [0.01; 1.64] | 0.69 [-0.48; 1.87] | 6.57 [3.05; 10.39] | 19.73 [9.10; 30.95] |
| LGBM-Q | -13.09 [-14.71; -11.50] | 1.54 [-0.31; 3.56] | 1.63 [0.64; 2.69] | -0.15 [-2.14; 1.76] | 3.43 [-1.23; 8.68] | 23.95 [12.89; 35.72] |
| LGBM-PoisNB | -6.79 [-8.55; -5.14] | 4.71 [2.92; 6.60] | 1.69 [0.63; 2.49] | 1.31 [-0.91; 3.78] | 5.17 [0.83; 10.16] | 21.00 [9.88; 33.04] |
| MLP-NB | -10.98 [-13.11; -8.91] | 3.08 [1.13; 5.14] | 3.79 [2.73; 4.88] | 2.11 [0.68; 3.77] | 8.67 [5.26; 12.60] | 26.52 [16.68; 37.17] |

L=3:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -15.56 [-18.11; -13.21] | -3.02 [-4.79; -1.23] | 0.54 [-0.82; 1.72] | -3.54 [-6.00; -1.51] | 4.53 [1.30; 8.04] | 36.89 [26.11; 48.77] |
| SeasonalNaive | -22.03 [-24.67; -19.38] | -8.20 [-10.22; -6.33] | 2.87 [1.14; 4.77] | -4.49 [-6.22; -2.91] | -4.11 [-6.77; -1.47] | -10.09 [-21.12; 0.97] |
| CrostonSBA | -11.44 [-13.78; -9.33] | -1.72 [-3.31; -0.11] | 1.01 [0.20; 1.79] | -4.27 [-5.40; -3.16] | 5.96 [0.89; 11.27] | 37.30 [24.24; 51.07] |
| TSB | -16.81 [-19.54; -14.24] | -3.58 [-5.25; -1.94] | 1.51 [0.36; 2.54] | -1.29 [-3.04; 0.38] | 1.95 [-1.29; 5.28] | 22.30 [12.40; 33.07] |
| LGBM-Q | -20.39 [-22.70; -18.14] | -4.20 [-6.02; -2.29] | 2.95 [1.72; 4.12] | -0.51 [-2.78; 1.54] | -2.35 [-7.71; 3.36] | 4.81 [-10.72; 20.97] |
| LGBM-PoisNB | -14.14 [-16.51; -11.85] | -0.52 [-2.41; 1.38] | 3.54 [2.17; 4.88] | -0.51 [-2.80; 1.36] | 2.35 [-1.71; 6.76] | 22.41 [7.95; 36.45] |
| MLP-NB | -17.43 [-20.09; -14.80] | -1.47 [-3.61; 0.61] | 2.90 [1.72; 4.15] | 2.06 [0.94; 3.33] | 6.39 [3.03; 10.02] | 25.81 [13.48; 39.36] |

Test, L=2: phân vị q* − điểm thuần (k=1):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -37.02 [-39.11; -35.01] | -22.97 [-24.56; -21.36] | 0.09 [-0.84; 1.22] | -23.53 [-29.59; -18.08] | -134.36 [-156.95; -115.18] | -561.10 [-638.70; -493.37] |
| SeasonalNaive | -41.59 [-43.93; -39.31] | -29.65 [-31.28; -27.97] | -3.15 [-4.00; -2.17] | -12.58 [-16.05; -9.36] | -96.14 [-109.83; -83.21] | -466.20 [-517.28; -416.74] |
| CrostonSBA | -39.61 [-42.67; -36.77] | -24.00 [-26.69; -21.53] | -0.04 [-0.65; 0.58] | -31.03 [-37.91; -24.73] | -160.06 [-184.21; -138.80] | -663.59 [-746.25; -590.66] |
| TSB | -38.56 [-40.58; -36.64] | -25.04 [-26.60; -23.47] | -0.88 [-1.64; -0.01] | -23.12 [-28.94; -17.98] | -128.64 [-149.80; -109.80] | -540.85 [-619.55; -471.12] |
| LGBM-Q | -31.32 [-32.97; -29.72] | -20.44 [-21.70; -19.16] | -1.63 [-2.69; -0.64] | -22.56 [-26.91; -18.73] | -120.20 [-136.87; -105.49] | -502.03 [-562.55; -449.19] |
| LGBM-PoisNB | -37.61 [-39.35; -36.04] | -23.61 [-24.92; -22.35] | -1.69 [-2.49; -0.63] | -24.02 [-29.80; -18.90] | -121.94 [-140.98; -105.48] | -499.08 [-561.94; -446.12] |
| MLP-NB | -36.80 [-38.60; -35.14] | -25.16 [-26.64; -23.72] | -2.00 [-2.88; -1.10] | -17.30 [-21.74; -13.44] | -98.63 [-113.73; -85.35] | -429.60 [-479.66; -385.23] |

Test, L=2: điểm an toàn k − điểm thuần (k=1):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -44.49 [-48.06; -41.06] | -20.51 [-22.82; -18.26] | 0.00 [0.00; 0.00] | -23.37 [-28.97; -18.16] | -124.32 [-144.45; -106.35] | -524.40 [-597.66; -460.19] |
| SeasonalNaive | -56.19 [-59.91; -52.49] | -32.57 [-35.34; -29.91] | 1.85 [-0.83; 4.88] | -15.17 [-18.03; -12.64] | -99.42 [-113.22; -86.26] | -471.56 [-524.87; -423.04] |
| CrostonSBA | -44.51 [-48.80; -40.51] | -21.31 [-24.81; -18.14] | 0.00 [0.00; 0.00] | -31.07 [-37.36; -25.43] | -153.78 [-178.16; -133.22] | -620.24 [-699.65; -549.85] |
| TSB | -46.65 [-50.35; -43.07] | -22.30 [-24.57; -20.10] | 0.00 [0.00; 0.00] | -22.43 [-27.96; -17.12] | -122.07 [-142.25; -104.31] | -521.12 [-594.69; -456.33] |
| LGBM-Q | -44.40 [-47.43; -41.44] | -18.90 [-21.18; -16.57] | 0.00 [0.00; 0.00] | -22.71 [-27.18; -18.78] | -116.77 [-133.84; -101.17] | -478.08 [-536.20; -426.83] |
| LGBM-PoisNB | -44.40 [-47.43; -41.44] | -18.90 [-21.18; -16.57] | 0.00 [0.00; 0.00] | -22.71 [-27.18; -18.78] | -116.77 [-133.84; -101.17] | -478.08 [-536.20; -426.83] |
| MLP-NB | -47.78 [-51.37; -44.34] | -22.08 [-24.36; -19.81] | 1.79 [0.61; 3.15] | -15.20 [-18.74; -11.98] | -89.96 [-103.57; -78.35] | -403.09 [-452.87; -359.89] |

(L=3: `results/safety_point_test.csv`.) Đếm ô (7 mô hình × 6 tỷ lệ = 42 ô mỗi L) theo vị trí KTC so với 0:
| L | so_sanh | KTC<0 (vế đầu rẻ hơn) | KTC chứa 0 | KTC>0 (vế đầu đắt hơn) | tong_o |
|---|---|---|---|---|---|
| 2 | k_minus_q | 9 | 11 | 22 | 42 |
| 2 | q_minus_point | 40 | 2 | 0 | 42 |
| 2 | k_minus_point | 35 | 6 | 1 | 42 |
| 3 | k_minus_q | 16 | 11 | 15 | 42 |
| 3 | q_minus_point | 40 | 2 | 0 | 42 |
| 3 | k_minus_point | 37 | 5 | 0 | 42 |

k_minus_q = điểm an toàn − phân vị; q_minus_point = phân vị − điểm thuần; k_minus_point = điểm an toàn − điểm thuần.

**Kết luận mới cho câu 1 (thay thế kết luận ở 04/05).**
- Phần lớn "lợi thế của phân vị" trong GĐ4–5 là lợi thế của việc CÓ hệ số an toàn: so với điểm thuần, điểm an toàn rẻ hơn rõ ở 35/42 ô (L=2) và 37/42 ô (L=3) (KTC dưới 0), gần bằng mức rẻ hơn của phân vị (40/42 ở cả hai L). Chênh lệch giữa điểm an toàn và phân vị đã tinh chỉnh nhỏ ở tỷ lệ 5–10: L=2, tỷ lệ 10, KTC chứa 0 ở 5/7 mô hình (WindowAverage 0.17 [−0.97; 1.40], CrostonSBA −0.04 [−1.42; 1.30], TSB 0.69 [−0.48; 1.87], LGBM-Q −0.15 [−2.14; 1.76], LGBM-PoisNB 1.31 [−0.91; 3.78]); phân vị chỉ rẻ hơn nhẹ ở MLP-NB (2.11 [0.68; 3.77], ≈0.8% chi phí) và điểm an toàn rẻ hơn ở SeasonalNaive (−2.59 [−3.81; −1.34]).
- **Lợi thế của phân vị còn lại ở tỷ lệ chi phí cao**: tỷ lệ 50 (L=2) điểm an toàn đắt hơn phân vị 19.73 … 43.35 đơn vị ở 6/7 mô hình (KTC dương; ví dụ CrostonSBA 43.35 [30.88; 57.46], WindowAverage 36.70 [26.27; 48.45]); tỷ lệ 20: dương ở 5/7 (L=2). Hợp lý vì ở đuôi cần hình dạng phân phối (độ phân tán theo từng chuỗi) mà một hệ số nhân chung k không biểu diễn được.
- **Tỷ lệ thấp (1–2)**: điểm an toàn thường rẻ hơn (tỷ lệ 1: âm ở cả 7 mô hình, ví dụ SeasonalNaive −14.61 [−16.59; −12.68]), nhưng ở tỷ lệ 1–2 cả k*=0.5 (cận dưới lưới k) lẫn q*=0.3 (cận dưới lưới q) đều chạm biên nên so sánh này bị chặn bởi lưới ở cả hai phía, không nên kết luận chắc.
- Tóm lại: "phân vị tốt hơn điểm" chỉ đúng khi so với điểm THUẦN (k=1). Khi điểm được tinh chỉnh hệ số an toàn, phân vị hầu như ngang ở tỷ lệ chi phí trung bình (5–10) và hơn rõ chỉ ở tỷ lệ cao (≥20). Kết luận GĐ4–5 ("phân vị rẻ hơn điểm ở mọi tỷ lệ") phải được hiểu là so với điểm thuần.
- Hạn chế: k và q* cùng chọn trên val (cùng thiên vị lạc quan); một k chung cho mọi chuỗi (không theo nhóm); 28 ngày; KTC coi chuỗi độc lập.

## 4. Mở rộng lưới q xuống 0.3, 0.4 (mục 3)
Phạm vi: **tất cả 7 mô hình** được mở rộng (không phải chỉ NB): bốn mô hình thống kê, LGBM-PoisNB và MLP-NB dùng cùng mean/alpha đã chốt nên chỉ tính thêm phân vị NB; LGBM-Q huấn luyện thêm 2 mô hình phân vị (0.3 và 0.4; thêm ≈10 giây mỗi H). Lưới tuning/đánh giá chính (9 mức, 0.5–0.99) giữ nguyên để không đổi các kết quả cũ; lưới mở rộng chỉ dùng cho mô phỏng (`inventory_sim_*_ext.csv`). Phân vị LGBM-Q phụ được ép không vượt phân vị q=0.5 và tăng dần.
q* trên val (L=2), tỷ lệ 1–5, lưới chính (main) và lưới mở rộng (ext):
| model | main_r1 | main_r2 | main_r5 | ext_r1 | ext_r2 | ext_r5 |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5000 | 0.5000 | 0.6000 | 0.3000 | 0.3000 | 0.6000 |
| SeasonalNaive | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| CrostonSBA | 0.5000 | 0.5000 | 0.6000 | 0.3000 | 0.3000 | 0.6000 |
| TSB | 0.5000 | 0.5000 | 0.6000 | 0.3000 | 0.3000 | 0.6000 |
| LGBM-Q | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| LGBM-PoisNB | 0.5000 | 0.5000 | 0.6000 | 0.3000 | 0.3000 | 0.6000 |
| MLP-NB | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |

L=3:
| model | main_r1 | main_r2 | main_r5 | ext_r1 | ext_r2 | ext_r5 |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| SeasonalNaive | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.4000 |
| CrostonSBA | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| TSB | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| LGBM-Q | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.4000 |
| LGBM-PoisNB | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| MLP-NB | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.4000 |

Chi phí TEST tại q*(val) (L=2) với lưới chính (main) và lưới mở rộng (ext):
| model | test_main_r1 | test_main_r2 | test_main_r5 | test_ext_r1 | test_ext_r2 | test_ext_r5 |
|---|---|---|---|---|---|---|
| WindowAverage | 126.1652 | 148.9696 | 215.1276 | 104.6027 | 137.0000 | 215.1276 |
| SeasonalNaive | 144.3536 | 164.7327 | 225.8700 | 114.0085 | 144.3013 | 225.8700 |
| CrostonSBA | 127.5796 | 153.8894 | 229.4022 | 105.1640 | 141.9453 | 229.4022 |
| TSB | 129.0583 | 150.5468 | 213.5334 | 106.5516 | 137.3913 | 213.5334 |
| LGBM-Q | 134.6318 | 151.9465 | 203.8906 | 111.4350 | 138.0061 | 203.8906 |
| LGBM-PoisNB | 126.3621 | 146.2916 | 203.8335 | 105.1397 | 134.8372 | 203.8335 |
| MLP-NB | 131.1968 | 148.9064 | 202.0352 | 109.8165 | 135.8129 | 202.0352 |

L=3:
| model | test_main_r1 | test_main_r2 | test_main_r5 | test_ext_r1 | test_ext_r2 | test_ext_r5 |
|---|---|---|---|---|---|---|
| WindowAverage | 142.6549 | 163.8870 | 227.5832 | 115.8639 | 145.9162 | 227.5832 |
| SeasonalNaive | 158.0547 | 176.9295 | 233.5541 | 123.6999 | 151.6051 | 232.2430 |
| CrostonSBA | 145.7072 | 170.5577 | 245.1094 | 117.0243 | 151.7546 | 245.1094 |
| TSB | 146.7934 | 166.6853 | 226.3609 | 119.0207 | 147.4933 | 226.3609 |
| LGBM-Q | 150.0583 | 166.5990 | 216.2211 | 122.2187 | 147.3147 | 217.1446 |
| LGBM-PoisNB | 142.0462 | 160.6731 | 216.5541 | 115.9635 | 143.6391 | 216.5541 |
| MLP-NB | 143.8177 | 161.1896 | 213.3050 | 119.4301 | 144.5030 | 215.0948 |

Nhận xét: ở tỷ lệ 1 và 2, q* mới chạm cận dưới mới (0.3) ở mọi mô hình và chi phí test giảm đáng kể so với q*=0.5 (ví dụ L=2, tỷ lệ 1: WindowAverage 126.17 → 104.60, MLP-NB 131.20 → 109.82); ở tỷ lệ 5 q* mới là 0.5–0.6 (L=2) hoặc 0.4–0.5 (L=3) nên chi phí đổi ít hoặc không đổi. Như vậy trước đây (GĐ4–5) q*=0.5 ở tỷ lệ ≤2 là cận lưới và đã đánh giá thấp lợi ích của việc đặt hàng ít hơn; q* thật ở tỷ lệ 1–2 nằm ở hoặc dưới 0.3 (vẫn có thể chạm cận mới).

## 5. Hiệu chuẩn rời rạc (mục 4)
P(y<q̂) và P(y≤q̂) cho mọi mô hình, q ≤ 0.95 (7 mức), tổng thể 823 chuỗi, H=3 và H=4, val và test; "trong khoảng" = P(y<q̂) ≤ q ≤ P(y≤q̂). Đếm số q (trên 7) thỏa: (a) khoảng chặt; (b) khoảng nới ±0.05; và số q thỏa |P(y≤q̂) − q| ≤ 0.05:
| model | test_H3_strict | test_H4_strict | val_H3_strict | val_H4_strict | test_H3_tol0.05 | test_H4_tol0.05 | val_H3_tol0.05 | val_H4_tol0.05 |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 4 | 4 | 4 | 4 | 7 | 7 | 7 | 7 |
| SeasonalNaive | 1 | 1 | 2 | 2 | 3 | 3 | 3 | 3 |
| CrostonSBA | 2 | 1 | 2 | 2 | 7 | 7 | 7 | 7 |
| TSB | 5 | 4 | 5 | 4 | 7 | 7 | 7 | 7 |
| LGBM-Q | 0 | 0 | 0 | 0 | 7 | 7 | 7 | 7 |
| LGBM-PoisNB | 7 | 6 | 7 | 6 | 7 | 7 | 7 | 7 |
| MLP-NB | 6 | 6 | 7 | 6 | 7 | 7 | 7 | 7 |

|P(y≤q̂) − q| ≤ 0.05:
| model | test_H3 | test_H4 | val_H3 | val_H4 |
|---|---|---|---|---|
| WindowAverage | 6 | 7 | 5 | 6 |
| SeasonalNaive | 2 | 3 | 2 | 2 |
| CrostonSBA | 7 | 7 | 7 | 7 |
| TSB | 5 | 6 | 5 | 6 |
| LGBM-Q | 7 | 7 | 7 | 7 |
| LGBM-PoisNB | 4 | 5 | 4 | 5 |
| MLP-NB | 4 | 5 | 4 | 5 |

**Lưu ý**: LGBM-Q cho phân vị liên tục (không làm tròn xuống số nguyên) nên P(y<q̂) = P(y≤q̂) hầu như trùng nhau (khoảng suy biến thành một điểm), khiến tiêu chí khoảng chặt luôn 0/7 (không phản ánh chất lượng); với LGBM-Q hãy dùng cột |P(y≤q̂)−q|≤0.05 (7/7 ở cả val và test, H=3 và H=4). SeasonalNaive chỉ 3/7 theo khoảng nới ở mọi (split,H).
Bảng "P(y<q̂) / P(y≤q̂)" (val, H=3):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.375 / 0.585 | 0.460 / 0.657 | 0.557 / 0.734 | 0.654 / 0.811 | 0.709 / 0.849 | 0.765 / 0.890 | 0.828 / 0.930 |
| SeasonalNaive | 0.365 / 0.577 | 0.409 / 0.614 | 0.486 / 0.680 | 0.564 / 0.744 | 0.603 / 0.774 | 0.641 / 0.804 | 0.694 / 0.842 |
| CrostonSBA | 0.392 / 0.544 | 0.486 / 0.621 | 0.586 / 0.700 | 0.692 / 0.785 | 0.753 / 0.827 | 0.813 / 0.871 | 0.879 / 0.920 |
| TSB | 0.378 / 0.592 | 0.467 / 0.663 | 0.561 / 0.738 | 0.659 / 0.812 | 0.714 / 0.850 | 0.773 / 0.890 | 0.840 / 0.932 |
| LGBM-Q | 0.537 / 0.541 | 0.632 / 0.632 | 0.725 / 0.725 | 0.812 / 0.812 | 0.863 / 0.863 | 0.906 / 0.906 | 0.950 / 0.950 |
| LGBM-PoisNB | 0.394 / 0.599 | 0.494 / 0.677 | 0.597 / 0.757 | 0.726 / 0.837 | 0.808 / 0.874 | 0.861 / 0.912 | 0.921 / 0.950 |
| MLP-NB | 0.394 / 0.620 | 0.484 / 0.692 | 0.578 / 0.765 | 0.678 / 0.842 | 0.744 / 0.877 | 0.818 / 0.913 | 0.913 / 0.952 |

val, H=4:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.376 / 0.563 | 0.466 / 0.642 | 0.559 / 0.721 | 0.658 / 0.801 | 0.710 / 0.844 | 0.769 / 0.884 | 0.828 / 0.926 |
| SeasonalNaive | 0.374 / 0.562 | 0.423 / 0.607 | 0.506 / 0.680 | 0.586 / 0.747 | 0.625 / 0.780 | 0.665 / 0.812 | 0.720 / 0.851 |
| CrostonSBA | 0.400 / 0.520 | 0.486 / 0.601 | 0.590 / 0.689 | 0.699 / 0.776 | 0.757 / 0.820 | 0.813 / 0.866 | 0.882 / 0.916 |
| TSB | 0.385 / 0.571 | 0.469 / 0.650 | 0.568 / 0.727 | 0.666 / 0.804 | 0.719 / 0.844 | 0.778 / 0.885 | 0.841 / 0.926 |
| LGBM-Q | 0.530 / 0.532 | 0.619 / 0.620 | 0.711 / 0.711 | 0.811 / 0.811 | 0.858 / 0.858 | 0.905 / 0.905 | 0.948 / 0.948 |
| LGBM-PoisNB | 0.404 / 0.577 | 0.501 / 0.659 | 0.617 / 0.744 | 0.752 / 0.827 | 0.807 / 0.868 | 0.863 / 0.906 | 0.918 / 0.946 |
| MLP-NB | 0.393 / 0.594 | 0.480 / 0.670 | 0.575 / 0.747 | 0.676 / 0.824 | 0.737 / 0.863 | 0.817 / 0.903 | 0.908 / 0.946 |

test, H=3:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.388 / 0.569 | 0.478 / 0.649 | 0.581 / 0.730 | 0.689 / 0.809 | 0.743 / 0.850 | 0.802 / 0.891 | 0.867 / 0.932 |
| SeasonalNaive | 0.375 / 0.552 | 0.420 / 0.594 | 0.510 / 0.666 | 0.596 / 0.736 | 0.635 / 0.768 | 0.676 / 0.797 | 0.732 / 0.835 |
| CrostonSBA | 0.371 / 0.524 | 0.466 / 0.608 | 0.570 / 0.694 | 0.686 / 0.783 | 0.748 / 0.828 | 0.812 / 0.874 | 0.880 / 0.922 |
| TSB | 0.394 / 0.580 | 0.490 / 0.663 | 0.592 / 0.740 | 0.699 / 0.815 | 0.755 / 0.855 | 0.815 / 0.896 | 0.884 / 0.937 |
| LGBM-Q | 0.509 / 0.513 | 0.606 / 0.607 | 0.706 / 0.706 | 0.806 / 0.806 | 0.855 / 0.855 | 0.903 / 0.903 | 0.948 / 0.948 |
| LGBM-PoisNB | 0.410 / 0.589 | 0.514 / 0.671 | 0.625 / 0.755 | 0.749 / 0.839 | 0.810 / 0.877 | 0.866 / 0.916 | 0.924 / 0.952 |
| MLP-NB | 0.403 / 0.600 | 0.499 / 0.679 | 0.599 / 0.757 | 0.706 / 0.836 | 0.769 / 0.874 | 0.846 / 0.911 | 0.917 / 0.949 |

test, H=4:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.394 / 0.548 | 0.490 / 0.636 | 0.588 / 0.720 | 0.695 / 0.801 | 0.747 / 0.842 | 0.804 / 0.884 | 0.865 / 0.925 |
| SeasonalNaive | 0.382 / 0.534 | 0.438 / 0.585 | 0.525 / 0.666 | 0.614 / 0.738 | 0.660 / 0.773 | 0.703 / 0.807 | 0.760 / 0.851 |
| CrostonSBA | 0.374 / 0.503 | 0.468 / 0.588 | 0.575 / 0.682 | 0.692 / 0.776 | 0.751 / 0.820 | 0.811 / 0.869 | 0.881 / 0.917 |
| TSB | 0.407 / 0.565 | 0.499 / 0.647 | 0.598 / 0.729 | 0.705 / 0.808 | 0.761 / 0.848 | 0.822 / 0.890 | 0.884 / 0.931 |
| LGBM-Q | 0.509 / 0.511 | 0.608 / 0.608 | 0.706 / 0.706 | 0.806 / 0.806 | 0.855 / 0.855 | 0.899 / 0.899 | 0.949 / 0.949 |
| LGBM-PoisNB | 0.429 / 0.574 | 0.535 / 0.662 | 0.646 / 0.749 | 0.760 / 0.834 | 0.814 / 0.873 | 0.870 / 0.913 | 0.925 / 0.949 |
| MLP-NB | 0.409 / 0.579 | 0.502 / 0.662 | 0.606 / 0.743 | 0.715 / 0.822 | 0.780 / 0.862 | 0.846 / 0.904 | 0.918 / 0.945 |

(Bảng dài đầy đủ: `results/calibration_table.csv`.)

## 6. Ablation biến ngoại sinh cho LightGBM-Q (mục 5)
"Bỏ ngoại sinh" = bỏ giá (price_fut, price_ratio_fut_past28, price_ratio_next_today), SNAP (fut_snap_days), sự kiện (fut_event_days, fut_event_index): 30 → 24 đặc trưng. GIỮ lại lịch sử doanh số, thứ/tháng của ngày dự báo và chỉ số hiệu ứng thứ (fut_wd_index), speed/demand_type. Cùng cấu hình LightGBM, cùng huấn luyện/đánh giá như bản chính ("full" tái hiện đúng số của GĐ2/5).
| split | H | model | mase | pinball_avg | wape |
|---|---|---|---|---|---|
| test | 3 | LGBM-Q[full] | 0.7870 | 0.9354 | 0.3508 |
| test | 3 | LGBM-Q[no_exog] | 0.7845 | 0.9385 | 0.3503 |
| test | 4 | LGBM-Q[full] | 0.8051 | 1.1344 | 0.3166 |
| test | 4 | LGBM-Q[no_exog] | 0.8014 | 1.1347 | 0.3163 |
| val | 3 | LGBM-Q[full] | 0.7672 | 0.9109 | 0.3697 |
| val | 3 | LGBM-Q[no_exog] | 0.7688 | 0.9185 | 0.3704 |
| val | 4 | LGBM-Q[full] | 0.7884 | 1.0922 | 0.3333 |
| val | 4 | LGBM-Q[no_exog] | 0.7913 | 1.1001 | 0.3345 |

| split | H | pinball_full | pinball_no_exog | tang_tuong_doi |
|---|---|---|---|---|
| test | 3 | 0.9354 | 0.9385 | 0.0033 |
| test | 4 | 1.1344 | 1.1347 | 0.0003 |
| val | 3 | 0.9109 | 0.9185 | 0.0084 |
| val | 4 | 1.0922 | 1.1001 | 0.0073 |

So sánh tham chiếu (pinball_avg): 
| index | val_H3 | val_H4 | test_H3 | test_H4 |
|---|---|---|---|---|
| LGBM-Q | 0.9109 | 1.0922 | 0.9354 | 1.1344 |
| TSB | 1.0398 | 1.2377 | 1.0799 | 1.3023 |
| CrostonSBA | 1.2539 | 1.5197 | 1.2659 | 1.5402 |

**Diễn giải lại**: bỏ giá/SNAP/sự kiện chỉ làm pinball_avg tăng ≤ 1% (xem cột tang_tuong_doi), trong khi LightGBM-Q không ngoại sinh vẫn hơn hẳn TSB (test H=3: 0.9385 vs 1.0799) và CrostonSBA. Do đó lời giải thích ở báo cáo trước ("lợi thế đến từ biến ngoại sinh giá/SNAP/sự kiện") KHÔNG được dữ liệu ủng hộ và được rút lại. Lợi thế của các mô hình học máy so với Croston/TSB đến từ các phần còn lại: đặc trưng lịch sử đa thang (trung bình/độ lệch chuẩn 7–180 ngày, tỷ lệ ngày có bán, số ngày kể từ lần bán cuối, tuổi mặt hàng), doanh số cùng thứ của các tuần trước, thành phần thứ trong cửa sổ (EDA: Chủ nhật 4.1099 vs Thứ 4 2.5597), và việc huấn luyện trực tiếp phân vị/phân phối của D_H. Ablation này chưa tách riêng yếu tố lịch (thứ/tháng vẫn còn trong bản "no_exog") nên chưa kết luận được trọng số từng yếu tố.

## 7. MLP-NB ba seed và hội tụ (mục 6)
Mỗi seed: huấn luyện trên train (epoch chọn bằng holdout nội bộ), kappa chọn trên val, đánh giá val; huấn luyện lại trên train+val với epoch/kappa đó và đánh giá test. Trung bình ± độ lệch chuẩn (ddof=1, n=3) của pinball_avg, WAPE, MASE:
| split | H | n_seed | pinball_avg | wape | mase | epochs | kappa |
|---|---|---|---|---|---|---|---|
| test | 3 | 3 | 0.9477 ± 0.0021 | 0.3515 ± 0.0004 | 0.7843 ± 0.0005 | 17/25/20 | 0.5/0.5/0.5 |
| test | 4 | 3 | 1.1449 ± 0.0045 | 0.3170 ± 0.0006 | 0.8031 ± 0.0032 | 30/16/26 | 0.5/0.5/0.5 |
| val | 3 | 3 | 0.9026 ± 0.0005 | 0.3680 ± 0.0029 | 0.7559 ± 0.0043 | 17/25/20 | 0.5/0.5/0.5 |
| val | 4 | 3 | 1.0899 ± 0.0038 | 0.3315 ± 0.0022 | 0.7748 ± 0.0043 | 30/16/26 | 0.5/0.5/0.5 |

Chi tiết từng seed:
| seed | H | split | kappa | epochs | pinball_avg | wape | mase |
|---|---|---|---|---|---|---|---|
| 42 | 3 | val | 0.5000 | 17 | 0.9025 | 0.3701 | 0.7597 |
| 42 | 4 | val | 0.5000 | 30 | 1.0929 | 0.3328 | 0.7746 |
| 42 | 3 | test | 0.5000 | 17 | 0.9477 | 0.3512 | 0.7839 |
| 42 | 4 | test | 0.5000 | 30 | 1.1489 | 0.3165 | 0.8024 |
| 43 | 3 | val | 0.5000 | 25 | 0.9021 | 0.3646 | 0.7513 |
| 43 | 4 | val | 0.5000 | 16 | 1.0911 | 0.3327 | 0.7792 |
| 43 | 3 | test | 0.5000 | 25 | 0.9499 | 0.3520 | 0.7849 |
| 43 | 4 | test | 0.5000 | 16 | 1.1457 | 0.3171 | 0.8002 |
| 44 | 3 | val | 0.5000 | 20 | 0.9031 | 0.3693 | 0.7567 |
| 44 | 4 | val | 0.5000 | 26 | 1.0857 | 0.3290 | 0.7706 |
| 44 | 3 | test | 0.5000 | 20 | 0.9456 | 0.3512 | 0.7842 |
| 44 | 4 | test | 0.5000 | 26 | 1.1401 | 0.3176 | 0.8066 |

Seed 42 tái hiện đúng số GĐ2/5 (0.9025 và 1.0929 trên val; 0.9477 và 1.1489 trên test). Độ biến thiên theo seed nhỏ (độ lệch chuẩn ≤ 0.0045), nhỏ hơn nhiều chênh lệch giữa MLP-NB và Croston/TSB. Thứ hạng MLP-NB vs LightGBM-Q lật giữa val và test KHÔNG phải do nhiễu seed: trên val cả 3 seed đều tốt hơn LGBM-Q (0.9021–0.9031 vs 0.9109 ở H=3), trên test cả 3 seed đều kém hơn LGBM-Q (0.9456–0.9499 vs 0.9354 ở H=3; 1.1401–1.1489 vs 1.1344 ở H=4).
Hội tụ (NLL holdout nội bộ; seed 42; chạy tối đa 80 epoch; `results/figures/mlp_nll_curves.png`):
| H | epoch_tot_nhat(<=80) | NLL_tai_epoch_30 | NLL_tot_nhat | NLL_epoch_80 |
|---|---|---|---|---|
| 3.0000 | 58.0000 | 2.1620 | 2.1591 | 2.1621 |
| 4.0000 | 46.0000 | 2.3424 | 2.3414 | 2.3448 |

Epoch tốt nhất khi cho chạy tới 80 là 58 (H=3) và 46 (H=4); đường NLL gần như phẳng (H=4: NLL ở epoch 30 hơn NLL tốt nhất 0.0010, ≈0.04%) rồi tăng nhẹ ở epoch 80 (quá khớp). Kết luận: H=4 với trần 30 epoch chưa đạt đúng cực tiểu nhưng đã nằm trên đoạn phẳng; chưa chạy lại pinball với 46 epoch nên chưa định lượng tác động lên pinball (dự kiến nhỏ).

## 8. MASE trong bảng chính (mục 7)
MASE (trung bình trên chuỗi của MAE theo cửa sổ H / MAE naive trong mẫu theo cửa sổ H, scale CHỈ tính trên train, kiểm bằng `test_mase_scale_uses_train_only_and_equals_hand_calculation`). Tổng thể 823 chuỗi, theo mô hình, không phụ thuộc tỷ lệ chi phí/L/q (đánh giá dự báo cửa sổ). H=3:
| model | pinball_avg_val_H3 | wape_val_H3 | mase_val_H3 | pinball_avg_test_H3 | wape_test_H3 | mase_test_H3 |
|---|---|---|---|---|---|---|
| WindowAverage | 1.0757 | 0.4129 | 0.8125 | 1.1316 | 0.3987 | 0.8474 |
| SeasonalNaive | 1.4002 | 0.4741 | 0.9645 | 1.4112 | 0.4399 | 0.9882 |
| CrostonSBA | 1.2539 | 0.4948 | 0.9502 | 1.2659 | 0.4484 | 0.9174 |
| TSB | 1.0398 | 0.4058 | 0.8009 | 1.0799 | 0.3898 | 0.8287 |
| LGBM-Q | 0.9109 | 0.3697 | 0.7672 | 0.9354 | 0.3508 | 0.7870 |
| LGBM-PoisNB | 0.9302 | 0.3697 | 0.7672 | 0.9487 | 0.3508 | 0.7870 |
| MLP-NB | 0.9025 | 0.3701 | 0.7597 | 0.9477 | 0.3512 | 0.7839 |

H=4:
| model | pinball_avg_val_H4 | wape_val_H4 | mase_val_H4 | pinball_avg_test_H4 | wape_test_H4 | mase_test_H4 |
|---|---|---|---|---|---|---|
| WindowAverage | 1.2918 | 0.3682 | 0.8320 | 1.3782 | 0.3575 | 0.8677 |
| SeasonalNaive | 1.6261 | 0.4214 | 0.9792 | 1.6239 | 0.3859 | 0.9882 |
| CrostonSBA | 1.5197 | 0.4506 | 0.9886 | 1.5402 | 0.4060 | 0.9445 |
| TSB | 1.2377 | 0.3605 | 0.8173 | 1.3023 | 0.3474 | 0.8444 |
| LGBM-Q | 1.0922 | 0.3333 | 0.7884 | 1.1344 | 0.3166 | 0.8051 |
| LGBM-PoisNB | 1.1192 | 0.3333 | 0.7884 | 1.1501 | 0.3166 | 0.8051 |
| MLP-NB | 1.0929 | 0.3328 | 0.7746 | 1.1489 | 0.3165 | 0.8024 |


## 9. Chú thích bắt buộc cho bảng theo nhóm (mục 8)
Đã bổ sung chú thích tỷ lệ chi phí, L và q cho các bảng theo nhóm nhanh/chậm trong `02`, `03`, `04`, `05` và `final_report.md`. Quy ước: bảng sai số dự báo theo nhóm (pinball_avg, WAPE, MASE) tính trên dự báo cửa sổ H=3 và không phụ thuộc tỷ lệ chi phí/L/q (pinball_avg là trung bình 9 mức q); bảng mô phỏng theo nhóm ghi rõ cost_ratio, L và q (hoặc "q*" kèm bảng q*).

## 10. Cấu hình
Seed 42 (MLP seed 42/43/44). k_grid 0.5–3.0 bước 0.1; lưới q mở rộng [0.3, 0.4]; bootstrap 2000 lần seed 42. lightgbm 4.7.0, torch 2.15.0+cpu, scipy 1.18.1, numpy 2.5.4, pandas 2.3.3. Thời gian: chạy lại val/test ≈ 9.5 + 4.5 phút, ablation ≈ 4.5 phút, mlp_seeds ≈ 16 phút (CPU 12 luồng).

## 11. Vấn đề và hạn chế
- q* và k vẫn chạm cận dưới lưới ở tỷ lệ 1–2 (q*=0.3; k*=0.5).
- Mọi lựa chọn (k, q*) trên val, báo cáo test; KTC bootstrap coi chuỗi độc lập; chỉ 28 ngày.
- MLP-NB H=4 chưa chạy với epoch tối ưu 46; ablation chưa tách yếu tố lịch; chưa có đối chứng "Croston/TSB + biến ngoại sinh".
- Kết quả chính của GĐ2–5 không đổi (đã kiểm chứng), nhưng các kết luận diễn giải ở câu 1 và câu 3 được sửa như trên.

## 12. Chưa làm
Chạy lại MLP-NB H=4 với 46 epoch; k theo nhóm; lưới q xuống dưới 0.3; ablation lịch.
