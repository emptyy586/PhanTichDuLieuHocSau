# Báo cáo GĐ6 – Bổ sung trước khi chốt đồ án

## 1. Đã làm gì (file tạo/sửa)
- **Mới**: `src/diem_an_toan.py` (chính sách điểm có hệ số an toàn + q* lưới mở rộng + bootstrap), `src/hieu_chinh.py` (bảng hiệu chuẩn rời rạc), `src/loai_bo_dac_trung.py` (bỏ biến ngoại sinh), `src/mlp_cac_seed.py` (3 seed + hội tụ), `src/bang_bao_cao_gd6.py`, `tests/kiem_thu_gd6.py` (7 test). Kết quả: `results/diem_an_toan_test.csv`, `q_sao_ext_val.csv`, `mo_phong_ton_kho_{val,test}_ext.csv`, `hieu_chinh_bang.csv`, `hieu_chinh_tom_tat.csv`, `loai_bo_bien_ngoai_sinh.csv`, `mlp_cac_seed.csv`, `mlp_duong_nll.csv`, hình `results/figures/mlp_duong_nll.png`.
- **Sửa (không đổi kết quả chính)**: `src/mo_hinh_ml.py` (`fit_lgbm(extra_q=...)`, `fit_mlp_nb(seed=, max_epochs=)`, `Qext`), `src/mo_hinh_xac_suat.py` (`EXT_QGRID`, khóa `Qext`), `src/chay_du_bao.py` (lưu `Qext` vào npz), `src/mo_phong_ton_kho.py` (`--ext`), `configs/thong_so.yaml` (`quantile_grid_ext`, `safety_k_grid`, `mlp_seeds`, `mlp_convergence_epochs`).
- **Lưu ý thời điểm**: báo cáo này viết ở GĐ6 và được DỰNG LẠI ở GĐ7 từ kết quả mới (làm tròn LGBM-Q về số nguyên trong mô phỏng; lưới kappa mới; xem `07_sua_loi.md`). Ở GĐ6, chạy lại `run_forecasts` chỉ để sinh `Qext` không đổi kết quả; GĐ7 mới làm đổi nhẹ kết quả của CrostonSBA/TSB/LGBM-PoisNB (kappa) và các số mô phỏng của LGBM-Q (làm tròn).
- **Mục 1 (yêu cầu dán pytest vào 05)**: `05_kiem_tra_cuoi.md` có output `pytest -v` nguyên văn ở mục 2; đã thay bằng output mới nhất ở cuối GĐ7 (cùng output với mục 2 của `07_sua_loi.md`).

## 2. Lệnh đã chạy và pytest
```
python -m src.chay_du_bao --split val ; python -m src.chay_du_bao --split test
python -m src.mo_phong_ton_kho --split val --ext ; python -m src.mo_phong_ton_kho --split test --ext
python -m src.diem_an_toan ; python -m src.hieu_chinh ; python -m src.loai_bo_dac_trung ; python -m src.mlp_cac_seed
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

============================= 56 passed in 12.58s =============================
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
| LGBM-Q | 0.3000 | 0.3000 | 0.5000 | 0.8000 | 0.8500 | 0.9500 |
| LGBM-PoisNB | 0.3000 | 0.3000 | 0.6000 | 0.8000 | 0.9000 | 0.9500 |
| MLP-NB | 0.3000 | 0.3000 | 0.5000 | 0.8000 | 0.9000 | 0.9500 |

Test, L=2, tỷ lệ 10 – chi phí TB/chuỗi:
| model | k_val | q_star_val | cost_val_point | cost_val_k | cost_val_q | cost_test_point | cost_test_k | cost_test_q |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 1.3000 | 0.8000 | 291.1191 | 272.0267 | 271.4933 | 306.8226 | 283.4569 | 283.2892 |
| SeasonalNaive | 1.1000 | 0.7000 | 311.5844 | 301.9174 | 305.1470 | 320.7995 | 305.6343 | 308.2199 |
| CrostonSBA | 1.3000 | 0.8000 | 325.9490 | 300.7631 | 301.4739 | 335.2916 | 304.2211 | 304.2539 |
| TSB | 1.3000 | 0.8000 | 290.5103 | 270.5614 | 269.6221 | 301.0486 | 278.6160 | 278.0923 |
| LGBM-Q | 1.2000 | 0.8000 | 276.0923 | 257.7740 | 254.4751 | 283.9842 | 261.2734 | 258.5018 |
| LGBM-PoisNB | 1.2000 | 0.8000 | 276.0923 | 257.7740 | 256.5772 | 283.9842 | 261.2734 | 260.0826 |
| MLP-NB | 1.2000 | 0.8000 | 267.0243 | 255.9186 | 252.4508 | 275.8068 | 260.6087 | 258.5030 |

**Test, L=2: chi phí[điểm an toàn k] − chi phí[phân vị q*] [KTC 95%]** (dương = phân vị rẻ hơn):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -7.48 [-9.26; -5.81] | 2.46 [0.91; 4.08] | -0.09 [-1.22; 0.84] | 0.17 [-0.97; 1.40] | 10.04 [5.80; 14.82] | 36.70 [26.27; 48.45] |
| SeasonalNaive | -14.61 [-16.59; -12.68] | -2.92 [-4.80; -0.91] | 5.00 [2.65; 7.63] | -2.59 [-3.81; -1.34] | -3.29 [-6.54; 0.06] | -5.36 [-18.21; 5.58] |
| CrostonSBA | -3.20 [-4.61; -1.93] | 3.29 [2.00; 4.72] | -0.09 [-0.74; 0.54] | -0.03 [-1.63; 1.53] | 6.88 [2.90; 11.30] | 40.01 [25.61; 54.99] |
| TSB | -6.80 [-8.61; -5.14] | 3.20 [1.72; 4.68] | 0.84 [-0.06; 1.61] | 0.52 [-0.79; 1.95] | 7.27 [3.42; 11.82] | 16.21 [5.05; 27.92] |
| LGBM-Q | -6.49 [-8.19; -4.82] | 5.42 [3.53; 7.41] | 2.10 [0.94; 3.32] | 2.77 [0.82; 4.99] | 0.76 [-4.05; 5.96] | 23.39 [12.15; 35.23] |
| LGBM-PoisNB | -6.30 [-8.00; -4.70] | 4.90 [3.15; 6.78] | 1.76 [0.68; 2.57] | 1.19 [-1.09; 3.76] | 5.07 [0.61; 10.14] | 21.16 [9.89; 33.30] |
| MLP-NB | -10.98 [-13.11; -8.91] | 3.08 [1.13; 5.14] | 3.79 [2.73; 4.88] | 2.11 [0.68; 3.77] | 8.67 [5.26; 12.60] | 26.52 [16.68; 37.17] |

L=3:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -15.56 [-18.11; -13.21] | -3.02 [-4.79; -1.23] | 0.54 [-0.82; 1.72] | -3.54 [-6.00; -1.51] | 4.53 [1.30; 8.04] | 36.89 [26.11; 48.77] |
| SeasonalNaive | -22.03 [-24.67; -19.38] | -8.20 [-10.22; -6.33] | 2.87 [1.14; 4.77] | -4.49 [-6.22; -2.91] | -4.11 [-6.77; -1.47] | -10.09 [-21.12; 0.97] |
| CrostonSBA | -9.16 [-11.23; -7.28] | -0.47 [-1.88; 0.97] | 0.95 [0.03; 1.82] | -3.89 [-4.96; -2.85] | 1.79 [-1.92; 5.57] | 38.00 [25.06; 52.04] |
| TSB | -15.09 [-17.63; -12.70] | -2.61 [-4.25; -1.03] | 1.56 [0.32; 2.57] | -0.80 [-2.58; 0.82] | 3.21 [-0.00; 6.63] | 23.34 [13.18; 34.81] |
| LGBM-Q | -13.65 [-16.04; -11.35] | 0.56 [-1.36; 2.44] | 5.56 [3.94; 7.37] | -0.02 [-2.39; 2.20] | 3.11 [-1.27; 7.93] | 22.81 [10.73; 35.55] |
| LGBM-PoisNB | -13.61 [-15.96; -11.34] | -0.28 [-2.15; 1.59] | 3.56 [2.17; 4.92] | -0.43 [-2.84; 1.44] | 2.65 [-1.48; 7.00] | 21.50 [6.80; 35.62] |
| MLP-NB | -17.43 [-20.09; -14.80] | -1.47 [-3.61; 0.61] | 2.90 [1.72; 4.15] | 2.06 [0.94; 3.33] | 6.39 [3.03; 10.02] | 25.81 [13.48; 39.36] |

Test, L=2: phân vị q* − điểm thuần (k=1):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -37.02 [-39.11; -35.01] | -22.97 [-24.56; -21.36] | 0.09 [-0.84; 1.22] | -23.53 [-29.59; -18.08] | -134.36 [-156.95; -115.18] | -561.10 [-638.70; -493.37] |
| SeasonalNaive | -41.59 [-43.93; -39.31] | -29.65 [-31.28; -27.97] | -3.15 [-4.00; -2.17] | -12.58 [-16.05; -9.36] | -96.14 [-109.83; -83.21] | -466.20 [-517.28; -416.74] |
| CrostonSBA | -41.32 [-44.62; -38.26] | -24.60 [-27.47; -21.91] | 0.09 [-0.54; 0.74] | -31.04 [-38.22; -24.52] | -160.66 [-185.02; -139.04] | -660.25 [-745.09; -586.25] |
| TSB | -39.86 [-42.02; -37.80] | -25.51 [-27.14; -23.84] | -0.84 [-1.61; 0.06] | -22.96 [-29.08; -17.50] | -129.34 [-151.36; -110.03] | -537.32 [-618.04; -466.32] |
| LGBM-Q | -37.91 [-39.55; -36.37] | -24.32 [-25.67; -22.96] | -2.10 [-3.32; -0.94] | -25.48 [-31.20; -20.37] | -117.52 [-133.93; -103.35] | -501.46 [-561.30; -448.74] |
| LGBM-PoisNB | -38.11 [-39.90; -36.48] | -23.80 [-25.16; -22.49] | -1.76 [-2.57; -0.68] | -23.90 [-29.79; -18.79] | -121.84 [-140.82; -105.20] | -499.24 [-561.83; -445.81] |
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

(L=3: `results/diem_an_toan_test.csv`.) Đếm ô (7 mô hình × 6 tỷ lệ = 42 ô mỗi L) theo vị trí KTC so với 0:
| L | so_sanh | KTC<0 (vế đầu rẻ hơn) | KTC chứa 0 | KTC>0 (vế đầu đắt hơn) | tong_o |
|---|---|---|---|---|---|
| 2 | k_minus_q | 9 | 10 | 23 | 42 |
| 2 | q_minus_point | 39 | 3 | 0 | 42 |
| 2 | k_minus_point | 35 | 6 | 1 | 42 |
| 3 | k_minus_q | 14 | 13 | 15 | 42 |
| 3 | q_minus_point | 40 | 2 | 0 | 42 |
| 3 | k_minus_point | 37 | 5 | 0 | 42 |

k_minus_q = điểm an toàn − phân vị; q_minus_point = phân vị − điểm thuần; k_minus_point = điểm an toàn − điểm thuần.

**Kết luận cho câu 1 (số liệu sau GĐ7; thay thế kết luận ở 04/05).**
- Phần lớn "lợi thế của phân vị" trong GĐ4–5 là lợi thế của việc CÓ hệ số an toàn: so với điểm thuần, điểm an toàn rẻ hơn rõ ở 35/42 ô (L=2) và 37/42 ô (L=3), gần bằng mức rẻ hơn của phân vị (39/42 và 40/42; ví dụ L=2 tỷ lệ 10: CrostonSBA -31.04 [-38.22; -24.52], MLP-NB -17.30 [-21.74; -13.44]; tỷ lệ 50: -660.25 … -429.60).
- Điểm an toàn so với phân vị (k* − q*; 42 ô mỗi L): L=2: 9 ô điểm an toàn rẻ hơn, 10 ô không phân biệt được, 23 ô phân vị rẻ hơn; L=3: 14 / 13 / 15.
  - **Tỷ lệ 10 (L=2)**: không phân biệt được ở 4/7 mô hình (WindowAverage, CrostonSBA, TSB, LGBM-PoisNB); phân vị rẻ hơn ở: LGBM-Q 2.77 [0.82; 4.99]; MLP-NB 2.11 [0.68; 3.77]; điểm an toàn rẻ hơn ở: SeasonalNaive -2.59 [-3.81; -1.34].
  - **Tỷ lệ ≥20 (L=2)**: phân vị rẻ hơn rõ ở 5/7 (tỷ lệ 20) và 6/7 mô hình (tỷ lệ 50; chênh 16.21 … 40.01 đơn vị chi phí).
  - **Tỷ lệ 1**: điểm an toàn rẻ hơn ở 7/7 mô hình (L=2; ví dụ SeasonalNaive -14.61 [-16.59; -12.68]). **Tỷ lệ 2 (L=2)**: câu cũ "điểm an toàn thường rẻ hơn hoặc ngang" là SAI cho tỷ lệ 2 — phân vị rẻ hơn rõ ở 6/7 mô hình (WindowAverage, CrostonSBA, TSB, LGBM-Q, LGBM-PoisNB, MLP-NB; ví dụ WindowAverage 2.46 [0.91; 4.08], MLP-NB 3.08 [1.13; 5.14]), điểm an toàn rẻ hơn ở: SeasonalNaive. Ở tỷ lệ 1–2 cả k*=0.5 và q*=0.3 chạm cận lưới nên so sánh bị chặn.
- Tóm lại: "phân vị tốt hơn điểm" đúng khi so với điểm THUẦN; so với điểm có hệ số an toàn phân vị hầu như ngang ở tỷ lệ 5–10, hơn nhẹ ở tỷ lệ 2 và hơn rõ ở tỷ lệ cao (≥20). Kết luận GĐ4–5 ("phân vị rẻ hơn điểm ở mọi tỷ lệ") chỉ đúng so với điểm thuần.
- Hạn chế: k và q* cùng chọn trên val; một k chung cho mọi chuỗi (k theo nhóm tốc độ: `07_sua_loi.md` mục 3.5); 28 ngày; KTC coi chuỗi độc lập.

## 4. Mở rộng lưới q xuống 0.3, 0.4 (mục 3)
Phạm vi: **tất cả 7 mô hình** được mở rộng (không phải chỉ NB): bốn mô hình thống kê, LGBM-PoisNB và MLP-NB dùng cùng mean/alpha đã chốt nên chỉ tính thêm phân vị NB; LGBM-Q huấn luyện thêm 2 mô hình phân vị (0.3 và 0.4; thêm ≈10 giây mỗi H). Lưới tuning/đánh giá chính (9 mức, 0.5–0.99) giữ nguyên để không đổi các kết quả cũ; lưới mở rộng chỉ dùng cho mô phỏng (`mo_phong_ton_kho_*_ext.csv`). Phân vị LGBM-Q phụ được ép không vượt phân vị q=0.5 và tăng dần.
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
| LGBM-Q | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| LGBM-PoisNB | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.5000 |
| MLP-NB | 0.5000 | 0.5000 | 0.5000 | 0.3000 | 0.3000 | 0.4000 |

Chi phí TEST tại q*(val) (L=2) với lưới chính (main) và lưới mở rộng (ext):
| model | test_main_r1 | test_main_r2 | test_main_r5 | test_ext_r1 | test_ext_r2 | test_ext_r5 |
|---|---|---|---|---|---|---|
| WindowAverage | 126.1652 | 148.9696 | 215.1276 | 104.6027 | 137.0000 | 215.1276 |
| SeasonalNaive | 144.3536 | 164.7327 | 225.8700 | 114.0085 | 144.3013 | 225.8700 |
| CrostonSBA | 126.3196 | 153.0996 | 229.5346 | 103.4532 | 141.3439 | 229.5346 |
| TSB | 128.2248 | 150.0255 | 213.5747 | 105.2503 | 136.9271 | 213.5747 |
| LGBM-Q | 125.9441 | 145.3135 | 203.4216 | 104.8396 | 134.1276 | 203.4216 |
| LGBM-PoisNB | 126.0693 | 146.1057 | 203.7582 | 104.6452 | 134.6428 | 203.7582 |
| MLP-NB | 131.1968 | 148.9064 | 202.0352 | 109.8165 | 135.8129 | 202.0352 |

L=3:
| model | test_main_r1 | test_main_r2 | test_main_r5 | test_ext_r1 | test_ext_r2 | test_ext_r5 |
|---|---|---|---|---|---|---|
| WindowAverage | 142.6549 | 163.8870 | 227.5832 | 115.8639 | 145.9162 | 227.5832 |
| SeasonalNaive | 158.0547 | 176.9295 | 233.5541 | 123.6999 | 151.6051 | 232.2430 |
| CrostonSBA | 144.0571 | 169.3366 | 245.1750 | 114.7485 | 150.5055 | 245.1750 |
| TSB | 145.7193 | 165.8663 | 226.3074 | 117.3050 | 146.5298 | 226.3074 |
| LGBM-Q | 142.0049 | 160.1373 | 214.5346 | 115.4739 | 142.5577 | 214.5346 |
| LGBM-PoisNB | 141.6367 | 160.3621 | 216.5383 | 115.4374 | 143.3985 | 216.5383 |
| MLP-NB | 143.8177 | 161.1896 | 213.3050 | 119.4301 | 144.5030 | 215.0948 |

Nhận xét: ở tỷ lệ 1 q* mới là 0.3 (L=2), 0.3 (L=3) (cận lưới mới 0.3) và chi phí test giảm đáng kể so với q* trên lưới chính (ví dụ L=2, tỷ lệ 1: WindowAverage 126.17 → 104.60, MLP-NB 131.20 → 109.82); ở tỷ lệ 2 q* mới 0.3 (L=2), 0.3 (L=3); ở tỷ lệ 5 q* mới là 0.5–0.6 (L=2) và 0.4–0.5 (L=3). Như vậy trước đây q*=0.5 ở tỷ lệ thấp là cận lưới và đã đánh giá thấp lợi ích của việc đặt hàng ít hơn; q* thật ở tỷ lệ 1–2 nằm ở hoặc dưới 0.3.

## 5. Hiệu chuẩn rời rạc (mục 4)
P(y<q̂) và P(y≤q̂) cho mọi mô hình, q ≤ 0.95 (7 mức), tổng thể 823 chuỗi, H=3 và H=4, val và test; "trong khoảng" = P(y<q̂) ≤ q ≤ P(y≤q̂). Đếm số q (trên 7) thỏa: (a) khoảng chặt; (b) khoảng nới ±0.05; và số q thỏa |P(y≤q̂) − q| ≤ 0.05:
| model | test_H3_strict | test_H4_strict | val_H3_strict | val_H4_strict | test_H3_tol0.05 | test_H4_tol0.05 | val_H3_tol0.05 | val_H4_tol0.05 |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 4 | 4 | 4 | 4 | 7 | 7 | 7 | 7 |
| SeasonalNaive | 1 | 1 | 2 | 2 | 3 | 3 | 3 | 3 |
| CrostonSBA | 2 | 0 | 3 | 1 | 7 | 7 | 7 | 7 |
| TSB | 6 | 5 | 5 | 4 | 7 | 7 | 7 | 7 |
| LGBM-Q | 0 | 0 | 0 | 0 | 7 | 7 | 7 | 7 |
| LGBM-PoisNB | 7 | 7 | 7 | 6 | 7 | 7 | 7 | 7 |
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

**Lưu ý**: LGBM-Q cho phân vị liên tục (không làm tròn xuống số nguyên) nên P(y<q̂) = P(y≤q̂) hầu như trùng nhau (khoảng suy biến thành một điểm), khiến tiêu chí khoảng chặt không phản ánh chất lượng (0/7); với LGBM-Q dùng |P(y≤q̂)−q|≤0.05 (7/7 ở mọi split và H). Theo khoảng nới ±0.05: mọi mô hình trừ SeasonalNaive đạt tối thiểu 7/7; SeasonalNaive 3–3/7 (mọi split, H). (Khoảng này là của dự báo/phân vị, không bị ảnh hưởng bởi làm tròn trong mô phỏng ở GĐ7.)
Bảng "P(y<q̂) / P(y≤q̂)" (val, H=3):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.375 / 0.585 | 0.460 / 0.657 | 0.557 / 0.734 | 0.654 / 0.811 | 0.709 / 0.849 | 0.765 / 0.890 | 0.828 / 0.930 |
| SeasonalNaive | 0.365 / 0.577 | 0.409 / 0.614 | 0.486 / 0.680 | 0.564 / 0.744 | 0.603 / 0.774 | 0.641 / 0.804 | 0.694 / 0.842 |
| CrostonSBA | 0.386 / 0.538 | 0.484 / 0.619 | 0.587 / 0.700 | 0.699 / 0.791 | 0.761 / 0.833 | 0.821 / 0.877 | 0.888 / 0.926 |
| TSB | 0.373 / 0.588 | 0.465 / 0.661 | 0.563 / 0.740 | 0.664 / 0.817 | 0.720 / 0.855 | 0.780 / 0.895 | 0.847 / 0.936 |
| LGBM-Q | 0.537 / 0.541 | 0.632 / 0.632 | 0.725 / 0.725 | 0.812 / 0.812 | 0.863 / 0.863 | 0.906 / 0.906 | 0.950 / 0.950 |
| LGBM-PoisNB | 0.392 / 0.598 | 0.494 / 0.676 | 0.597 / 0.757 | 0.728 / 0.839 | 0.811 / 0.876 | 0.864 / 0.914 | 0.923 / 0.952 |
| MLP-NB | 0.394 / 0.620 | 0.484 / 0.692 | 0.578 / 0.765 | 0.678 / 0.842 | 0.744 / 0.877 | 0.818 / 0.913 | 0.913 / 0.952 |

val, H=4:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.376 / 0.563 | 0.466 / 0.642 | 0.559 / 0.721 | 0.658 / 0.801 | 0.710 / 0.844 | 0.769 / 0.884 | 0.828 / 0.926 |
| SeasonalNaive | 0.374 / 0.562 | 0.423 / 0.607 | 0.506 / 0.680 | 0.586 / 0.747 | 0.625 / 0.780 | 0.665 / 0.812 | 0.720 / 0.851 |
| CrostonSBA | 0.393 / 0.514 | 0.483 / 0.598 | 0.593 / 0.691 | 0.705 / 0.782 | 0.764 / 0.827 | 0.822 / 0.873 | 0.891 / 0.923 |
| TSB | 0.380 / 0.567 | 0.467 / 0.648 | 0.570 / 0.729 | 0.671 / 0.809 | 0.726 / 0.849 | 0.785 / 0.890 | 0.848 / 0.931 |
| LGBM-Q | 0.530 / 0.532 | 0.619 / 0.620 | 0.711 / 0.711 | 0.811 / 0.811 | 0.858 / 0.858 | 0.905 / 0.905 | 0.948 / 0.948 |
| LGBM-PoisNB | 0.402 / 0.575 | 0.500 / 0.658 | 0.617 / 0.744 | 0.754 / 0.829 | 0.810 / 0.869 | 0.866 / 0.908 | 0.920 / 0.947 |
| MLP-NB | 0.393 / 0.594 | 0.480 / 0.670 | 0.575 / 0.747 | 0.676 / 0.824 | 0.737 / 0.863 | 0.817 / 0.903 | 0.908 / 0.946 |

test, H=3:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.388 / 0.569 | 0.478 / 0.649 | 0.581 / 0.730 | 0.689 / 0.809 | 0.743 / 0.850 | 0.802 / 0.891 | 0.867 / 0.932 |
| SeasonalNaive | 0.375 / 0.552 | 0.420 / 0.594 | 0.510 / 0.666 | 0.596 / 0.736 | 0.635 / 0.768 | 0.676 / 0.797 | 0.732 / 0.835 |
| CrostonSBA | 0.364 / 0.516 | 0.464 / 0.605 | 0.571 / 0.696 | 0.692 / 0.788 | 0.756 / 0.835 | 0.820 / 0.880 | 0.889 / 0.928 |
| TSB | 0.388 / 0.575 | 0.487 / 0.661 | 0.593 / 0.741 | 0.704 / 0.819 | 0.762 / 0.860 | 0.823 / 0.902 | 0.890 / 0.941 |
| LGBM-Q | 0.509 / 0.513 | 0.606 / 0.607 | 0.706 / 0.706 | 0.806 / 0.806 | 0.855 / 0.855 | 0.903 / 0.903 | 0.948 / 0.948 |
| LGBM-PoisNB | 0.408 / 0.587 | 0.513 / 0.670 | 0.626 / 0.755 | 0.751 / 0.840 | 0.812 / 0.879 | 0.868 / 0.918 | 0.926 / 0.953 |
| MLP-NB | 0.403 / 0.600 | 0.499 / 0.679 | 0.599 / 0.757 | 0.706 / 0.836 | 0.769 / 0.874 | 0.846 / 0.911 | 0.917 / 0.949 |

test, H=4:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| WindowAverage | 0.394 / 0.548 | 0.490 / 0.636 | 0.588 / 0.720 | 0.695 / 0.801 | 0.747 / 0.842 | 0.804 / 0.884 | 0.865 / 0.925 |
| SeasonalNaive | 0.382 / 0.534 | 0.438 / 0.585 | 0.525 / 0.666 | 0.614 / 0.738 | 0.660 / 0.773 | 0.703 / 0.807 | 0.760 / 0.851 |
| CrostonSBA | 0.366 / 0.495 | 0.464 / 0.585 | 0.577 / 0.684 | 0.698 / 0.782 | 0.760 / 0.828 | 0.821 / 0.875 | 0.889 / 0.923 |
| TSB | 0.402 / 0.561 | 0.497 / 0.646 | 0.600 / 0.731 | 0.711 / 0.812 | 0.768 / 0.855 | 0.828 / 0.895 | 0.891 / 0.936 |
| LGBM-Q | 0.509 / 0.511 | 0.608 / 0.608 | 0.706 / 0.706 | 0.806 / 0.806 | 0.855 / 0.855 | 0.899 / 0.899 | 0.949 / 0.949 |
| LGBM-PoisNB | 0.427 / 0.572 | 0.534 / 0.662 | 0.647 / 0.750 | 0.763 / 0.835 | 0.816 / 0.875 | 0.873 / 0.914 | 0.927 / 0.951 |
| MLP-NB | 0.409 / 0.579 | 0.502 / 0.662 | 0.606 / 0.743 | 0.715 / 0.822 | 0.780 / 0.862 | 0.846 / 0.904 | 0.918 / 0.945 |

(Bảng dài đầy đủ: `results/hieu_chinh_bang.csv`.)

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
| TSB | 1.0382 | 1.2372 | 1.0772 | 1.2991 |
| CrostonSBA | 1.2526 | 1.5195 | 1.2631 | 1.5381 |

**Diễn giải lại**: bỏ giá/SNAP/sự kiện chỉ làm pinball_avg tăng tối đa 0.84% (xem cột tang_tuong_doi), trong khi LightGBM-Q không ngoại sinh vẫn hơn hẳn TSB (test H=3: 0.9385 vs 1.0772) và CrostonSBA. Do đó lời giải thích ở báo cáo trước ("lợi thế đến từ biến ngoại sinh giá/SNAP/sự kiện") KHÔNG được dữ liệu ủng hộ và được rút lại. Lợi thế của các mô hình học máy so với Croston/TSB đến từ các phần còn lại: đặc trưng lịch sử đa thang (trung bình/độ lệch chuẩn 7–180 ngày, tỷ lệ ngày có bán, số ngày kể từ lần bán cuối, tuổi mặt hàng), doanh số cùng thứ của các tuần trước, thành phần thứ trong cửa sổ (EDA: Chủ nhật 4.1099 vs Thứ 4 2.5597), và việc huấn luyện trực tiếp phân vị/phân phối của D_H. Ablation này chưa tách riêng yếu tố lịch (thứ/tháng vẫn còn trong bản "no_exog") nên chưa kết luận được trọng số từng yếu tố. (GĐ7 bổ sung: bỏ riêng hai cột mã hóa mục tiêu fut_wd_index, fut_event_index — tác động không đáng kể, xem `07`.)

## 7. MLP-NB ba seed và hội tụ (mục 6)
Mỗi seed: huấn luyện trên train (epoch chọn bằng holdout nội bộ), kappa chọn trên val, đánh giá val; huấn luyện lại trên train+val với epoch/kappa đó và đánh giá test. Trung bình ± độ lệch chuẩn (ddof=1, n=3) của pinball_avg, WAPE, MASE:
| split | H | n_seed | pinball_avg | wape | mase | epochs | kappa |
|---|---|---|---|---|---|---|---|
| test | 3 | 3 | 0.9460 ± 0.0024 | 0.3515 ± 0.0004 | 0.7843 ± 0.0005 | 17/25/20 | 0.5/0.6/0.6 |
| test | 4 | 3 | 1.1433 ± 0.0058 | 0.3170 ± 0.0006 | 0.8031 ± 0.0032 | 30/16/26 | 0.5/0.6/0.6 |
| val | 3 | 3 | 0.9023 ± 0.0008 | 0.3680 ± 0.0029 | 0.7559 ± 0.0043 | 17/25/20 | 0.5/0.6/0.6 |
| val | 4 | 3 | 1.0897 ± 0.0039 | 0.3315 ± 0.0022 | 0.7748 ± 0.0043 | 30/16/26 | 0.5/0.6/0.6 |

Chi tiết từng seed:
| seed | H | split | kappa | epochs | pinball_avg | wape | mase |
|---|---|---|---|---|---|---|---|
| 42 | 3 | val | 0.5000 | 17 | 0.9025 | 0.3701 | 0.7597 |
| 42 | 4 | val | 0.5000 | 30 | 1.0929 | 0.3328 | 0.7746 |
| 42 | 3 | test | 0.5000 | 17 | 0.9477 | 0.3512 | 0.7839 |
| 42 | 4 | test | 0.5000 | 30 | 1.1489 | 0.3165 | 0.8024 |
| 43 | 3 | val | 0.6000 | 25 | 0.9014 | 0.3646 | 0.7513 |
| 43 | 4 | val | 0.6000 | 16 | 1.0909 | 0.3327 | 0.7792 |
| 43 | 3 | test | 0.6000 | 25 | 0.9471 | 0.3520 | 0.7849 |
| 43 | 4 | test | 0.6000 | 16 | 1.1435 | 0.3171 | 0.8002 |
| 44 | 3 | val | 0.6000 | 20 | 0.9029 | 0.3693 | 0.7567 |
| 44 | 4 | val | 0.6000 | 26 | 1.0853 | 0.3290 | 0.7706 |
| 44 | 3 | test | 0.6000 | 20 | 0.9432 | 0.3512 | 0.7842 |
| 44 | 4 | test | 0.6000 | 26 | 1.1374 | 0.3176 | 0.8066 |

Seed 42 tái hiện đúng số của lần chạy chính (val 0.9025 và 1.0929; test 0.9477 và 1.1489, pinball_avg H=3 và H=4). Độ biến thiên theo seed nhỏ (độ lệch chuẩn lớn nhất 0.0058), nhỏ hơn nhiều chênh lệch giữa MLP-NB và Croston/TSB. So với LGBM-Q: val H=3: MLP-NB 0.9014–0.9029 vs LGBM-Q 0.9109 → cả 3 seed tốt hơn LGBM-Q; val H=4: MLP-NB 1.0853–1.0929 vs LGBM-Q 1.0922 → lẫn lộn (một số seed tốt hơn, một số kém hơn); test H=3: MLP-NB 0.9432–0.9477 vs LGBM-Q 0.9354 → cả 3 seed kém hơn LGBM-Q; test H=4: MLP-NB 1.1374–1.1489 vs LGBM-Q 1.1344 → cả 3 seed kém hơn LGBM-Q. Vậy ở H=3 thứ hạng lật val→test ổn định theo seed; ở H=4 trên val MLP-NB và LGBM-Q gần như ngang nhau (lẫn lộn giữa các seed).
Hội tụ (NLL holdout nội bộ; seed 42; chạy tối đa 80 epoch; `results/figures/mlp_duong_nll.png`):
| H | epoch_tot_nhat(<=80) | NLL_tai_epoch_30 | NLL_tot_nhat | NLL_epoch_80 |
|---|---|---|---|---|
| 3.0000 | 58.0000 | 2.1620 | 2.1591 | 2.1621 |
| 4.0000 | 46.0000 | 2.3424 | 2.3414 | 2.3448 |

Epoch tốt nhất khi cho chạy tới 80 là 58 (H=3) và 46 (H=4); đường NLL gần như phẳng (H=4: NLL ở epoch 30 hơn NLL tốt nhất 0.0010, ≈0.04%) rồi tăng nhẹ ở epoch 80 (quá khớp). Kết luận: H=4 với trần 30 epoch chưa đạt đúng cực tiểu nhưng đã nằm trên đoạn phẳng; chưa chạy lại pinball với epoch tối ưu nên chưa định lượng tác động lên pinball (dự kiến nhỏ).

## 8. MASE trong bảng chính (mục 7)
MASE (trung bình trên chuỗi của MAE theo cửa sổ H / MAE naive trong mẫu theo cửa sổ H, scale CHỈ tính trên train, kiểm bằng `test_mase_scale_uses_train_only_and_equals_hand_calculation`). Tổng thể 823 chuỗi, theo mô hình, không phụ thuộc tỷ lệ chi phí/L/q (đánh giá dự báo cửa sổ). H=3:
| model | pinball_avg_val_H3 | wape_val_H3 | mase_val_H3 | pinball_avg_test_H3 | wape_test_H3 | mase_test_H3 |
|---|---|---|---|---|---|---|
| WindowAverage | 1.0757 | 0.4129 | 0.8125 | 1.1316 | 0.3987 | 0.8474 |
| SeasonalNaive | 1.4002 | 0.4741 | 0.9645 | 1.4112 | 0.4399 | 0.9882 |
| CrostonSBA | 1.2526 | 0.4948 | 0.9502 | 1.2631 | 0.4484 | 0.9174 |
| TSB | 1.0382 | 0.4058 | 0.8009 | 1.0772 | 0.3898 | 0.8287 |
| LGBM-Q | 0.9109 | 0.3697 | 0.7672 | 0.9354 | 0.3508 | 0.7870 |
| LGBM-PoisNB | 0.9300 | 0.3697 | 0.7672 | 0.9490 | 0.3508 | 0.7870 |
| MLP-NB | 0.9025 | 0.3701 | 0.7597 | 0.9477 | 0.3512 | 0.7839 |

H=4:
| model | pinball_avg_val_H4 | wape_val_H4 | mase_val_H4 | pinball_avg_test_H4 | wape_test_H4 | mase_test_H4 |
|---|---|---|---|---|---|---|
| WindowAverage | 1.2918 | 0.3682 | 0.8320 | 1.3782 | 0.3575 | 0.8677 |
| SeasonalNaive | 1.6261 | 0.4214 | 0.9792 | 1.6239 | 0.3859 | 0.9882 |
| CrostonSBA | 1.5195 | 0.4506 | 0.9886 | 1.5381 | 0.4060 | 0.9445 |
| TSB | 1.2372 | 0.3605 | 0.8173 | 1.2991 | 0.3474 | 0.8444 |
| LGBM-Q | 1.0922 | 0.3333 | 0.7884 | 1.1344 | 0.3166 | 0.8051 |
| LGBM-PoisNB | 1.1192 | 0.3333 | 0.7884 | 1.1506 | 0.3166 | 0.8051 |
| MLP-NB | 1.0929 | 0.3328 | 0.7746 | 1.1489 | 0.3165 | 0.8024 |


## 9. Chú thích bắt buộc cho bảng theo nhóm (mục 8)
Đã bổ sung chú thích tỷ lệ chi phí, L và q cho các bảng theo nhóm nhanh/chậm trong `02`, `03`, `04`, `05` và `bao_cao_cuoi_cung.md`. Quy ước: bảng sai số dự báo theo nhóm (pinball_avg, WAPE, MASE) tính trên dự báo cửa sổ H=3 và không phụ thuộc tỷ lệ chi phí/L/q (pinball_avg là trung bình 9 mức q); bảng mô phỏng theo nhóm ghi rõ cost_ratio, L và q (hoặc "q*" kèm bảng q*).

## 10. Cấu hình
Seed 42 (MLP seed 42/43/44). k_grid 0.5–3.0 bước 0.1; lưới q mở rộng [0.3, 0.4]; bootstrap 2000 lần seed 42. lightgbm 4.7.0, torch 2.15.0+cpu, scipy 1.18.1, numpy 2.5.4, pandas 2.3.3. Thời gian: chạy lại val/test ≈ 9.5 + 4.5 phút, ablation ≈ 4.5 phút, mlp_seeds ≈ 16 phút (CPU 12 luồng).

## 11. Vấn đề và hạn chế
- q* và k vẫn chạm cận dưới lưới ở tỷ lệ 1–2 (q*=0.3; k*=0.5).
- Mọi lựa chọn (k, q*) trên val, báo cáo test; KTC bootstrap coi chuỗi độc lập; chỉ 28 ngày.
- MLP-NB H=4 chưa chạy với epoch tối ưu 46; ablation chưa tách yếu tố lịch; chưa có đối chứng "Croston/TSB + biến ngoại sinh".
- Kết quả chính của GĐ2–5 không đổi (đã kiểm chứng), nhưng các kết luận diễn giải ở câu 1 và câu 3 được sửa như trên.

## 12. Chưa làm
Chạy lại MLP-NB H=4 với 46 epoch; k theo nhóm; lưới q xuống dưới 0.3; ablation lịch.
