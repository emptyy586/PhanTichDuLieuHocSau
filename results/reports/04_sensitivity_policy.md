# Báo cáo GĐ4 – Độ nhạy, q* và chính sách bổ sung đề xuất (validation)

> **Lưu ý GĐ7**: số liệu của báo cáo này là ảnh chụp tại GĐ4 (trước làm tròn LGBM-Q và lưới kappa mới); q*, bảng chính sách và các so sánh đã thay đổi một phần. Dùng `05_final_test.md`, `06_bo_sung.md`, `07_sua_loi.md`, `final_report.md` cho số liệu hiện hành.

## 1. Đã làm gì
- `src/policy_analysis.py`: bảng q* theo (mô hình, L, cost_ratio, nhóm), bảng chính sách theo nhóm, nội suy "cùng fill rate", bootstrap theo chuỗi (2000 lần, seed 42, có hoàn lại theo chuỗi) và 4 biểu đồ tiếng Việt. Chạy: `python -m src.policy_analysis` (≈14 s).
- Không sinh thêm mô phỏng: dùng `results/inventory_sim_val.csv` của GĐ3 (đã chứa cost_ratio ∈ {1,2,5,10,20,50} × L ∈ {2,3}); bootstrap mô phỏng lại từng chuỗi từ `results/forecasts/val.npz`.
- Chưa thêm test mới ở GĐ4 (mã phân tích dùng lại hàm đã có test ở GĐ2–3); các test hiện có vẫn pass (mục 2).

## 2. Lệnh đã chạy và pytest
`python -m src.policy_analysis`, `python -m pytest -v`:
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

============================= 31 passed in 14.28s =============================
```

## 3. Kết quả
### 3.1 q* theo tỷ lệ chi phí thiếu/giữ (tổng thể, 823 chuỗi)
**Cảnh báo giới hạn lưới**: lưới q bắt đầu từ 0.5; ở tỷ lệ 1–2 (và 5 với nhiều mô hình) q*=0.5 là CẬN DƯỚI của lưới, nên q* thật có thể thấp hơn 0.5 và các giá trị này chỉ nên đọc là "≤ 0.5".
q*, L=2:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5000 | 0.5000 | 0.6000 | 0.8000 | 0.9000 | 0.9750 |
| SeasonalNaive | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |
| CrostonSBA | 0.5000 | 0.5000 | 0.6000 | 0.8000 | 0.9000 | 0.9750 |
| TSB | 0.5000 | 0.5000 | 0.6000 | 0.8000 | 0.9000 | 0.9750 |
| LGBM-Q | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |
| LGBM-PoisNB | 0.5000 | 0.5000 | 0.6000 | 0.8000 | 0.9000 | 0.9500 |
| MLP-NB | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9500 |

q*, L=3:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |
| SeasonalNaive | 0.5000 | 0.5000 | 0.5000 | 0.6000 | 0.8000 | 0.9000 |
| CrostonSBA | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.9000 | 0.9500 |
| TSB | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |
| LGBM-Q | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.8000 | 0.9000 |
| LGBM-PoisNB | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |
| MLP-NB | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.8500 | 0.9500 |

Thay đổi tương đối của tổng chi phí khi dùng q* so với chính sách điểm (cost_q*/cost_điểm − 1; âm = rẻ hơn), L=2:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -0.1106 | -0.0728 | 0.0004 | -0.0674 | -0.2552 | -0.5209 |
| SeasonalNaive | -0.0732 | -0.0559 | -0.0197 | -0.0207 | -0.1650 | -0.4214 |
| CrostonSBA | -0.1245 | -0.0814 | -0.0013 | -0.0759 | -0.2722 | -0.5556 |
| TSB | -0.1174 | -0.0817 | -0.0090 | -0.0729 | -0.2586 | -0.5351 |
| LGBM-Q | -0.0506 | -0.0377 | -0.0102 | -0.0733 | -0.2550 | -0.5381 |
| LGBM-PoisNB | -0.1179 | -0.0828 | -0.0125 | -0.0712 | -0.2526 | -0.5288 |
| MLP-NB | -0.1077 | -0.0789 | -0.0153 | -0.0546 | -0.2226 | -0.5008 |

L=3:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -0.0978 | -0.0698 | -0.0102 | -0.0336 | -0.1937 | -0.4615 |
| SeasonalNaive | -0.0696 | -0.0558 | -0.0255 | -0.0065 | -0.1199 | -0.3667 |
| CrostonSBA | -0.1153 | -0.0819 | -0.0125 | -0.0335 | -0.2081 | -0.4891 |
| TSB | -0.1060 | -0.0793 | -0.0210 | -0.0354 | -0.1950 | -0.4707 |
| LGBM-Q | -0.0432 | -0.0338 | -0.0130 | -0.0439 | -0.1999 | -0.4797 |
| LGBM-PoisNB | -0.1076 | -0.0808 | -0.0216 | -0.0434 | -0.1990 | -0.4728 |
| MLP-NB | -0.0992 | -0.0768 | -0.0259 | -0.0328 | -0.1718 | -0.4442 |

Mô hình có chi phí thấp nhất tại q* theo từng tỷ lệ (tổng thể) và chính sách điểm tốt nhất:
| L | cost_ratio | mo hinh tot nhat (q*) | q* | cost_q* | fill_q* | diem tot nhat | cost_diem |
|---|---|---|---|---|---|---|---|
| 2 | 1 | WindowAverage | 0.5000 | 124.3536 | 0.7214 | WindowAverage | 139.8141 |
| 2 | 2 | WindowAverage | 0.5000 | 145.2236 | 0.7214 | WindowAverage | 156.6258 |
| 2 | 5 | MLP-NB | 0.5000 | 198.0267 | 0.7810 | MLP-NB | 201.1069 |
| 2 | 10 | MLP-NB | 0.8000 | 252.4508 | 0.9062 | MLP-NB | 267.0243 |
| 2 | 20 | MLP-NB | 0.9000 | 310.0838 | 0.9462 | MLP-NB | 398.8591 |
| 2 | 50 | MLP-NB | 0.9500 | 396.5832 | 0.9685 | MLP-NB | 794.3633 |
| 3 | 1 | WindowAverage | 0.5000 | 141.4714 | 0.7440 | WindowAverage | 156.8044 |
| 3 | 2 | LGBM-PoisNB | 0.5000 | 159.7315 | 0.7694 | WindowAverage | 172.7084 |
| 3 | 5 | MLP-NB | 0.5000 | 209.0510 | 0.7922 | MLP-NB | 214.6100 |
| 3 | 10 | MLP-NB | 0.7000 | 270.0267 | 0.8695 | MLP-NB | 279.1786 |
| 3 | 20 | MLP-NB | 0.8500 | 338.1616 | 0.9237 | MLP-NB | 408.3159 |
| 3 | 50 | MLP-NB | 0.9500 | 442.2819 | 0.9646 | MLP-NB | 795.7278 |

(Quy tắc newsvendor ratio/(1+ratio) làm tròn lên lưới q: 1→0.5, 2→0.7, 5→0.85, 10→0.9, 20→0.95, 50→0.975.)

### 3.2 q* theo nhóm (L=2)
MLP-NB theo speed [q* của MLP-NB, L=2; cột = tỷ lệ chi phí thiếu/giữ]:
| group | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| fast | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9500 |
| medium | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9750 |
| slow | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9500 |

TSB theo speed [q* của TSB, L=2; cột = tỷ lệ chi phí thiếu/giữ]:
| group | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| fast | 0.5000 | 0.5000 | 0.6000 | 0.8000 | 0.9000 | 0.9750 |
| medium | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9750 |
| slow | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9500 |

MLP-NB theo demand_type [q* của MLP-NB, L=2; cột = tỷ lệ chi phí thiếu/giữ]:
| group | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| erratic | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9750 |
| intermittent | 0.5000 | 0.5000 | 0.6000 | 0.8000 | 0.9000 | 0.9500 |
| lumpy | 0.5000 | 0.5000 | 0.5000 | 0.7000 | 0.9000 | 0.9500 |
| smooth | 0.5000 | 0.5000 | 0.5000 | 0.8000 | 0.9000 | 0.9750 |

Sai số dự báo theo nhóm (GĐ2, H=3; không phụ thuộc tỷ lệ chi phí/L/q): pinball_avg (trung bình 9 mức q) theo speed:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 1.8014 | 0.7119 | 0.3950 |
| SeasonalNaive | 2.2183 | 0.9999 | 0.6150 |
| CrostonSBA | 2.1232 | 0.8135 | 0.4472 |
| TSB | 1.7472 | 0.6812 | 0.3835 |
| LGBM-Q | 1.4905 | 0.6237 | 0.3609 |
| LGBM-PoisNB | 1.5303 | 0.6310 | 0.3642 |
| MLP-NB | 1.4806 | 0.6143 | 0.3574 |

WAPE theo speed (H=3; không phụ thuộc tỷ lệ chi phí/L/q):
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 0.3616 | 0.5519 | 0.7006 |
| SeasonalNaive | 0.4048 | 0.6673 | 0.8418 |
| CrostonSBA | 0.4455 | 0.6231 | 0.7930 |
| TSB | 0.3554 | 0.5404 | 0.6964 |
| LGBM-Q | 0.3123 | 0.5220 | 0.7039 |
| LGBM-PoisNB | 0.3123 | 0.5220 | 0.7039 |
| MLP-NB | 0.3150 | 0.5169 | 0.6888 |

WAPE theo demand_type (H=3; không phụ thuộc tỷ lệ chi phí/L/q):
| model | erratic | intermittent | lumpy | smooth |
|---|---|---|---|---|
| WindowAverage | 0.4131 | 0.4491 | 0.5635 | 0.3083 |
| SeasonalNaive | 0.4240 | 0.5374 | 0.6461 | 0.3503 |
| CrostonSBA | 0.5001 | 0.5769 | 0.6972 | 0.3267 |
| TSB | 0.3945 | 0.4458 | 0.5469 | 0.3059 |
| LGBM-Q | 0.3341 | 0.4139 | 0.5191 | 0.2684 |
| LGBM-PoisNB | 0.3341 | 0.4139 | 0.5191 | 0.2684 |
| MLP-NB | 0.3321 | 0.4113 | 0.5176 | 0.2727 |


### 3.3 Bảng chính sách bổ sung đề xuất (cost_ratio=10, chọn (mô hình, q) tối thiểu hóa tổng chi phí trên val; nhóm <30 chuỗi dùng chính sách của nhóm speed)
Các giá trị fill rate/chi phí là KỲ VỌNG TRÊN VAL (chọn và đánh giá cùng một tập ⇒ lạc quan); GĐ5 đánh giá lại trên test. Chi phí = TB/chuỗi trong 28 ngày, đơn vị tương đối.
L=2 [cost_ratio=10; q = cột q_star của từng dòng]:
| group | n_series_group | model | q_star | fill_rate | stockout_days | avg_inventory | total_cost | ghi_chu |
|---|---|---|---|---|---|---|---|---|
| erratic_fast | 45 | LGBM-Q | 0.7000 | 0.8629 | 4.1556 | 10.2444 | 474.6222 |  |
| intermittent_fast | 74 | MLP-NB | 0.8000 | 0.9313 | 2.8784 | 9.5738 | 355.7703 |  |
| intermittent_medium | 203 | MLP-NB | 0.8000 | 0.8680 | 2.5665 | 3.5422 | 146.7685 |  |
| intermittent_slow | 142 | MLP-NB | 0.8000 | 0.8270 | 1.9085 | 1.9298 | 80.5141 |  |
| lumpy_fast | 89 | MLP-NB | 0.7000 | 0.8407 | 3.4382 | 7.8752 | 354.5506 |  |
| lumpy_medium | 116 | MLP-NB | 0.8000 | 0.8354 | 2.5172 | 3.9544 | 169.6897 |  |
| lumpy_slow | 34 | TSB | 0.7000 | 0.6966 | 2.6471 | 1.6155 | 99.9412 |  |
| smooth_fast | 118 | MLP-NB | 0.8000 | 0.9388 | 2.9068 | 14.7022 | 537.0000 |  |
| smooth_medium | 2 | MLP-NB | 0.8000 | 0.8568 | 2.5607 | 3.6943 | 155.2461 | dùng nhóm speed=medium |

L=3 [cost_ratio=10; q = cột q_star của từng dòng]:
| group | n_series_group | model | q_star | fill_rate | stockout_days | avg_inventory | total_cost | ghi_chu |
|---|---|---|---|---|---|---|---|---|
| erratic_fast | 45 | LGBM-Q | 0.7000 | 0.8593 | 4.3778 | 11.7952 | 522.9333 |  |
| intermittent_fast | 74 | MLP-NB | 0.7000 | 0.8995 | 4.0405 | 9.0763 | 382.5135 |  |
| intermittent_medium | 203 | MLP-NB | 0.8000 | 0.8717 | 2.3399 | 3.9745 | 157.5419 |  |
| intermittent_slow | 142 | MLP-NB | 0.7000 | 0.7584 | 2.6479 | 1.6283 | 82.5634 |  |
| lumpy_fast | 89 | MLP-NB | 0.6000 | 0.8092 | 4.1011 | 7.6982 | 376.1124 |  |
| lumpy_medium | 116 | MLP-NB | 0.7000 | 0.7842 | 3.3190 | 3.6068 | 178.3190 |  |
| lumpy_slow | 34 | TSB | 0.7000 | 0.7276 | 2.5000 | 1.9653 | 104.1471 |  |
| smooth_fast | 118 | MLP-NB | 0.7000 | 0.9122 | 4.0085 | 14.2043 | 577.4661 |  |
| smooth_medium | 2 | MLP-NB | 0.7000 | 0.8052 | 3.3801 | 3.3935 | 165.4860 | dùng nhóm speed=medium |

Biểu đồ: `results/figures/policy_tradeoff_fill_vs_inventory.png` (Hình 1), `policy_total_cost_by_q.png` (Hình 2), `policy_qstar_heatmap.png` (Hình 3), `policy_forecast_error_by_group.png` (Hình 4).

## 4. Trả lời ba câu hỏi nghiên cứu
Mọi KTC 95% dưới đây là bootstrap theo chuỗi trên validation (đơn vị: chi phí tương đối TB/chuỗi; chênh lệch ÂM = phương án thứ nhất rẻ hơn). Chúng chỉ phản ánh sai số lấy mẫu theo chuỗi trong cùng 28 ngày, KHÔNG phản ánh biến động giữa các giai đoạn khác.

### Câu 1 – Dự báo phân vị có chọn mức tồn kho tốt hơn dự báo điểm không?
**Kết luận: có điều kiện, và phần lớn lợi ích nằm ở khả năng chọn mức phục vụ chứ không phải ở việc phân vị "tốt hơn" khi so ở cùng mức phục vụ.**
- Dùng q* chọn theo chi phí (tối thiểu hóa trên val) rẻ hơn chính sách điểm của cùng mô hình ở mọi tỷ lệ (bảng 3.1): ví dụ MLP-NB L=2 giảm −0.108 (tỷ lệ 1), −0.015 (5), −0.055 (10), −0.223 (20), −0.501 (50). Ở tỷ lệ 5 lợi ích gần 0 (0.000…−0.020). Lưu ý q* được tinh chỉnh còn chính sách điểm không có tham số nào để chỉnh ⇒ so sánh thiên vị có lợi cho phân vị.
- So sánh KHÔNG tinh chỉnh (q = quy tắc newsvendor, bootstrap, L=2; hiệu = chi phí[phân vị] − chi phí[điểm]; [KTC 95%]):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -15.46 [-16.26; -14.72] | 15.29 [13.47; 17.38] | 29.53 [26.00; 33.50] | -5.69 [-11.82; 0.33] | -105.91 [-122.92; -89.78] | -501.98 [-558.43; -450.16] |
| SeasonalNaive | -11.47 [-12.53; -10.51] | 32.20 [29.84; 35.03] | 65.60 [60.43; 71.56] | 42.63 [36.28; 49.92] | -32.94 [-47.19; -17.96] | -399.78 [-447.59; -354.78] |
| CrostonSBA | -18.47 [-19.73; -17.35] | 15.91 [13.71; 18.37] | 30.57 [25.74; 36.04] | -12.25 [-20.86; -3.47] | -135.98 [-157.59; -115.21] | -619.82 [-689.86; -554.94] |
| TSB | -16.90 [-17.55; -16.30] | 13.54 [11.74; 15.62] | 27.09 [23.95; 30.62] | -8.10 [-14.03; -2.30] | -108.72 [-126.53; -92.21] | -504.20 [-563.56; -449.12] |
| LGBM-Q | -7.27 [-8.12; -6.47] | 24.60 [22.81; 26.66] | 45.15 [41.38; 49.36] | 14.55 [8.81; 20.63] | -78.02 [-92.31; -63.09] | -447.55 [-496.50; -400.11] |
| LGBM-PoisNB | -16.94 [-17.66; -16.25] | 13.39 [11.71; 15.44] | 30.39 [27.04; 34.35] | -1.03 [-6.33; 4.56] | -90.05 [-103.81; -76.45] | -450.17 [-497.10; -405.89] |
| MLP-NB | -15.97 [-16.63; -15.35] | 12.75 [10.98; 14.80] | 28.50 [25.21; 32.24] | 2.76 [-2.09; 7.59] | -72.96 [-85.03; -60.36] | -395.32 [-434.49; -355.23] |

  Tỷ lệ 2 (q=0.7): phân vị đắt hơn điểm ở mọi mô hình (KTC không chứa 0). Tỷ lệ 10 (q=0.9): hỗn hợp – CrostonSBA/TSB rẻ hơn rõ (KTC âm), WindowAverage biên (KTC chứa 0 sát biên), LGBM-PoisNB và MLP-NB không khác biệt (KTC chứa 0), LGBM-Q và SeasonalNaive đắt hơn. Tỷ lệ 50 (q=0.975): phân vị rẻ hơn rất nhiều ở mọi mô hình (−395…−620).
  Tách theo speed (tỷ lệ chi phí 10, L=2, q=0.9 = q newsvendor; hiệu chi phí phân vị − điểm):
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | -16.23 [-31.08; -1.40] | -0.11 [-4.46; 4.13] | 3.67 [0.34; 7.16] |
| SeasonalNaive | 41.57 [26.15; 57.50] | 46.05 [38.97; 53.47] | 38.35 [33.24; 43.42] |
| CrostonSBA | -25.42 [-47.03; -3.46] | -6.31 [-11.82; -1.23] | 1.32 [-2.62; 5.11] |
| TSB | -25.46 [-39.34; -12.03] | 1.43 [-2.70; 5.38] | 6.70 [3.18; 10.27] |
| LGBM-Q | 18.08 [4.06; 32.38] | 11.53 [5.34; 17.41] | 13.52 [9.06; 17.75] |
| LGBM-PoisNB | -5.60 [-18.33; 6.80] | 0.71 [-4.46; 5.79] | 4.27 [0.82; 7.52] |
| MLP-NB | 4.83 [-5.25; 15.49] | 0.64 [-4.30; 5.34] | 2.77 [-0.78; 6.29] |

- So ở CÙNG fill rate (nội suy trên đường q; tồn kho TB/chuỗi):
| model | inv_quantile_same_fill|fast | inv_quantile_same_fill|overall | inv_quantile_same_fill|slow | inv_point|fast | inv_point|overall | inv_point|slow |
|---|---|---|---|---|---|---|
| WindowAverage | 7.9738 | 4.4015 | 1.3418 | 7.9489 | 4.3929 | 1.3377 |
| SeasonalNaive | 8.8316 | 4.9737 | 1.5759 | 8.8092 | 4.9798 | 1.5641 |
| CrostonSBA | 8.4575 | 4.5969 | 1.4099 | 8.4395 | 4.5915 | 1.3811 |
| TSB | 8.0728 | 4.4900 | 1.5049 | 8.0766 | 4.5556 | 1.5795 |
| LGBM-Q | 8.1462 | 4.5602 | 1.5365 | 8.2328 | 4.6087 | 1.5542 |
| LGBM-PoisNB | 8.1904 | 4.5184 | 1.4452 | 8.2328 | 4.6087 | 1.5542 |
| MLP-NB | 8.6451 | 4.7487 | 1.4754 | 8.6873 | 4.8282 | 1.5852 |

  Tồn kho cần để đạt đúng fill rate của chính sách điểm gần như bằng tồn kho của chính sách điểm (chênh nhỏ, ví dụ TSB tổng thể 4.4900 vs 4.5560; MLP-NB 4.7490 vs 4.8280; WindowAverage 4.4010 vs 4.3930): trên mặt phẳng fill rate–tồn kho chính sách điểm nằm gần như TRÊN đường phân vị (Hình 1, ô vuông). Phân vị cho lợi thế thực chất khi cần mức phục vụ cao hơn nhiều so với điểm (tỷ lệ chi phí lớn), vì chính sách điểm cố định ở fill rate ≈ 0.74–0.82 (tổng thể).
- Độ chắc chắn/hạn chế: chọn q* và so sánh cùng tập val; chỉ 28 ngày; chi phí thiếu/giữ là giả định tương đối; ở tỷ lệ ≤5 q* chạm cận dưới lưới.

### Câu 2 – Hàng bán nhanh vs hàng bán chậm/gián đoạn
- Sai số dự báo tương đối lớn hơn nhiều ở hàng chậm: WAPE (H=3) theo speed fast/slow, ví dụ MLP-NB 0.3150 vs 0.6888, TSB 0.3554 vs 0.6964 (bảng 3.2); pinball tuyệt đối lại lớn hơn ở hàng nhanh (do thang đo đơn vị lớn hơn).
- q* tại tỷ lệ 10 (L=2) của MLP-NB: nhanh 0.80, trung bình 0.80, chậm 0.80; fill rate tại q* nhanh 0.9236 so với chậm 0.8169. TSB: q* nhanh 0.80, chậm 0.80. Tức tại cùng giả định chi phí, tối ưu ở hàng chậm/gián đoạn là chấp nhận fill rate thấp hơn (giữ ít hàng vì mỗi đơn vị tồn tốn chi phí cả ngày trong khi nhu cầu hiếm), còn hàng nhanh/smooth nên giữ mức phục vụ cao hơn (bảng chính sách: smooth_fast fill 0.9388, intermittent_slow 0.8270, lumpy_slow 0.6966 ở L=2).
- Lợi ích của phân vị so với điểm không đồng đều: với TSB/CrostonSBA ở q=0.9, tỷ lệ 10, phân vị rẻ hơn rõ ở hàng nhanh (−25.46 và −25.42; KTC âm) nhưng KHÔNG rẻ hơn (TSB +6.70 [3.18; 10.27]) hoặc không khác biệt (Croston +1.32) ở hàng chậm.
- Hạn chế: nhóm lumpy_slow chỉ 34 chuỗi và smooth_medium 2 chuỗi (đã dùng fallback); q* ở nhóm chậm nhạy với lưới q thô.

### Câu 3 – Học sâu xác suất có hơn Croston/TSB không?
**Kết luận: MLP-NB hơn Croston/TSB rõ rệt trên val, nhưng KHÔNG hơn LightGBM, và "học sâu" ở đây chỉ là MLP tiêu chuẩn có biến ngoại sinh – phần lớn lợi thế nhiều khả năng đến từ đặc trưng (giá, SNAP, sự kiện, mùa vụ tuần), không phải từ kiến trúc sâu.**
- Dự báo (pinball_avg H=3, MLP-NB − đối thủ, theo chuỗi; [KTC 95%]):
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: pinball_avg MLP-NB - CrostonSBA (H=3) | -0.3514 [-0.4240; -0.2878] | -0.6426 [-0.8171; -0.4860] | -0.1992 [-0.2554; -0.1504] | -0.0897 [-0.1278; -0.0604] |
| Q3: pinball_avg MLP-NB - LGBM-Q (H=3) | -0.0083 [-0.0185; 0.0054] | -0.0099 [-0.0331; 0.0218] | -0.0094 [-0.0155; -0.0036] | -0.0035 [-0.0096; 0.0035] |
| Q3: pinball_avg MLP-NB - TSB (H=3) | -0.1373 [-0.1663; -0.1106] | -0.2665 [-0.3340; -0.2068] | -0.0669 [-0.0816; -0.0528] | -0.0261 [-0.0349; -0.0173] |

  MLP-NB < TSB và < CrostonSBA ở mọi nhóm speed (KTC âm). So với LGBM-Q: chênh −0.0083 tổng thể, KTC chứa 0 ⇒ không phân biệt được.
- Chi phí mô phỏng (cùng q newsvendor, tỷ lệ 10, L=2):
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: MLP-NB - CrostonSBA (cung q_newsvendor) | -43.92 [-57.58; -32.69] | -88.09 [-121.10; -59.96] | -19.22 [-27.29; -12.15] | -7.15 [-12.70; -2.45] |
| Q3: MLP-NB - LGBM-Q (cung q_newsvendor) | -20.86 [-23.75; -18.07] | -31.16 [-38.15; -24.59] | -14.70 [-16.92; -12.49] | -13.01 [-15.05; -11.10] |
| Q3: MLP-NB - TSB (cung q_newsvendor) | -12.63 [-16.28; -9.14] | -23.28 [-31.88; -14.65] | -6.75 [-9.61; -3.96] | -3.65 [-6.19; -1.08] |

  MLP-NB rẻ hơn TSB và CrostonSBA (KTC âm ở mọi nhóm) và cả LGBM-Q (do cùng q nhưng mức hiệu chuẩn khác nhau, xem Hình 1: hai mô hình nằm ở các điểm khác nhau trên mặt phẳng fill–tồn kho); tại q* chi phí tốt nhất tổng thể ở tỷ lệ ≥5 vẫn thuộc MLP-NB (bảng 3.1 "WIN").
- Hạn chế: (i) không dùng DeepAR/N-HiTS/TFT (không cài); (ii) MLP-NB và LightGBM dùng cùng bộ 30 đặc trưng còn Croston/TSB chỉ dùng lịch sử doanh số nên so sánh gộp hai yếu tố (mô hình + biến ngoại sinh); chưa có đối chứng "Croston/TSB + biến ngoại sinh" hay MLP chỉ dùng lịch sử; (iii) siêu tham số chọn trên val; (iv) một seed, chưa kiểm tra độ ổn định theo seed.

## 5. File sinh ra
`results/q_star_val.csv`, `results/policy_table_val.csv`, `results/research_questions_val.csv`, `results/tables/equal_fill_inventory_val.csv`, 4 hình `results/figures/policy_*.png`.

## 6. Cấu hình
Seed 42 (bootstrap `default_rng(42)`, 2000 lần). cost_ratios [1,2,5,10,20,50], L ∈ {2,3}, base ratio 10, ngưỡng nhóm tối thiểu 30 chuỗi. Python 3.12, numpy/pandas/matplotlib như GĐ1–2.

## 7. Vấn đề / giả định chưa chắc chắn
- q* ở tỷ lệ thấp chạm cận dưới lưới (0.5).
- Chọn mô hình + q theo nhóm trên val rồi báo cáo kỳ vọng trên val ⇒ lạc quan; nhóm nhỏ dễ chọn theo nhiễu (đã dùng ngưỡng 30 chuỗi nhưng 34 chuỗi vẫn nhỏ).
- KTC bootstrap coi các chuỗi độc lập; thực tế các chuỗi cùng cửa hàng có tương quan theo ngày (SNAP, sự kiện) nên KTC có thể hẹp hơn thực tế.
- Chi phí thiếu/giữ là giả định; kết luận về q* và mức đánh đổi phụ thuộc trực tiếp vào chúng.
- Hình 3 chỉ vẽ L=2 (L=3 trong `q_star_val.csv`).

## 8. Chưa làm
Đánh giá trên test, so sánh val–test (GĐ5). Không làm lưới q mở rộng xuống <0.5 (sẽ cần chạy lại GĐ2–3).


> **Cập nhật GĐ6**: kết luận câu 1 và giải thích câu 3 (biến ngoại sinh) ở báo cáo này đã được sửa bởi `06_bo_sung.md` (so với điểm có hệ số an toàn; ablation ngoại sinh). q*=0.5 ở tỷ lệ ≤5 là cận lưới – xem lưới mở rộng ở 06.
