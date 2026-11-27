# Báo cáo GĐ2 – Mô hình dự báo xác suất (validation)

> **Lưu ý GĐ7**: báo cáo này là ảnh chụp tại GĐ2. Sau GĐ7 (lưới kappa mới) pinball_avg/coverage của CrostonSBA, TSB, LGBM-PoisNB thay đổi nhẹ (chữ số thập phân thứ 3–4; xem `07_sua_loi.md` mục 3.2); bản số liệu cũ ở `archive/pre_gd7/`.

## 1. Đã làm gì
**File mới**: `src/mo_hinh_xac_suat.py` (Panel, mức dự báo Croston-SBA/TSB/WindowAverage/SeasonalNaive vector hóa, NB chính xác bằng `scipy.stats.nbinom.ppf`, ước lượng alpha cuốn chiếu, chỉ số), `src/tinh_chinh.py` (chọn siêu tham số trên val), `src/mo_hinh_ml.py` (30 đặc trưng không rò rỉ, LightGBM phân vị + Poisson, MLP-NB bằng PyTorch), `src/chay_du_bao.py` (pipeline val/test), `src/tien_ich_bao_cao.py`; `tests/kiem_thu_mo_hinh_xac_suat.py` (11 test), `tests/kiem_thu_dau_ra.py` (4 test).
**File sửa** (theo mục D; hành vi phục vụ 7 test cũ giữ nguyên, 7 test cũ vẫn pass): `src/du_bao_xac_suat.py` – (a) `sample_total_window` dùng Gamma-Poisson (NB chính xác với mọi n>0, bỏ `max(n,1.0)`); (d) bỏ các cột trùng lặp `roll_mean_{lag}` trong `build_lag_features`; ghi chú *legacy* cho `rolling_origin_eval` (lỗi b: chỉ dùng lịch sử train, không phải rolling-origin thật — không dùng nữa). `configs/thong_so.yaml`: thêm `quantile_grid`, `lead_times`, `alpha_*`, lưới tuning, tham số ML/MLP (giữ nguyên `quantile_levels` cũ cho code legacy).

**Mục tiêu dự báo** D_H(t)=y[t+1]+…+y[t+H], H=L+R ∈ {3,4} (L=2,3; R=1). 26 gốc (H=3) và 25 gốc (H=4) trong 28 ngày val; gốc đầu tiên t=2016-03-27 chỉ dùng dữ liệu train, các gốc sau dùng thêm dữ liệu val đã quan sát đến t (rolling-origin thật; lỗi b).
**Mô hình (7)**: WindowAverage(28), SeasonalNaive(lag 7), CrostonSBA, TSB — mỗi cái bọc NB(mean=M, alpha cuốn chiếu); LightGBM phân vị trực tiếp trên D_H (1 mô hình/q, 9 q; phân vị được sắp tăng dần theo q để không cắt nhau); LightGBM-Poisson+NB (mean Poisson, alpha ước lượng từ phần dư của chính nó); MLP-NB (đầu ra NB(mu, alpha) huấn luyện bằng NLL trực tiếp trên D_H).
**Alpha NB (lỗi c)**: ước lượng moment alpha = Σ(e²−M)/ΣM² trên ≤180 gốc lịch sử đã quan sát xong (s ≤ t−H); các ngày trước lần bán đầu tiên đóng góp 0 (M=0, W=0); trộn với alpha của nhóm demand_type×speed theo w_series; nhân hệ số hiệu chuẩn kappa.
**Về mô hình học sâu (E4)**: KHÔNG dùng DeepAR/N-HiTS (neuralforecast/gluonts chưa cài và chỉ cho phân vị biên từng ngày; tổng các phân vị ngày KHÁC phân vị của tổng). Mô hình học sâu ở đây là **MLP-NB** (MLP 2 lớp ẩn {128, 64}, dropout 0.1, Adam lr 0.001, batch 2048), gọi đúng tên đó, không phải DeepAR/N-HiTS/TFT. Chronos-2 không làm (tùy chọn).
**WRMSSE: KHÔNG làm.** WRMSSE của M5 định nghĩa trên dự báo ngày 28 bước và 12 mức gộp theo trọng số doanh thu; ở đây chỉ có 1 cửa hàng × 1 phòng ban (chỉ có mức mặt hàng) và mô hình dự báo trực tiếp tổng cửa sổ 3–4 ngày, nên không tái hiện đúng công thức M5. Thay bằng WAPE (trên trung bình cửa sổ) và MASE (mẫu số = MAE trong mẫu train của dự báo naive W(s)≈W(s−H)).

## 2. Lệnh đã chạy và pytest
`python -m src.chay_du_bao --split val` (log `results/chay_val.log`), `python -m pytest -v`:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\lt\261\PTDLHS\DA\.venv\Scripts\python.exe
cachedir: .pytest_cache
Fugue tests will be initialized with options:
rootdir: D:\lt\261\PTDLHS\DA
plugins: anyio-4.15.1, fugue-0.9.7, platformdirs-4.14.0
collecting ... collected 31 items

tests/kiem_thu_kham_pha_du_lieu.py::test_load_train_excludes_val PASSED                   [  3%]
tests/kiem_thu_kham_pha_du_lieu.py::test_active_rows_drops_pre_launch_days PASSED         [  6%]
tests/kiem_thu_kham_pha_du_lieu.py::test_weekday_table_has_seven_rows_and_mean_two PASSED [  9%]
tests/kiem_thu_kham_pha_du_lieu.py::test_demand_speed_table_totals PASSED                 [ 12%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_daily_target_date_tracks_horizon_offset PASSED [ 16%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_evaluate_validation_summary_matches_by_date_and_id PASSED [ 19%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_join_rows_do_not_duplicate PASSED [ 22%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_n_samples_changes_quantile_output PASSED [ 25%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_pinball_known_answer PASSED [ 29%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_poisson_distribution_scaling PASSED [ 32%]
tests/kiem_thu_du_bao_xac_suat.py::ForecastProbEvaluationTest::test_sample_total_window_does_not_blow_up_for_zero_mean PASSED [ 35%]
tests/kiem_thu_mo_phong_ton_kho.py::test_constant_demand_matches_hand_calculation PASSED [ 38%]
tests/kiem_thu_mo_phong_ton_kho.py::test_inventory_conservation_every_day PASSED [ 41%]
tests/kiem_thu_mo_phong_ton_kho.py::test_costs_nonnegative_and_fill_rate_range_and_monotone_in_level PASSED [ 45%]
tests/kiem_thu_mo_phong_ton_kho.py::test_reproducible_same_inputs PASSED        [ 48%]
tests/kiem_thu_dau_ra.py::test_val_npz_shapes_cover_all_series_and_origins_and_monotone PASSED [ 51%]
tests/kiem_thu_dau_ra.py::test_val_eval_csv_schema_and_coverage_matches_npz PASSED [ 54%]
tests/kiem_thu_dau_ra.py::test_actual_in_npz_equals_window_sum_of_val_demand PASSED [ 58%]
tests/kiem_thu_dau_ra.py::test_md_table_rounds_to_four_decimals PASSED      [ 61%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_pinball_known_answer PASSED              [ 64%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_nb_quantiles_match_scipy_and_cdf_definition PASSED [ 67%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_nb_quantiles_monotone_in_q_and_zero_mean PASSED [ 70%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_legacy_sampler_mean_not_inflated_when_n_below_one PASSED [ 74%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_window_targets_values_and_nan_tail PASSED [ 77%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_origins_window_inside_split PASSED       [ 80%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_croston_tsb_constant_demand PASSED       [ 83%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_no_leak_stat_forecast_and_alpha PASSED   [ 87%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_no_leak_features PASSED                  [ 90%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_estimate_alpha_known_poisson_is_small PASSED [ 93%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_evaluate_forecast_coverage_and_shapes PASSED [ 96%]
tests/kiem_thu_mo_hinh_xac_suat.py::test_lgbm_quantiles_monotone_and_shape PASSED [100%]

============================= 31 passed in 3.70s ==============================
```

## 3. Kiểm chứng lỗi (a) và bảng coverage trước/sau
Lỗi (a) – lấy mẫu NB khi n<1 (chạy lại `sample_total_window`, 200000 mẫu; cột TRUOC là output đã chạy trước khi sửa):
| mean | var | mean_out_TRUOC_sua | mean_out_SAU_sua |
|---|---|---|---|
| 1.0000 | 25.0000 | 24.0079 | 0.9975 |
| 0.5000 | 5.0000 | 8.9914 | 0.4998 |

Coverage cửa sổ H=3 (L=2,R=1) trên val. TRUOC = `results/danh_gia_du_bao_val.csv` (code cũ, mức window); SAU = pipeline mới (WindowAverage+NB, kappa/w chọn trên val):
| q | TRUOC: legacy baseline_nb | SAU: WindowAverage+NB H=3 | SAU: P(y<Q) | lech TRUOC | lech SAU |
|---|---|---|---|---|---|
| 0.5000 | 0.6050 | 0.5850 | 0.3754 | 0.1050 | 0.0850 |
| 0.7000 | 0.7554 | 0.7341 | 0.5570 | 0.0554 | 0.0341 |
| 0.8000 | 0.8325 | 0.8115 | 0.6541 | 0.0325 | 0.0115 |
| 0.9000 | 0.9055 | 0.8897 | 0.7645 | 0.0055 | 0.0103 |
| 0.9500 | 0.9382 | 0.9302 | 0.8278 | 0.0118 | 0.0198 |
| 0.9900 | 0.9632 | 0.9678 | 0.8879 | 0.0268 | 0.0222 |

**Lưu ý về mục tiêu |coverage−q| ≤ 0.05 (E7)**: nhu cầu cửa sổ là số nguyên, nhiều chuỗi có P(D=0) lớn nên phân vị là số nguyên; coverage P(y≤Q) luôn ≥ q một khoảng bằng khối xác suất tại Q và KHÔNG thể ≈ q ở q thấp. Vì vậy báo cáo thêm coverage_lt = P(y<Q); mô hình hiệu chuẩn tốt khi coverage_lt ≤ q ≤ coverage. Đếm trên 7 mức q ≤ 0.95 (H=3) số q thỏa |coverage−q|≤0.05: CrostonSBA 7, LGBM-Q 7, TSB 5, WindowAverage 5, LGBM-PoisNB 4, MLP-NB 4, SeasonalNaive 2 (H=4: 7, 7, 6, 6, 5, 5, 2). Với tiêu chí rời rạc (q nằm trong [coverage_lt−0.05, coverage+0.05]): 7/7 cho mọi mô hình TRỪ SeasonalNaive (3/7). **Kết luận: E7 theo nghĩa đen chưa đạt cho WindowAverage/TSB/LGBM-PoisNB/MLP-NB ở q=0.5–0.6 (coverage cao hơn q do rời rạc) và SeasonalNaive kém hiệu chuẩn ở đuôi.**

## 4. Siêu tham số đã chốt trên val (`results/sieu_tham_so.json`, log đầy đủ `results/nhat_ky_tinh_chinh.csv`)
Tiêu chí: pinball trung bình trên mọi q và cả H=3,4 (E5). LGBM-Q không có tham số hiệu chuẩn (phân vị trực tiếp).
| index | tham so muc | kappa | w_series | pinball_avg (val) |
|---|---|---|---|---|
| WindowAverage | {} | 0.5000 | 0.5000 | 1.1837 |
| SeasonalNaive | {} | 0.5000 | 1.0000 | 1.5132 |
| CrostonSBA | {'alpha': 0.1} | 0.5000 | 0.5000 | 1.3868 |
| TSB | {'alpha_d': 0.05, 'alpha_p': 0.1} | 0.5000 | 0.5000 | 1.1387 |
| LGBM-PoisNB |  | 0.7500 | 1.0000 | 1.0247 |
| MLP-NB |  | 0.5000 |  | 0.9977 |

MLP-NB: số epoch chọn bằng tập kiểm tra nội bộ (56 gốc cuối của train, purge H ngày): H=3 → 17, H=4 → 30 (giới hạn cấu hình 30; H=4 chạm giới hạn).

## 5. Kết quả validation (`results/danh_gia_du_bao_val_all.csv`)
Tổng thể (823 chuỗi; pinball_avg = trung bình pinball trên 9 mức q; WAPE/MASE theo trung bình cửa sổ; LGBM-Q dùng mean của LightGBM-Poisson nên WAPE/MASE trùng LGBM-PoisNB):
| model | H3_pinball_avg | H3_wape | H3_mase | H4_pinball_avg | H4_wape | H4_mase |
|---|---|---|---|---|---|---|
| WindowAverage | 1.0757 | 0.4129 | 0.8125 | 1.2918 | 0.3682 | 0.8320 |
| SeasonalNaive | 1.4002 | 0.4741 | 0.9645 | 1.6261 | 0.4214 | 0.9792 |
| CrostonSBA | 1.2539 | 0.4948 | 0.9502 | 1.5197 | 0.4506 | 0.9886 |
| TSB | 1.0398 | 0.4058 | 0.8009 | 1.2377 | 0.3605 | 0.8173 |
| LGBM-Q | 0.9109 | 0.3697 | 0.7672 | 1.0922 | 0.3333 | 0.7884 |
| LGBM-PoisNB | 0.9302 | 0.3697 | 0.7672 | 1.1192 | 0.3333 | 0.7884 |
| MLP-NB | 0.9025 | 0.3701 | 0.7597 | 1.0929 | 0.3328 | 0.7746 |

Pinball từng q, tổng thể, H=3:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 1.6301 | 1.6487 | 1.5600 | 1.3458 | 1.1773 | 0.9519 | 0.6493 | 0.4425 | 0.2757 |
| SeasonalNaive | 1.8292 | 1.9121 | 1.8741 | 1.6864 | 1.5365 | 1.3309 | 1.0211 | 0.7944 | 0.6172 |
| CrostonSBA | 1.9449 | 1.9550 | 1.8452 | 1.5748 | 1.3725 | 1.1025 | 0.7318 | 0.4794 | 0.2794 |
| TSB | 1.6027 | 1.6160 | 1.5253 | 1.3062 | 1.1369 | 0.9124 | 0.6078 | 0.4053 | 0.2456 |
| LGBM-Q | 1.4404 | 1.4443 | 1.3503 | 1.1572 | 1.0097 | 0.8112 | 0.5108 | 0.3154 | 0.1581 |
| LGBM-PoisNB | 1.4530 | 1.4559 | 1.3679 | 1.1714 | 1.0213 | 0.8139 | 0.5368 | 0.3501 | 0.2013 |
| MLP-NB | 1.4361 | 1.4426 | 1.3530 | 1.1460 | 0.9935 | 0.7878 | 0.5004 | 0.3064 | 0.1569 |

Coverage P(y≤Q) tổng thể, H=3:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.5850 | 0.6572 | 0.7341 | 0.8115 | 0.8488 | 0.8897 | 0.9302 | 0.9531 | 0.9678 |
| SeasonalNaive | 0.5765 | 0.6137 | 0.6796 | 0.7436 | 0.7740 | 0.8043 | 0.8420 | 0.8677 | 0.8844 |
| CrostonSBA | 0.5445 | 0.6209 | 0.6996 | 0.7853 | 0.8273 | 0.8707 | 0.9201 | 0.9473 | 0.9671 |
| TSB | 0.5925 | 0.6632 | 0.7381 | 0.8120 | 0.8501 | 0.8905 | 0.9321 | 0.9548 | 0.9712 |
| LGBM-Q | 0.5412 | 0.6324 | 0.7254 | 0.8123 | 0.8629 | 0.9063 | 0.9495 | 0.9724 | 0.9881 |
| LGBM-PoisNB | 0.5989 | 0.6769 | 0.7566 | 0.8372 | 0.8741 | 0.9124 | 0.9504 | 0.9694 | 0.9820 |
| MLP-NB | 0.6198 | 0.6917 | 0.7647 | 0.8424 | 0.8771 | 0.9135 | 0.9524 | 0.9741 | 0.9874 |

Coverage_lt P(y<Q) tổng thể, H=3:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.3754 | 0.4601 | 0.5570 | 0.6541 | 0.7086 | 0.7645 | 0.8278 | 0.8623 | 0.8879 |
| SeasonalNaive | 0.3645 | 0.4086 | 0.4863 | 0.5642 | 0.6030 | 0.6408 | 0.6941 | 0.7288 | 0.7513 |
| CrostonSBA | 0.3920 | 0.4862 | 0.5862 | 0.6921 | 0.7525 | 0.8131 | 0.8795 | 0.9214 | 0.9516 |
| TSB | 0.3782 | 0.4671 | 0.5611 | 0.6586 | 0.7144 | 0.7727 | 0.8396 | 0.8812 | 0.9127 |
| LGBM-Q | 0.5375 | 0.6318 | 0.7254 | 0.8123 | 0.8629 | 0.9063 | 0.9495 | 0.9724 | 0.9881 |
| LGBM-PoisNB | 0.3937 | 0.4939 | 0.5970 | 0.7259 | 0.8085 | 0.8614 | 0.9211 | 0.9522 | 0.9725 |
| MLP-NB | 0.3942 | 0.4837 | 0.5775 | 0.6784 | 0.7444 | 0.8180 | 0.9126 | 0.9550 | 0.9791 |

Coverage tổng thể, H=4:
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.5633 | 0.6416 | 0.7211 | 0.8006 | 0.8436 | 0.8844 | 0.9264 | 0.9489 | 0.9649 |
| SeasonalNaive | 0.5617 | 0.6067 | 0.6803 | 0.7469 | 0.7800 | 0.8122 | 0.8509 | 0.8789 | 0.8972 |
| CrostonSBA | 0.5200 | 0.6012 | 0.6892 | 0.7757 | 0.8203 | 0.8663 | 0.9159 | 0.9459 | 0.9642 |
| TSB | 0.5713 | 0.6497 | 0.7272 | 0.8042 | 0.8436 | 0.8848 | 0.9262 | 0.9497 | 0.9680 |
| LGBM-Q | 0.5325 | 0.6200 | 0.7113 | 0.8111 | 0.8580 | 0.9048 | 0.9482 | 0.9726 | 0.9883 |
| LGBM-PoisNB | 0.5765 | 0.6590 | 0.7436 | 0.8268 | 0.8676 | 0.9055 | 0.9458 | 0.9668 | 0.9800 |
| MLP-NB | 0.5939 | 0.6699 | 0.7472 | 0.8239 | 0.8631 | 0.9032 | 0.9455 | 0.9676 | 0.9840 |

Pinball_avg theo speed (H=3) [đánh giá dự báo cửa sổ H=3; không phụ thuộc tỷ lệ chi phí, L, q; pinball_avg = trung bình 9 mức q]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 1.8014 | 0.7119 | 0.3950 |
| SeasonalNaive | 2.2183 | 0.9999 | 0.6150 |
| CrostonSBA | 2.1232 | 0.8135 | 0.4472 |
| TSB | 1.7472 | 0.6812 | 0.3835 |
| LGBM-Q | 1.4905 | 0.6237 | 0.3609 |
| LGBM-PoisNB | 1.5303 | 0.6310 | 0.3642 |
| MLP-NB | 1.4806 | 0.6143 | 0.3574 |

WAPE theo speed (H=3) [đánh giá dự báo cửa sổ H=3; không phụ thuộc tỷ lệ chi phí, L, q; pinball_avg = trung bình 9 mức q]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 0.3616 | 0.5519 | 0.7006 |
| SeasonalNaive | 0.4048 | 0.6673 | 0.8418 |
| CrostonSBA | 0.4455 | 0.6231 | 0.7930 |
| TSB | 0.3554 | 0.5404 | 0.6964 |
| LGBM-Q | 0.3123 | 0.5220 | 0.7039 |
| LGBM-PoisNB | 0.3123 | 0.5220 | 0.7039 |
| MLP-NB | 0.3150 | 0.5169 | 0.6888 |

MASE theo speed (H=3) [đánh giá dự báo cửa sổ H=3; không phụ thuộc tỷ lệ chi phí, L, q; pinball_avg = trung bình 9 mức q]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 0.7248 | 0.8570 | 0.8938 |
| SeasonalNaive | 0.8373 | 1.0318 | 1.0773 |
| CrostonSBA | 0.9081 | 0.9627 | 1.0055 |
| TSB | 0.7150 | 0.8387 | 0.8909 |
| LGBM-Q | 0.6479 | 0.8114 | 0.9074 |
| LGBM-PoisNB | 0.6479 | 0.8114 | 0.9074 |
| MLP-NB | 0.6495 | 0.8033 | 0.8844 |

Pinball_avg theo demand_type (H=3) [đánh giá dự báo cửa sổ H=3; không phụ thuộc tỷ lệ chi phí, L, q; pinball_avg = trung bình 9 mức q]:
| model | erratic | intermittent | lumpy | smooth |
|---|---|---|---|---|
| WindowAverage | 1.9778 | 0.7092 | 1.0450 | 2.0781 |
| SeasonalNaive | 2.1807 | 0.9879 | 1.3864 | 2.5746 |
| CrostonSBA | 2.1526 | 0.8786 | 1.2822 | 2.1712 |
| TSB | 1.9055 | 0.6865 | 1.0001 | 2.0279 |
| LGBM-Q | 1.5123 | 0.6057 | 0.8955 | 1.7811 |
| LGBM-PoisNB | 1.5735 | 0.6154 | 0.9183 | 1.8118 |
| MLP-NB | 1.4720 | 0.6023 | 0.8802 | 1.7818 |

WAPE theo demand_type (H=3) [đánh giá dự báo cửa sổ H=3; không phụ thuộc tỷ lệ chi phí, L, q; pinball_avg = trung bình 9 mức q]:
| model | erratic | intermittent | lumpy | smooth |
|---|---|---|---|---|
| WindowAverage | 0.4131 | 0.4491 | 0.5635 | 0.3083 |
| SeasonalNaive | 0.4240 | 0.5374 | 0.6461 | 0.3503 |
| CrostonSBA | 0.5001 | 0.5769 | 0.6972 | 0.3267 |
| TSB | 0.3945 | 0.4458 | 0.5469 | 0.3059 |
| LGBM-Q | 0.3341 | 0.4139 | 0.5191 | 0.2684 |
| LGBM-PoisNB | 0.3341 | 0.4139 | 0.5191 | 0.2684 |
| MLP-NB | 0.3321 | 0.4113 | 0.5176 | 0.2727 |

Pinball_avg theo demand_type×speed (H=3) [đánh giá dự báo cửa sổ H=3; không phụ thuộc tỷ lệ chi phí, L, q; pinball_avg = trung bình 9 mức q]:
| model | erratic_fast | intermittent_fast | intermittent_medium | intermittent_slow | lumpy_fast | lumpy_medium | lumpy_slow | smooth_fast | smooth_medium |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 1.9778 | 1.4916 | 0.6636 | 0.3668 | 1.5703 | 0.7981 | 0.5125 | 2.1028 | 0.6180 |
| SeasonalNaive | 2.1807 | 1.9029 | 0.9289 | 0.5955 | 1.9891 | 1.1262 | 0.6966 | 2.6035 | 0.8723 |
| CrostonSBA | 2.1526 | 2.2526 | 0.7132 | 0.3992 | 1.9050 | 0.9903 | 0.6476 | 2.1953 | 0.7436 |
| TSB | 1.9055 | 1.4338 | 0.6397 | 0.3638 | 1.5254 | 0.7537 | 0.4658 | 2.0505 | 0.6952 |
| LGBM-Q | 1.5123 | 1.1892 | 0.5772 | 0.3425 | 1.3183 | 0.7054 | 0.4377 | 1.8011 | 0.6058 |
| LGBM-PoisNB | 1.5735 | 1.2291 | 0.5825 | 0.3425 | 1.3603 | 0.7151 | 0.4550 | 1.8311 | 0.6741 |
| MLP-NB | 1.4720 | 1.2077 | 0.5669 | 0.3375 | 1.2860 | 0.6976 | 0.4408 | 1.8019 | 0.5957 |

(Mọi chỉ số theo mọi nhóm, H=3 và H=4: xem CSV.)

## 6. File sinh ra
`results/danh_gia_du_bao_val_all.csv` (schema model,H,group_type,group,metric,q,value; metric: pinball, coverage, coverage_lt, pinball_avg, wape, mase, n_series), `results/forecasts/val.npz` (Q[823×n×9], mean, actual, origins), `results/sieu_tham_so.json`, `results/nhat_ky_tinh_chinh.csv`, `results/thoi_gian_val.json`, `results/chay_val.log`.

## 7. Cấu hình
Seed 42 (LightGBM `random_state`, `torch.manual_seed`, `np.random.default_rng`). alpha: n_hist=180, alpha∈[0.001, 20]. LightGBM: lr 0.05, 31 lá, 200 cây, min_child_samples 100, subsample 0.8, colsample 0.9; huấn luyện trên 730 ngày gốc gần nhất (585501 dòng với H=3, 585426 với H=4, 30 đặc trưng). CPU 12 luồng. Thời gian (giây): WindowAverage 34.6, SeasonalNaive 30.7, CrostonSBA 40.4, TSB 51.9 (đã gồm tuning), LightGBM fit H=3 43.7 / H=4 33.6 (10 mô hình mỗi H), MLP fit H=3 58.4 / H=4 53.8, tổng 424.6. Thư viện: lightgbm 4.7.0, torch 2.15.0+cpu, scipy 1.18.1, numpy 2.5.4, pandas 2.3.3. Đã cài thêm torch (CPU) — việc cài làm `setuptools` đổi sang 78.1.0.

## 8. Vấn đề / giả định chưa chắc chắn
- **Siêu tham số chọn trên val rồi đánh giá trên chính val** (theo E5) ⇒ số val hơi lạc quan cho các mô hình có tuning (kappa, w, alpha Croston/TSB). Chỉ số test (GĐ5) mới là ước lượng không chệch.
- kappa chọn = 0.5 cho cả 4 mô hình thống kê: alpha moment từ phần dư (gồm cả sai số dự báo điểm) quá lớn so với tối ưu pinball; kappa chỉ là hiệu chỉnh thực nghiệm, không có cơ sở lý thuyết.
- LightGBM phân vị KHÔNG tái huấn luyện trong 28 ngày val (huấn luyện một lần trên train; đặc trưng cập nhật theo lịch sử thật). Không refit theo `refit_every` — để giữ chi phí tính toán thấp.
- alpha của LGBM-PoisNB tính từ phần dư các gốc lịch sử gồm cả trong mẫu (train) lẫn ngoài mẫu (val đã quan sát): có thể thiếu phân tán nhẹ.
- Mean = 0 (SeasonalNaive thường xuyên, hoặc chuỗi chưa có lần bán nào) ⇒ phân vị 0 ⇒ đuôi SeasonalNaive hiệu chuẩn kém.
- Chỉ 26/25 gốc, các cửa sổ chồng lấn ⇒ sai số chuẩn lớn; chênh nhỏ giữa mô hình (ví dụ MLP-NB vs LGBM-Q) chưa có kiểm định thống kê.
- MLP-NB: epoch chạm giới hạn 30 ở H=4. Lần chạy thử trước đó với tối đa 12 epoch cho pinball_avg H=3 0.9016 và H=4 1.0848, tốt hơn lần 30 epoch (0.9025; 1.0929) — chênh nhỏ; epoch chọn bằng holdout nội bộ (NLL), không phải theo val.
- Giá khuyết (NaN) giữ nguyên ở LightGBM; ở MLP thay bằng 0 + cờ thiếu.

## 9. Chưa làm
WRMSSE (lý do ở mục 1); DeepAR/N-HiTS/TFT và Chronos-2 (không cài neuralforecast/gluonts; MLP-NB thay thế); refit LightGBM theo `refit_every` trong val; kiểm định thống kê giữa các mô hình.
