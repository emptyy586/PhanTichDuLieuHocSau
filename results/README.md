# Chỉ mục thư mục `results/`

> **File đóng băng – không sửa tay**: `hparams.json` (sha256 `8bc7d7507bf0167e83b1d31f831fb8edb5edf60c23a742c67b32b0f86366765b`), `hparams_deepar.json` và mọi file có hash ghi trong `gd8_baseline_hashes.txt` (CSV/npz/json của 7 mô hình gốc ở mốc tag git `pre-gd8`). Các test (`tests/test_gd8.py`) kiểm hash này; sửa tay sẽ làm test fail.
>
> Cột "tạo bởi" lấy từ mã trong `src/` (tìm tên file đầu ra). Chỗ nào không có dòng mã ghi file đó thì ghi rõ "chưa xác định" hoặc nêu nguồn khác. Lệnh chạy cụ thể xem `README.md` ở gốc dự án. "Báo cáo" = các file trong `reports/`.

## 1. Dự báo MLP / LightGBM / baseline thống kê
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `forecast_eval_val_all.csv`, `forecast_eval_test_all.csv` | `src/run_forecasts.py` (`--split val/test`) | Chỉ số dự báo (pinball, coverage, WAPE, MASE…) theo mô hình × H × nhóm, 7 mô hình | báo cáo 02, 05; bảng kết quả dự báo |
| `forecasts/val.npz`, `forecasts/test.npz` | `src/run_forecasts.py` | Phân vị (lưới chính + phụ), trung bình, thực tế của 7 mô hình × H ∈ {3,4}; đầu vào của mô phỏng tồn kho | mô phỏng (mục 4) |
| `hparams.json` | `src/run_forecasts.py` (`--split val`) | Siêu tham số chốt trên val (kappa, w_series, alpha Croston/TSB, epoch MLP) – **đóng băng** | báo cáo 02, 07 |
| `tuning_log.csv` | `src/run_forecasts.py` | Nhật ký tinh chỉnh siêu tham số trên val | báo cáo 02 |
| `timing_val.json`, `timing_test.json` | `src/run_forecasts.py` | Thời gian chạy từng bước | báo cáo 02, 05 |
| `mlp_seeds.csv`, `mlp_nll_curves.csv` | `src/mlp_seeds.py` | MLP-NB 3 seed (pinball/WAPE/MASE val/test, kappa, epoch) và đường NLL holdout | báo cáo 06, 07; hình `mlp_nll_curves.png` |
| `baseline_stats_val.csv` | `src/models_stats.py` | Baseline thống kê điểm (StatsForecast) trên val, giai đoạn đầu | báo cáo 00; notebook 03 |
| `forecast_eval_val.csv`, `window_quantiles_val.parquet` | `src/forecast_prob.py` (`export_validation_outputs`, lệnh `python -m src.forecast_prob`) | Kết quả của pipeline CŨ (legacy, không phải rolling-origin thật); chỉ giữ để tham chiếu "trước/sau" | báo cáo 02 (bảng trước/sau sửa lỗi) |

## 2. DeepAR-style (GĐ8)
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `hparams_deepar.json` | `src/deepar.py` (`--stage select`, bổ sung epoch ở `--stage freeze`) | Cấu hình, kappa, epoch từng seed – **đóng băng trước khi chạy test** | báo cáo 08 |
| `tuning_log_deepar.csv`, `tuning_kappa_deepar.csv` | `src/deepar.py` (`--stage select`) | Tuning 8 cấu hình và kappa trên val | báo cáo 08 |
| `deepar_seeds.csv` | `src/deepar.py` (`--stage freeze`, `--stage collect_test`) | pinball/WAPE/MASE, thời gian, epoch của 3 seed (val, test) | báo cáo 08 |
| `deepar_nll_curves.csv` | `src/deepar.py` (`--stage freeze`) | NLL huấn luyện/holdout theo epoch | hình `deepar_nll_curves.png` |
| `forecast_eval_deepar_val_all.csv`, `forecast_eval_deepar_test_all.csv` | `src/deepar.py` (`seed_val`, `seed_test`, seed 42) | Chỉ số dự báo của DeepAR (cùng schema 7 mô hình) | báo cáo 08 |
| `forecasts/deepar_val.npz`, `forecasts/deepar_test.npz` | `src/deepar.py` | Phân vị/trung bình của DeepAR (seed 42) | mô phỏng DeepAR |
| `inventory_sim_deepar_val.csv`, `…_test.csv`, `…_val_ext.csv`, `…_test_ext.csv` | `src/deepar_analysis.py` (`--what sim`, gọi `run_inventory_sim`) | Mô phỏng tồn kho riêng cho DeepAR (lưới chính và lưới phụ q<0.5) | báo cáo 08 |
| `deepar_forecast_main_table.csv`, `deepar_forecast_by_speed.csv` | `src/deepar_analysis.py` (`--what tables`) | Bảng pinball/WAPE/MASE gộp 8 mô hình; theo nhóm tốc độ | báo cáo 08 |
| `deepar_calibration_table.csv`, `deepar_calibration_summary.csv` | `src/deepar_analysis.py` (`--what tables`) | Hiệu chuẩn P(y<q̂), P(y≤q̂) của DeepAR | báo cáo 08 |
| `deepar_bootstrap_cost.csv`, `deepar_bootstrap_pinball.csv` | `src/deepar_analysis.py` (`--what tables`) | Bootstrap theo chuỗi: DeepAR − MLP-NB/LGBM-Q/TSB (chi phí, pinball) | báo cáo 08 |
| `q_star_with_deepar.csv` | `src/deepar_analysis.py` (`--what tables`) | q* (val) và chi phí/fill val-test của 8 mô hình | báo cáo 08; hình `final_cost_val_vs_test.png` |
| `deepar_cache/` (50 file, nằm trong `.gitignore`) | `src/deepar.py` (`CACHE_DIR`) | Trọng số mô hình, log tiến trình, cờ hoàn tất của các stage; **chạy lại được, không cần commit** | – |

## 3. Hiệu chuẩn
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `calibration_table.csv`, `calibration_summary.csv` | `src/calibration.py` (`python -m src.calibration`) | P(y<q̂), P(y≤q̂) từng q ≤ 0.95 và số q thỏa khoảng/dung sai, 7 mô hình | báo cáo 06, 07 |

## 4. Mô phỏng tồn kho và chính sách
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `inventory_sim_val.csv`, `inventory_sim_test.csv` | `src/inventory_sim.py` (`--split val/test`) | Mô phỏng order-up-to: mô hình × q × L × tỷ lệ chi phí × nhóm; 7 mô hình, lưới chính | báo cáo 03, 05 |
| `inventory_sim_val_ext.csv`, `inventory_sim_test_ext.csv` | `src/inventory_sim.py` (`--ext`) | Như trên cho lưới phụ q ∈ {0.3, 0.4} | báo cáo 06, 07 |
| `q_star_val.csv`, `policy_table_val.csv`, `research_questions_val.csv` | `src/policy_analysis.py` (`python -m src.policy_analysis`) | q* theo (mô hình, L, tỷ lệ, nhóm); bảng chính sách theo nhóm; bootstrap câu hỏi nghiên cứu (val) | báo cáo 04, 05; hình heatmap q* |
| `tables/equal_fill_inventory_val.csv` | `src/policy_analysis.py` | Tồn kho cần để đạt cùng fill rate với chính sách điểm | báo cáo 04 |
| `policy_eval_val_test.csv`, `policy_eval_bootstrap.csv`, `test_qstar_transfer.csv`, `research_questions_test.csv` | `src/final_eval.py` (`python -m src.final_eval`) | Chính sách chốt từ val áp lên test; bootstrap; chuyển q* val → test; câu hỏi nghiên cứu (test) | báo cáo 05, 06, 07, final_report |
| `q_star_ext_val.csv`, `safety_point_test.csv`, `safety_point_by_speed_test.csv` | `src/safety_point.py` (`python -m src.safety_point`) | q* lưới mở rộng; điểm có hệ số an toàn k vs phân vị; so sánh theo nhóm tốc độ | báo cáo 06, 07, final_report (câu 1, 2) |

## 5. Ablation và lớp chuỗi
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `ablation_exog.csv` | `src/ablation.py` (mặc định) | LightGBM-Q đầy đủ vs bỏ giá/SNAP/sự kiện | báo cáo 06 |
| `ablation_effect_cols.csv` | `src/ablation.py --effect-cols` | LightGBM-Q bỏ riêng `fut_wd_index`, `fut_event_index` | báo cáo 07 |
| `series_classes.csv` | `src/features.py` (`python -m src.features`) | ADI, CV², demand_type, speed của 823 chuỗi (tính trên train) | báo cáo 01; mọi bước theo nhóm; hình `adi_cv2_scatter.png` |

## 6. Kiểm tra, log, hash
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `check_report.txt` | `src/models_stats.py` | Ghi chú nguyên nhân thiếu TSB và bảng baseline thống kê giai đoạn đầu | báo cáo 00 |
| `gd8_baseline_hashes.txt` | chưa xác định trong mã (không có dòng mã ghi file này); được tạo bằng lệnh `sha256sum` ở GĐ8 | sha256 của 37 file CSV/npz/json ở mốc `pre-gd8` | test `tests/test_gd8.py` |
| `run_val.log`, `run_test.log`, `run_mlp_seeds.log`, `run_ablation.log`, `run_ablation_effect.log` | chưa xác định trong mã (không có dòng mã ghi các file này); là stdout của lệnh `run_forecasts` / `mlp_seeds` / `ablation` được chuyển hướng ra file khi chạy | Log chạy lần gần nhất của từng lệnh | tham chiếu thời gian chạy |

## 7. Thư mục con
| Thư mục | Số file | Nội dung |
|---|---|---|
| `figures/` | 20 PNG | Hình cho báo cáo Word (6.3 inch, 300 dpi). Do `src/figures_word.py` vẽ (`python -m src.figures_word`, chạy sau cùng); notebook 01/02 và các bước phân tích cũ ghi cùng tên file theo kiểu cũ |
| `tables/` | 10 CSV | `eda_*.csv` và `demand_speed_counts.csv` (do notebook 01/02 ghi, không phải module `src/`), `equal_fill_inventory_val.csv` (`policy_analysis.py`), `deepar_hparams.csv`, `deepar_tuning_log.csv` (`deepar_analysis.py`) |
| `reports/` | 10 MD | Báo cáo từng giai đoạn `00`–`08` và `final_report.md`. Dựng bằng script tạm ngoài repo (không có trong `src/`): **chưa xác định trong mã** |
| `forecasts/` | 4 NPZ | `val.npz`, `test.npz` (`run_forecasts.py`), `deepar_val.npz`, `deepar_test.npz` (`deepar.py`) |
| `deepar_cache/` | 50 file (bị ignore) | Xem mục 2 |
