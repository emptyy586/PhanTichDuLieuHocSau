# Chỉ mục thư mục `results/`

> **File đóng băng – không sửa tay**: `sieu_tham_so.json` (sha256 `8bc7d7507bf0167e83b1d31f831fb8edb5edf60c23a742c67b32b0f86366765b`), `sieu_tham_so_deepar.json` và mọi file có hash ghi trong `ma_bam_co_so_gd8.txt` (CSV/npz/json của 7 mô hình gốc ở mốc tag git `pre-gd8`). Các test (`tests/kiem_thu_gd8.py`) kiểm hash này; sửa tay sẽ làm test fail.
>
> Cột "tạo bởi" lấy từ mã trong `src/` (tìm tên file đầu ra). Chỗ nào không có dòng mã ghi file đó thì ghi rõ "chưa xác định" hoặc nêu nguồn khác. Lệnh chạy cụ thể xem `README.md` ở gốc dự án. "Báo cáo" = các file trong `reports/`.

## 1. Dự báo MLP / LightGBM / baseline thống kê
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `danh_gia_du_bao_val_all.csv`, `danh_gia_du_bao_test_all.csv` | `src/chay_du_bao.py` (`--split val/test`) | Chỉ số dự báo (pinball, coverage, WAPE, MASE…) theo mô hình × H × nhóm, 7 mô hình | báo cáo 02, 05; bảng kết quả dự báo |
| `forecasts/val.npz`, `forecasts/test.npz` | `src/chay_du_bao.py` | Phân vị (lưới chính + phụ), trung bình, thực tế của 7 mô hình × H ∈ {3,4}; đầu vào của mô phỏng tồn kho | mô phỏng (mục 4) |
| `sieu_tham_so.json` | `src/chay_du_bao.py` (`--split val`) | Siêu tham số chốt trên val (kappa, w_series, alpha Croston/TSB, epoch MLP) – **đóng băng** | báo cáo 02, 07 |
| `nhat_ky_tinh_chinh.csv` | `src/chay_du_bao.py` | Nhật ký tinh chỉnh siêu tham số trên val | báo cáo 02 |
| `thoi_gian_val.json`, `thoi_gian_test.json` | `src/chay_du_bao.py` | Thời gian chạy từng bước | báo cáo 02, 05 |
| `mlp_cac_seed.csv`, `mlp_duong_nll.csv` | `src/mlp_cac_seed.py` | MLP-NB 3 seed (pinball/WAPE/MASE val/test, kappa, epoch) và đường NLL holdout | báo cáo 06, 07; hình `mlp_duong_nll.png` |
| `thong_ke_co_so_val.csv` | `src/mo_hinh_thong_ke.py` | Baseline thống kê điểm (StatsForecast) trên val, giai đoạn đầu | báo cáo 00; notebook 03 |
| `danh_gia_du_bao_val.csv`, `phan_vi_cua_so_val.parquet` | `src/du_bao_xac_suat.py` (`export_validation_outputs`, lệnh `python -m src.du_bao_xac_suat`) | Kết quả của pipeline CŨ (legacy, không phải rolling-origin thật); chỉ giữ để tham chiếu "trước/sau" | báo cáo 02 (bảng trước/sau sửa lỗi) |

## 2. DeepAR-style (GĐ8)
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `sieu_tham_so_deepar.json` | `src/deepar.py` (`--stage select`, bổ sung epoch ở `--stage freeze`) | Cấu hình, kappa, epoch từng seed – **đóng băng trước khi chạy test** | báo cáo 08 |
| `nhat_ky_tinh_chinh_deepar.csv`, `tinh_chinh_kappa_deepar.csv` | `src/deepar.py` (`--stage select`) | Tuning 8 cấu hình và kappa trên val | báo cáo 08 |
| `deepar_cac_seed.csv` | `src/deepar.py` (`--stage freeze`, `--stage collect_test`) | pinball/WAPE/MASE, thời gian, epoch của 3 seed (val, test) | báo cáo 08 |
| `deepar_duong_nll.csv` | `src/deepar.py` (`--stage freeze`) | NLL huấn luyện/holdout theo epoch | hình `deepar_duong_nll.png` |
| `danh_gia_du_bao_deepar_val_all.csv`, `danh_gia_du_bao_deepar_test_all.csv` | `src/deepar.py` (`seed_val`, `seed_test`, seed 42) | Chỉ số dự báo của DeepAR (cùng schema 7 mô hình) | báo cáo 08 |
| `forecasts/deepar_val.npz`, `forecasts/deepar_test.npz` | `src/deepar.py` | Phân vị/trung bình của DeepAR (seed 42) | mô phỏng DeepAR |
| `mo_phong_ton_kho_deepar_val.csv`, `…_test.csv`, `…_val_ext.csv`, `…_test_ext.csv` | `src/deepar_phan_tich.py` (`--what sim`, gọi `run_inventory_sim`) | Mô phỏng tồn kho riêng cho DeepAR (lưới chính và lưới phụ q<0.5) | báo cáo 08 |
| `deepar_du_bao_bang_chinh.csv`, `deepar_du_bao_theo_toc_do.csv` | `src/deepar_phan_tich.py` (`--what tables`) | Bảng pinball/WAPE/MASE gộp 8 mô hình; theo nhóm tốc độ | báo cáo 08 |
| `deepar_hieu_chinh_bang.csv`, `deepar_hieu_chinh_tom_tat.csv` | `src/deepar_phan_tich.py` (`--what tables`) | Hiệu chuẩn P(y<q̂), P(y≤q̂) của DeepAR | báo cáo 08 |
| `deepar_bootstrap_chi_phi.csv`, `deepar_bootstrap_pinball.csv` | `src/deepar_phan_tich.py` (`--what tables`) | Bootstrap theo chuỗi: DeepAR − MLP-NB/LGBM-Q/TSB (chi phí, pinball) | báo cáo 08 |
| `q_sao_co_deepar.csv` | `src/deepar_phan_tich.py` (`--what tables`) | q* (val) và chi phí/fill val-test của 8 mô hình | báo cáo 08; hình `cuoi_chi_phi_val_va_test.png` |
| `deepar_cache/` (50 file, nằm trong `.gitignore`) | `src/deepar.py` (`CACHE_DIR`) | Trọng số mô hình, log tiến trình, cờ hoàn tất của các stage; **chạy lại được, không cần commit** | – |

## 3. Hiệu chuẩn
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `hieu_chinh_bang.csv`, `hieu_chinh_tom_tat.csv` | `src/hieu_chinh.py` (`python -m src.hieu_chinh`) | P(y<q̂), P(y≤q̂) từng q ≤ 0.95 và số q thỏa khoảng/dung sai, 7 mô hình | báo cáo 06, 07 |

## 4. Mô phỏng tồn kho và chính sách
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `mo_phong_ton_kho_val.csv`, `mo_phong_ton_kho_test.csv` | `src/mo_phong_ton_kho.py` (`--split val/test`) | Mô phỏng order-up-to: mô hình × q × L × tỷ lệ chi phí × nhóm; 7 mô hình, lưới chính | báo cáo 03, 05 |
| `mo_phong_ton_kho_val_ext.csv`, `mo_phong_ton_kho_test_ext.csv` | `src/mo_phong_ton_kho.py` (`--ext`) | Như trên cho lưới phụ q ∈ {0.3, 0.4} | báo cáo 06, 07 |
| `q_sao_val.csv`, `bang_chinh_sach_val.csv`, `cau_hoi_nghien_cuu_val.csv` | `src/phan_tich_chinh_sach.py` (`python -m src.phan_tich_chinh_sach`) | q* theo (mô hình, L, tỷ lệ, nhóm); bảng chính sách theo nhóm; bootstrap câu hỏi nghiên cứu (val) | báo cáo 04, 05; hình heatmap q* |
| `tables/ton_kho_cung_fill_val.csv` | `src/phan_tich_chinh_sach.py` | Tồn kho cần để đạt cùng fill rate với chính sách điểm | báo cáo 04 |
| `danh_gia_chinh_sach_val_test.csv`, `danh_gia_chinh_sach_bootstrap.csv`, `chuyen_q_sao_test.csv`, `cau_hoi_nghien_cuu_test.csv` | `src/danh_gia_cuoi.py` (`python -m src.danh_gia_cuoi`) | Chính sách chốt từ val áp lên test; bootstrap; chuyển q* val → test; câu hỏi nghiên cứu (test) | báo cáo 05, 06, 07, bao_cao_cuoi_cung |
| `q_sao_ext_val.csv`, `diem_an_toan_test.csv`, `diem_an_toan_theo_toc_do_test.csv` | `src/diem_an_toan.py` (`python -m src.diem_an_toan`) | q* lưới mở rộng; điểm có hệ số an toàn k vs phân vị; so sánh theo nhóm tốc độ | báo cáo 06, 07, bao_cao_cuoi_cung (câu 1, 2) |

## 5. Ablation và lớp chuỗi
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `loai_bo_bien_ngoai_sinh.csv` | `src/loai_bo_dac_trung.py` (mặc định) | LightGBM-Q đầy đủ vs bỏ giá/SNAP/sự kiện | báo cáo 06 |
| `loai_bo_hieu_ung_cot.csv` | `src/loai_bo_dac_trung.py --effect-cols` | LightGBM-Q bỏ riêng `fut_wd_index`, `fut_event_index` | báo cáo 07 |
| `phan_loai_chuoi.csv` | `src/dac_trung.py` (`python -m src.dac_trung`) | ADI, CV², demand_type, speed của 823 chuỗi (tính trên train) | báo cáo 01; mọi bước theo nhóm; hình `adi_cv2_phan_tan.png` |

## 6. Kiểm tra, log, hash
| File | Tạo bởi | Nội dung | Dùng cho |
|---|---|---|---|
| `bao_cao_kiem_tra.txt` | `src/mo_hinh_thong_ke.py` | Ghi chú nguyên nhân thiếu TSB và bảng baseline thống kê giai đoạn đầu | báo cáo 00 |
| `ma_bam_co_so_gd8.txt` | chưa xác định trong mã (không có dòng mã ghi file này); được tạo bằng lệnh `sha256sum` ở GĐ8 | sha256 của 37 file CSV/npz/json ở mốc `pre-gd8` | test `tests/kiem_thu_gd8.py` |
| `chay_val.log`, `chay_test.log`, `chay_mlp_cac_seed.log`, `chay_loai_bo_dac_trung.log`, `chay_loai_bo_hieu_ung.log` | chưa xác định trong mã (không có dòng mã ghi các file này); là stdout của lệnh `chay_du_bao` / `mlp_cac_seed` / `loai_bo_dac_trung` được chuyển hướng ra file khi chạy | Log chạy lần gần nhất của từng lệnh | tham chiếu thời gian chạy |

## 7. Thư mục con
| Thư mục | Số file | Nội dung |
|---|---|---|
| `figures/` | 20 PNG | Hình cho báo cáo Word (6.3 inch, 300 dpi). Do `src/ve_hinh_bao_cao.py` vẽ (`python -m src.ve_hinh_bao_cao`, chạy sau cùng); notebook 01/02 và các bước phân tích cũ ghi cùng tên file theo kiểu cũ |
| `tables/` | 10 CSV | `kham_pha_*.csv` và `so_luong_nhom_toc_do_nhu_cau.csv` (do notebook 01/02 ghi, không phải module `src/`), `ton_kho_cung_fill_val.csv` (`phan_tich_chinh_sach.py`), `deepar_sieu_tham_so.csv`, `deepar_nhat_ky_tinh_chinh.csv` (`deepar_phan_tich.py`) |
| `reports/` | 10 MD | Báo cáo từng giai đoạn `00`–`08` và `bao_cao_cuoi_cung.md`. Dựng bằng script tạm ngoài repo (không có trong `src/`): **chưa xác định trong mã** |
| `forecasts/` | 4 NPZ | `val.npz`, `test.npz` (`chay_du_bao.py`), `deepar_val.npz`, `deepar_test.npz` (`deepar.py`) |
| `deepar_cache/` | 50 file (bị ignore) | Xem mục 2 |
