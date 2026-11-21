# Dự báo nhu cầu xác suất và mô phỏng chính sách bổ sung hàng tồn kho (Đề tài 17) – Bản nháp báo cáo

## 1. Giới thiệu
Mục tiêu: dự báo phân phối xác suất của nhu cầu từng mặt hàng và dùng phân vị dự báo để chọn mức đặt hàng (order-up-to) cân bằng giữa thiếu hàng và chi phí tồn kho. Ba câu hỏi nghiên cứu: (1) dự báo phân vị có chọn mức tồn kho tốt hơn dự báo điểm không; (2) kết quả khác nhau thế nào giữa hàng bán nhanh và hàng bán chậm/gián đoạn; (3) học sâu xác suất có hơn Croston/TSB không. Toàn bộ mã, kết quả và các báo cáo trung gian (`results/reports/00…05`) nằm trong repo; mọi con số dưới đây lấy từ file kết quả đã sinh.

## 2. Dữ liệu và chia dữ liệu
- M5 (Walmart), cửa hàng CA_1 × phòng ban FOODS_3: 823 chuỗi, 1941 ngày (2011-01-29 → 2016-05-22). Train 2011-01-29 → 2016-03-27 (1885 ngày), validation 2016-03-28 → 2016-04-24 (28 ngày), test 2016-04-25 → 2016-05-22 (28 ngày). Giá bán, sự kiện, SNAP của các ngày tương lai coi là đã biết (19.8606% giá khuyết – trước khi mặt hàng được niêm yết).
- Phân loại chuỗi (chỉ tính trên train): demand_type theo ADI/CV² (ngưỡng 1.32/0.49) và speed theo tam phân vị doanh số trung bình. Số chuỗi: smooth 120, erratic 45, intermittent 419, lumpy 239; fast 326, medium 321, slow 176 (bảng đầy đủ: `results/reports/01_eda.md`).
- EDA (train, sau ngày ra mắt): mùa vụ tuần rõ (Chủ nhật 4.1099 vs Thứ 4 2.5597 đơn vị/chuỗi/ngày), ngày SNAP cao hơn 11.9%, sự kiện phân hóa mạnh (SuperBowl chỉ số 1.3460, Thanksgiving 0.6986, Christmas 0 vì đóng cửa).
- Chống rò rỉ: đặc trưng tại gốc t chỉ dùng doanh số đến hết ngày t (có test đơn vị); các chỉ số hiệu ứng thứ/sự kiện chỉ tính đến hết giai đoạn đã biết; phân loại và ngưỡng speed chỉ từ train; test chỉ dùng để đánh giá sau khi mọi siêu tham số, mô hình, q* và chính sách đã chốt bằng validation.

## 3. Phương pháp
**Mục tiêu dự báo**: tổng nhu cầu cửa sổ D_H(t) = y[t+1] + … + y[t+H], H = lead time + chu kỳ đặt (1) ∈ {3, 4}, đánh giá bằng rolling-origin thật (26 gốc với H=3, 25 gốc với H=4 trong mỗi giai đoạn 28 ngày).
**Mô hình** (7): WindowAverage(28), SeasonalNaive(lag 7), CrostonSBA, TSB – mỗi mô hình cho trung bình cửa sổ và được bọc bằng phân phối Negative Binomial (độ phân tán alpha ước lượng cuốn chiếu từ phần dư quá khứ, trộn với nhóm demand_type × speed, nhân hệ số hiệu chuẩn kappa chọn trên val; phân vị chính xác bằng `scipy.stats.nbinom.ppf`); LightGBM phân vị (một mô hình/phân vị, huấn luyện trực tiếp trên D_H, 30 đặc trưng gồm lag/rolling, giá, thay đổi giá, SNAP, sự kiện, thứ, speed/demand_type); LightGBM-Poisson + NB; MLP-NB (mạng MLP hai đầu ra NB(mu, alpha), NLL trực tiếp trên D_H). Lưu ý: MLP-NB không phải DeepAR/N-HiTS/TFT (không cài neuralforecast/gluonts; các thư viện đó chỉ cho phân vị biên theo ngày, mà tổng phân vị ngày khác phân vị của tổng). Chronos-2 không làm. Lưới phân vị: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99.
**Chỉ số dự báo**: pinball (từng q và trung bình), coverage P(y≤Q) và P(y<Q), WAPE (trên trung bình cửa sổ), MASE (chia cho MAE trong mẫu train của dự báo naive theo cửa sổ). WRMSSE không làm vì định nghĩa M5 cần dự báo ngày 28 bước và 12 mức gộp, không tái hiện được ở một cửa hàng × một phòng ban với dự báo cửa sổ.

## 4. Mô phỏng tồn kho và giả định chi phí
Chính sách order-up-to hằng ngày: cuối ngày t đặt hàng với S = ceil(phân vị q của D_H(t)), đặt max(0, S − tồn kho vị trí); hàng về cuối ngày t+L và dùng được từ ngày t+L+1; L ∈ {2, 3}; thiếu hàng bị mất (lost sales); tồn đầu = S ở gốc đầu tiên, không có hàng đang về. Chi phí **tương đối, không phải chi phí thực**: giữ 1 đơn vị-ngày = 1.0; thiếu 1 đơn vị = cost_ratio (cơ sở 10; quét 1, 2, 5, 10, 20, 50). Đối chứng dự báo điểm: S = ceil(trung bình dự báo của D_H). Áp dụng giống hệt cho mọi mô hình. Chỉ số: fill rate (tổng bán/tổng cầu), số ngày hết hàng, tồn kho cuối ngày trung bình, chi phí giữ, thiếu, tổng; tách theo speed, demand_type. Chi tiết quy ước: `results/reports/03_inventory_sim.md`.

## 5. Kết quả (test; val ở các báo cáo 02–04)
Dự báo, H=3, tổng thể (validation vs test):
| model | pinball_avg_val | pinball_avg_test | wape_val | wape_test | mase_val | mase_test |
|---|---|---|---|---|---|---|
| WindowAverage | 1.0757 | 1.1316 | 0.4129 | 0.3987 | 0.8125 | 0.8474 |
| SeasonalNaive | 1.4002 | 1.4112 | 0.4741 | 0.4399 | 0.9645 | 0.9882 |
| CrostonSBA | 1.2539 | 1.2659 | 0.4948 | 0.4484 | 0.9502 | 0.9174 |
| TSB | 1.0398 | 1.0799 | 0.4058 | 0.3898 | 0.8009 | 0.8287 |
| LGBM-Q | 0.9109 | 0.9354 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| LGBM-PoisNB | 0.9302 | 0.9487 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| MLP-NB | 0.9025 | 0.9477 | 0.3701 | 0.3512 | 0.7597 | 0.7839 |

WAPE theo speed (test, H=3; đánh giá dự báo cửa sổ, không phụ thuộc tỷ lệ chi phí/L/q):
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | 0.3460 | 0.5409 | 0.7675 |
| SeasonalNaive | 0.3627 | 0.6613 | 0.9250 |
| CrostonSBA | 0.3990 | 0.5800 | 0.8008 |
| TSB | 0.3372 | 0.5336 | 0.7494 |
| LGBM-Q | 0.2916 | 0.5148 | 0.7447 |
| LGBM-PoisNB | 0.2916 | 0.5148 | 0.7447 |
| MLP-NB | 0.2931 | 0.5141 | 0.7320 |

Coverage test (H=3):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.5694 | 0.6488 | 0.7303 | 0.8092 | 0.8499 | 0.8906 | 0.9318 | 0.9539 | 0.9688 |
| SeasonalNaive | 0.5518 | 0.5944 | 0.6655 | 0.7356 | 0.7677 | 0.7974 | 0.8354 | 0.8634 | 0.8805 |
| CrostonSBA | 0.5236 | 0.6080 | 0.6943 | 0.7833 | 0.8278 | 0.8738 | 0.9215 | 0.9475 | 0.9665 |
| TSB | 0.5801 | 0.6629 | 0.7398 | 0.8152 | 0.8547 | 0.8963 | 0.9369 | 0.9583 | 0.9741 |
| LGBM-Q | 0.5127 | 0.6069 | 0.7062 | 0.8064 | 0.8553 | 0.9027 | 0.9476 | 0.9720 | 0.9874 |
| LGBM-PoisNB | 0.5893 | 0.6711 | 0.7546 | 0.8386 | 0.8774 | 0.9156 | 0.9517 | 0.9711 | 0.9842 |
| MLP-NB | 0.6004 | 0.6787 | 0.7571 | 0.8359 | 0.8739 | 0.9105 | 0.9493 | 0.9702 | 0.9834 |

(Coverage P(y≤Q) cao hơn q ở các q thấp do phân phối nhu cầu rời rạc với nhiều giá trị 0; mục tiêu |coverage − q| ≤ 0.05 theo nghĩa đen không đạt ở q=0.5–0.6 cho WindowAverage/TSB/LGBM-PoisNB/MLP-NB, và SeasonalNaive kém phủ ở đuôi – xem báo cáo 02.)

Mô phỏng, cost_ratio=10, q*(val) áp dụng cho test, L=2 (L=3 trong `05_final_test.md`):
| model | q_star_val | fill_val | fill_test | cost_val | cost_test | cost_point_test | gain_vs_point_val | gain_vs_point_test |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.8000 | 0.8740 | 0.8683 | 271.4933 | 283.2892 | 306.8226 | -0.0674 | -0.0767 |
| SeasonalNaive | 0.7000 | 0.8347 | 0.8377 | 305.1470 | 308.2199 | 320.7995 | -0.0207 | -0.0392 |
| CrostonSBA | 0.8000 | 0.8484 | 0.8476 | 301.1944 | 304.2625 | 335.2916 | -0.0759 | -0.0925 |
| TSB | 0.8000 | 0.8791 | 0.8799 | 269.3244 | 277.9247 | 301.0486 | -0.0729 | -0.0768 |
| LGBM-Q | 0.7000 | 0.8785 | 0.8744 | 255.8554 | 261.4253 | 283.9842 | -0.0733 | -0.0794 |
| LGBM-PoisNB | 0.8000 | 0.9002 | 0.9024 | 256.4374 | 259.9672 | 283.9842 | -0.0712 | -0.0846 |
| MLP-NB | 0.8000 | 0.9062 | 0.9034 | 252.4508 | 258.5030 | 275.8068 | -0.0546 | -0.0627 |

Chính sách theo nhóm (chốt từ val; cost_ratio=10, L=2; q = q_star trong từng dòng) – bảng đề xuất:
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

So sánh chính sách (val vs test):
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

Biểu đồ trong `results/figures/`: `policy_tradeoff_fill_vs_inventory.png` (đánh đổi fill rate–tồn kho, tách nhanh/trung bình/chậm), `policy_total_cost_by_q.png`, `policy_qstar_heatmap.png`, `policy_forecast_error_by_group.png`, `final_val_vs_test_forecast.png`, `final_cost_val_vs_test.png`.


### Bổ sung GĐ6 (chi tiết ở `06_bo_sung.md`)
- **Hiệu chuẩn rời rạc** (q ≤ 0.95, 7 mức, tổng thể): số q mà q nằm trong [P(y<q̂), P(y≤q̂)] ±0.05: 7/7 ở mọi mô hình trừ SeasonalNaive (3/7), ở cả val/test và H=3/4; số q thỏa |P(y≤q̂) − q| ≤ 0.05 (test, H=3): CrostonSBA 7, LGBM-Q 7, WindowAverage 6, TSB 5, LGBM-PoisNB 4, MLP-NB 4, SeasonalNaive 2.
- **MASE** có trong bảng dự báo chính phía trên (cột mase_*; scale chỉ trên train); H=4: `05_final_test.md`/`06_bo_sung.md` §8.
- **Lưới q mở rộng** (0.3, 0.4): q* ở tỷ lệ 1–2 chuyển từ 0.5 xuống 0.3 (cận lưới mới), chi phí test giảm khoảng 15–20% so với q*=0.5 (ví dụ L=2, tỷ lệ 1: WindowAverage 126.17 → 104.60).

## 6. Trả lời các câu hỏi nghiên cứu
Các KTC 95% là bootstrap theo chuỗi (2000 lần, seed 42) và chỉ phản ánh sai số lấy mẫu giữa các chuỗi trong cùng 28 ngày.

**Câu 1 – Phân vị hay điểm? (đã sửa theo GĐ6, xem `06_bo_sung.md`)** Kết luận: phần lớn lợi thế ban đầu của phân vị là lợi thế của việc có hệ số an toàn; phân vị chỉ còn hơn rõ ở tỷ lệ chi phí thiếu/giữ cao.
- So với chính sách điểm THUẦN (S = ceil(mean)), chính sách phân vị với q* chọn trên val rẻ hơn rõ trên test: KTC 95% dưới 0 ở 40/42 ô (7 mô hình × 6 tỷ lệ chi phí) cho cả L=2 và L=3 (ví dụ L=2, tỷ lệ 10: CrostonSBA −31.03, MLP-NB −17.30; tỷ lệ 50: −429.60 … −663.59 đơn vị chi phí).
- Nhưng khi chính sách điểm có hệ số an toàn S = ceil(k × mean), k ∈ {0.5, …, 3.0} chọn trên val cho từng (mô hình, tỷ lệ, L), nó cũng rẻ hơn điểm thuần ở 35/42 ô (L=2) và 37/42 ô (L=3), và gần ngang phân vị ở tỷ lệ trung bình: L=2, tỷ lệ 10, KTC của (điểm an toàn − phân vị) chứa 0 ở 5/7 mô hình; chênh nhỏ nhất có ý nghĩa là MLP-NB 2.11 [0.68; 3.77] (≈0.8% chi phí) và ngược chiều ở SeasonalNaive −2.59 [−3.81; −1.34].
- Phân vị còn hơn rõ ở tỷ lệ cao: tỷ lệ 50 (L=2) điểm an toàn đắt hơn phân vị 19.73 … 43.35 ở 6/7 mô hình (KTC dương); tỷ lệ 20 ở 5/7 mô hình. Ở tỷ lệ 1–2 điểm an toàn thường rẻ hơn hoặc ngang, nhưng cả k*=0.5 và q*=0.3 đều chạm cận lưới nên không kết luận chắc.
Test, L=2: chi phí[điểm an toàn k] − chi phí[phân vị q*] [KTC 95%] (dương = phân vị rẻ hơn; nhóm = toàn bộ 823 chuỗi; cột = tỷ lệ chi phí thiếu/giữ; k và q* chọn trên val cho từng ô):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -7.48 [-9.26; -5.81] | 2.46 [0.91; 4.08] | -0.09 [-1.22; 0.84] | 0.17 [-0.97; 1.40] | 10.04 [5.80; 14.82] | 36.70 [26.27; 48.45] |
| SeasonalNaive | -14.61 [-16.59; -12.68] | -2.92 [-4.80; -0.91] | 5.00 [2.65; 7.63] | -2.59 [-3.81; -1.34] | -3.29 [-6.54; 0.06] | -5.36 [-18.21; 5.58] |
| CrostonSBA | -4.91 [-6.49; -3.49] | 2.69 [1.26; 4.21] | 0.04 [-0.58; 0.65] | -0.04 [-1.42; 1.30] | 6.28 [2.08; 10.53] | 43.35 [30.88; 57.46] |
| TSB | -8.10 [-10.01; -6.35] | 2.74 [1.20; 4.26] | 0.88 [0.01; 1.64] | 0.69 [-0.48; 1.87] | 6.57 [3.05; 10.39] | 19.73 [9.10; 30.95] |
| LGBM-Q | -13.09 [-14.71; -11.50] | 1.54 [-0.31; 3.56] | 1.63 [0.64; 2.69] | -0.15 [-2.14; 1.76] | 3.43 [-1.23; 8.68] | 23.95 [12.89; 35.72] |
| LGBM-PoisNB | -6.79 [-8.55; -5.14] | 4.71 [2.92; 6.60] | 1.69 [0.63; 2.49] | 1.31 [-0.91; 3.78] | 5.17 [0.83; 10.16] | 21.00 [9.88; 33.04] |
| MLP-NB | -10.98 [-13.11; -8.91] | 3.08 [1.13; 5.14] | 3.79 [2.73; 4.88] | 2.11 [0.68; 3.77] | 8.67 [5.26; 12.60] | 26.52 [16.68; 37.17] |

Test, L=2: chi phí[phân vị q*] − chi phí[điểm thuần]:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -37.02 [-39.11; -35.01] | -22.97 [-24.56; -21.36] | 0.09 [-0.84; 1.22] | -23.53 [-29.59; -18.08] | -134.36 [-156.95; -115.18] | -561.10 [-638.70; -493.37] |
| SeasonalNaive | -41.59 [-43.93; -39.31] | -29.65 [-31.28; -27.97] | -3.15 [-4.00; -2.17] | -12.58 [-16.05; -9.36] | -96.14 [-109.83; -83.21] | -466.20 [-517.28; -416.74] |
| CrostonSBA | -39.61 [-42.67; -36.77] | -24.00 [-26.69; -21.53] | -0.04 [-0.65; 0.58] | -31.03 [-37.91; -24.73] | -160.06 [-184.21; -138.80] | -663.59 [-746.25; -590.66] |
| TSB | -38.56 [-40.58; -36.64] | -25.04 [-26.60; -23.47] | -0.88 [-1.64; -0.01] | -23.12 [-28.94; -17.98] | -128.64 [-149.80; -109.80] | -540.85 [-619.55; -471.12] |
| LGBM-Q | -31.32 [-32.97; -29.72] | -20.44 [-21.70; -19.16] | -1.63 [-2.69; -0.64] | -22.56 [-26.91; -18.73] | -120.20 [-136.87; -105.49] | -502.03 [-562.55; -449.19] |
| LGBM-PoisNB | -37.61 [-39.35; -36.04] | -23.61 [-24.92; -22.35] | -1.69 [-2.49; -0.63] | -24.02 [-29.80; -18.90] | -121.94 [-140.98; -105.48] | -499.08 [-561.94; -446.12] |
| MLP-NB | -36.80 [-38.60; -35.14] | -25.16 [-26.64; -23.72] | -2.00 [-2.88; -1.10] | -17.30 [-21.74; -13.44] | -98.63 [-113.73; -85.35] | -429.60 [-479.66; -385.23] |

Test, L=2: chi phí[điểm an toàn k] − chi phí[điểm thuần]:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -44.49 [-48.06; -41.06] | -20.51 [-22.82; -18.26] | 0.00 [0.00; 0.00] | -23.37 [-28.97; -18.16] | -124.32 [-144.45; -106.35] | -524.40 [-597.66; -460.19] |
| SeasonalNaive | -56.19 [-59.91; -52.49] | -32.57 [-35.34; -29.91] | 1.85 [-0.83; 4.88] | -15.17 [-18.03; -12.64] | -99.42 [-113.22; -86.26] | -471.56 [-524.87; -423.04] |
| CrostonSBA | -44.51 [-48.80; -40.51] | -21.31 [-24.81; -18.14] | 0.00 [0.00; 0.00] | -31.07 [-37.36; -25.43] | -153.78 [-178.16; -133.22] | -620.24 [-699.65; -549.85] |
| TSB | -46.65 [-50.35; -43.07] | -22.30 [-24.57; -20.10] | 0.00 [0.00; 0.00] | -22.43 [-27.96; -17.12] | -122.07 [-142.25; -104.31] | -521.12 [-594.69; -456.33] |
| LGBM-Q | -44.40 [-47.43; -41.44] | -18.90 [-21.18; -16.57] | 0.00 [0.00; 0.00] | -22.71 [-27.18; -18.78] | -116.77 [-133.84; -101.17] | -478.08 [-536.20; -426.83] |
| LGBM-PoisNB | -44.40 [-47.43; -41.44] | -18.90 [-21.18; -16.57] | 0.00 [0.00; 0.00] | -22.71 [-27.18; -18.78] | -116.77 [-133.84; -101.17] | -478.08 [-536.20; -426.83] |
| MLP-NB | -47.78 [-51.37; -44.34] | -22.08 [-24.36; -19.81] | 1.79 [0.61; 3.15] | -15.20 [-18.74; -11.98] | -89.96 [-103.57; -78.35] | -403.09 [-452.87; -359.89] |

Hạn chế: k và q* cùng chọn trên val; k là một hệ số chung cho mọi chuỗi; chi phí là giả định; chỉ 28 ngày; KTC bootstrap coi chuỗi độc lập. Ở cùng fill rate, trên val chính sách phân vị cần tồn kho gần bằng chính sách điểm (báo cáo 04), nhất quán với kết luận trên.

**Câu 2 – Nhanh vs chậm/gián đoạn.** Sai số tương đối lớn hơn nhiều ở hàng chậm: WAPE (test, H=3) nhanh/chậm của MLP-NB 0.2931/0.7320, TSB 0.3372/0.7494, SeasonalNaive 0.3627/0.9250. Lợi ích của chính sách phân vị theo q newsvendor (tỷ lệ 10, test, L=2) tập trung ở hàng nhanh: hiệu chi phí so với điểm của TSB là −44.34 [−63.49; −25.89] ở nhanh nhưng +15.35 [11.83; 19.01] ở chậm. Toàn bộ bảng [tỷ lệ chi phí 10, L=2, q=0.9 = q newsvendor; hiệu chi phí phân vị − điểm thuần, nhóm tốc độ bán; test]:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | -45.30 [-65.46; -25.59] | 5.04 [0.34; 9.81] | 11.32 [7.66; 15.33] |
| SeasonalNaive | 21.56 [3.95; 38.73] | 40.26 [34.77; 46.86] | 51.70 [44.37; 58.85] |
| CrostonSBA | -53.07 [-76.40; -30.68] | -4.35 [-9.17; 0.55] | 5.20 [1.85; 8.71] |
| TSB | -44.34 [-63.49; -25.89] | 7.28 [2.74; 11.81] | 15.35 [11.83; 19.01] |
| LGBM-Q | -26.94 [-43.25; -11.89] | 14.69 [9.55; 19.70] | 23.85 [19.36; 28.08] |
| LGBM-PoisNB | -28.89 [-45.60; -13.25] | 4.97 [0.39; 9.42] | 13.61 [10.07; 16.99] |
| MLP-NB | -11.71 [-26.86; 2.39] | 4.26 [-0.01; 8.73] | 9.25 [5.68; 12.60] |

Trên val, fill rate của chính sách chọn theo nhóm thấp hơn nhiều ở hàng chậm (intermittent_slow 0.8270, lumpy_slow 0.6966 so với smooth_fast 0.9388, L=2): tối ưu chi phí ở hàng chậm là chấp nhận mức phục vụ thấp hơn. Chính sách theo nhóm (9 nhóm) KHÔNG tốt hơn một cặp (mô hình, q) toàn cục trên test (theo nhóm − toàn cục = +2.311 [0.926; 4.047], L=2): lựa chọn theo nhóm bị overfit val. Khuyến nghị thực tế: dùng một mô hình và q toàn cục; q theo nhóm chỉ để tham khảo. Hạn chế: nhóm lumpy_slow chỉ 34 chuỗi, smooth_medium 2 chuỗi.

**Câu 3 – Học sâu xác suất vs Croston/TSB.** MLP-NB (một MLP, không phải DeepAR/N-HiTS/TFT) tốt hơn Croston/TSB ở cả val và test. Test, pinball_avg H=3: MLP-NB 0.9477, TSB 1.0799, CrostonSBA 1.2659; hiệu theo chuỗi MLP-NB − đối thủ [KTC]:
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: pinball_avg MLP-NB - CrostonSBA (H=3) | -0.3182 [-0.4210; -0.2305] | -0.6548 [-0.9119; -0.4414] | -0.1152 [-0.1561; -0.0818] | -0.0651 [-0.0940; -0.0405] |
| Q3: pinball_avg MLP-NB - LGBM-Q (H=3) | 0.0123 [0.0015; 0.0253] | 0.0320 [0.0059; 0.0646] | 0.0016 [-0.0039; 0.0071] | -0.0048 [-0.0097; 0.0001] |
| Q3: pinball_avg MLP-NB - TSB (H=3) | -0.1322 [-0.1695; -0.0996] | -0.2798 [-0.3727; -0.2000] | -0.0423 [-0.0555; -0.0297] | -0.0228 [-0.0322; -0.0138] |

Chi phí mô phỏng (cùng q newsvendor, tỷ lệ 10, L=2), MLP-NB − đối thủ:
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: MLP-NB - CrostonSBA (cung q_newsvendor) | -38.88 [-52.61; -27.20] | -86.83 [-120.66; -57.53] | -10.70 [-17.69; -4.93] | -1.47 [-5.75; 2.35] |
| Q3: MLP-NB - LGBM-Q (cung q_newsvendor) | -9.34 [-11.57; -6.90] | -3.00 [-7.95; 2.75] | -13.06 [-14.74; -11.47] | -14.29 [-16.59; -12.20] |
| Q3: MLP-NB - TSB (cung q_newsvendor) | -14.80 [-19.34; -10.73] | -28.24 [-38.95; -18.71] | -6.65 [-9.17; -4.25] | -4.75 [-7.43; -2.16] |

Tuy nhiên LightGBM phân vị (không phải học sâu) ngang hoặc hơn MLP-NB. Thứ hạng đổi giữa val (MLP-NB 0.9025 < LGBM-Q 0.9109, H=3) và test (LGBM-Q 0.9354 < MLP-NB 0.9477; hiệu +0.0123 [0.0015; 0.0253]), và KHÔNG do nhiễu seed: với 3 seed (42, 43, 44), pinball_avg (trung bình ± độ lệch chuẩn):
| split | H | n_seed | pinball_avg | wape | mase | epochs | kappa |
|---|---|---|---|---|---|---|---|
| test | 3 | 3 | 0.9477 ± 0.0021 | 0.3515 ± 0.0004 | 0.7843 ± 0.0005 | 17/25/20 | 0.5/0.5/0.5 |
| test | 4 | 3 | 1.1449 ± 0.0045 | 0.3170 ± 0.0006 | 0.8031 ± 0.0032 | 30/16/26 | 0.5/0.5/0.5 |
| val | 3 | 3 | 0.9026 ± 0.0005 | 0.3680 ± 0.0029 | 0.7559 ± 0.0043 | 17/25/20 | 0.5/0.5/0.5 |
| val | 4 | 3 | 1.0899 ± 0.0038 | 0.3315 ± 0.0022 | 0.7748 ± 0.0043 | 30/16/26 | 0.5/0.5/0.5 |

cả 3 seed tốt hơn LGBM-Q trên val và cả 3 kém hơn LGBM-Q trên test.
**Giải thích về biến ngoại sinh (đã sửa theo GĐ6)**: ablation bỏ giá, SNAP và sự kiện khỏi LightGBM-Q (30 → 24 đặc trưng) chỉ làm pinball_avg tăng ≤ 1% (val H=3 0.9109 → 0.9185; test H=3 0.9354 → 0.9385) và bản không ngoại sinh vẫn hơn hẳn TSB/CrostonSBA. Vì vậy lợi thế so với Croston/TSB KHÔNG đến chủ yếu từ giá/SNAP/sự kiện; nó đến từ đặc trưng lịch sử đa thang và thứ trong tuần (còn được giữ trong bản ablation) cùng việc huấn luyện trực tiếp phân phối của D_H. Chưa tách riêng yếu tố lịch và chưa có đối chứng Croston/TSB + biến ngoại sinh, nên chưa chứng minh được phần đóng góp của "học sâu" so với học máy cây.
Hội tụ MLP-NB: khi cho chạy tới 80 epoch, epoch tốt nhất là 58 (H=3) và 46 (H=4) nhưng NLL holdout gần như phẳng (H=4: chênh 0.0010 giữa epoch 30 và tốt nhất), nên kết quả với trần 30 epoch xấp xỉ hội tụ.

## 7. Hạn chế
- Chỉ 28 ngày val và 28 ngày test, một cửa hàng, một phòng ban; không lặp theo nhiều giai đoạn; KTC bootstrap coi các chuỗi độc lập.
- Siêu tham số (kappa, w, alpha Croston/TSB, q*, chính sách theo nhóm) chọn trên val nên số val lạc quan; test mới là ước lượng không chệch.
- Chi phí giữ/thiếu và quy tắc khởi tạo tồn kho là giả định tương đối; q* chạm cận dưới lưới ở tỷ lệ ≤5.
- Không có DeepAR/N-HiTS/TFT/Chronos-2; không có WRMSSE; LightGBM không refit theo `refit_every` trong từng giai đoạn; một seed duy nhất.
- Coverage theo nghĩa đen chưa đạt mục tiêu ở q thấp do tính rời rạc; SeasonalNaive kém hiệu chuẩn ở đuôi.
- Xếp hạng MLP-NB và LightGBM đổi chiều giữa val và test (không do nhiễu seed); MLP-NB H=4 dừng ở trần 30 epoch (cực tiểu NLL ở epoch 46, chênh rất nhỏ).
- Lưới q chính bắt đầu ở 0.5; lưới phụ 0.3/0.4 cho thấy q* ở tỷ lệ 1–2 chạm cận 0.3 (06); hệ số an toàn k cũng chạm cận 0.5 ở tỷ lệ 1–2.
- Hiệu chuẩn rời rạc: LGBM-Q cho phân vị liên tục nên khoảng [P(y<q̂), P(y≤q̂)] suy biến; dùng |P(y≤q̂) − q| ≤ 0.05 (đạt 7/7 ở mọi split, H).

## 8. Tài liệu tham khảo
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). The M5 accuracy competition: results, findings, and conclusions. International Journal of Forecasting 38(4). Dữ liệu: Kaggle "M5 Forecasting – Accuracy" (https://www.kaggle.com/c/m5-forecasting-accuracy).
- Croston, J. D. (1972). Forecasting and stock control for intermittent demands. Operational Research Quarterly 23(3). Syntetos, A. A., Boylan, J. E. (2005). The accuracy of intermittent demand estimates. International Journal of Forecasting 21(2). Teunter, R. H., Syntetos, A. A., Babai, M. Z. (2011). Intermittent demand: linking forecasting to inventory obsolescence. European Journal of Operational Research 214(3) (TSB).
- Nixtla StatsForecast (https://github.com/Nixtla/statsforecast) – tham chiếu cho Croston/TSB/WindowAverage/SeasonalNaive (mã dự án tự cài đặt phiên bản vector hóa). Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS.
- Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting 36(3) – chỉ làm tham chiếu cho loss NB; GluonTS/neuralforecast không được dùng.
- Hyndman, R. J., Koehler, A. B. (2006). Another look at measures of forecast accuracy (MASE). Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation (pinball/quantile loss).
- Guo, C., Berkhahn, F. (2016). Entity embeddings of categorical variables – chỉ liên quan nếu mở rộng MLP-NB bằng embedding mặt hàng; chưa dùng trong bản này.
