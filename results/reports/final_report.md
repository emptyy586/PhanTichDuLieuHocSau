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

WAPE theo speed (test, H=3):
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

Chính sách theo nhóm (chốt từ val, L=2) – bảng đề xuất:
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

## 6. Trả lời các câu hỏi nghiên cứu
Các KTC 95% là bootstrap theo chuỗi (2000 lần, seed 42) và chỉ phản ánh sai số lấy mẫu giữa các chuỗi trong cùng 28 ngày.

**Câu 1 – Phân vị hay điểm?** Có điều kiện. (i) Chọn q* theo chi phí rẻ hơn chính sách điểm của cùng mô hình ở mọi tỷ lệ chi phí trên test (thay đổi tương đối ở tỷ lệ 10: MLP-NB −0.063, TSB −0.077, CrostonSBA −0.093; ở tỷ lệ 50: −0.442 … −0.561; ở tỷ lệ 5 gần 0, từ 0.000 đến −0.014; bảng đầy đủ trong `05_final_test.md`). q* được chọn còn chính sách điểm không có tham số nào để chỉnh nên so sánh có lợi cho phân vị. (ii) So sánh KHÔNG tinh chỉnh (q = ratio/(1+ratio) làm tròn lên lưới), test, L=2, hiệu chi phí phân vị − điểm [KTC]:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -15.45 [-16.03; -14.90] | 14.83 [13.16; 16.68] | 26.96 [23.17; 30.85] | -13.56 [-22.41; -5.58] | -125.79 [-150.78; -104.20] | -561.10 [-638.70; -493.37] |
| SeasonalNaive | -11.24 [-12.11; -10.44] | 30.13 [28.25; 32.18] | 61.81 [57.15; 67.08] | 35.30 [27.89; 42.67] | -47.62 [-65.22; -29.53] | -441.91 [-497.13; -389.31] |
| CrostonSBA | -17.19 [-18.20; -16.26] | 14.77 [12.87; 16.81] | 26.03 [21.25; 30.86] | -21.60 [-31.37; -12.32] | -154.43 [-180.13; -131.04] | -663.59 [-746.25; -590.66] |
| TSB | -16.05 [-16.57; -15.54] | 14.52 [12.90; 16.26] | 27.49 [23.98; 30.89] | -11.44 [-19.79; -3.93] | -119.86 [-144.20; -98.47] | -540.85 [-619.55; -471.12] |
| LGBM-Q | -8.12 [-8.83; -7.48] | 22.42 [21.15; 23.71] | 36.03 [33.26; 38.96] | 0.16 [-7.11; 6.62] | -96.97 [-117.09; -79.21] | -491.24 [-555.43; -434.81] |
| LGBM-PoisNB | -16.39 [-17.04; -15.79] | 13.58 [12.21; 14.99] | 28.94 [25.70; 32.33] | -6.60 [-13.93; -0.50] | -102.11 [-122.20; -85.00] | -486.95 [-551.93; -431.27] |
| MLP-NB | -15.42 [-15.99; -14.84] | 13.32 [11.76; 15.03] | 27.75 [24.51; 31.16] | -1.00 [-7.58; 4.83] | -84.80 [-102.42; -69.98] | -430.20 [-484.08; -382.71] |

Tỷ lệ 2: phân vị đắt hơn điểm ở mọi mô hình; tỷ lệ 10: hỗn hợp (CrostonSBA, TSB, WindowAverage, LGBM-PoisNB rẻ hơn rõ; MLP-NB và LGBM-Q không khác biệt; SeasonalNaive đắt hơn); tỷ lệ 50: rẻ hơn rất nhiều ở mọi mô hình. (iii) Trên val, ở cùng fill rate, chính sách phân vị cần tồn kho gần như bằng chính sách điểm (`04_sensitivity_policy.md`): lợi ích chính là chọn được mức phục vụ cao khi thiếu hàng đắt, không phải hiệu quả tồn kho vượt trội ở cùng mức phục vụ. Hạn chế: chi phí là giả định; q* chạm cận dưới lưới (0.5) ở tỷ lệ ≤5; chỉ 28 ngày.

**Câu 2 – Nhanh vs chậm/gián đoạn.** Sai số tương đối lớn hơn nhiều ở hàng chậm: WAPE (test, H=3) nhanh/chậm của MLP-NB 0.2931/0.7320, TSB 0.3372/0.7494, SeasonalNaive 0.3627/0.9250. Lợi ích của chính sách phân vị theo q newsvendor (tỷ lệ 10, test, L=2) tập trung ở hàng nhanh: hiệu chi phí so với điểm của TSB là −44.34 [−63.49; −25.89] ở nhanh nhưng +15.35 [11.83; 19.01] ở chậm. Toàn bộ bảng:
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

Tuy nhiên LightGBM phân vị (không phải học sâu) ngang hoặc hơn MLP-NB: trên val MLP-NB hơn (0.9025 vs 0.9109), trên test LGBM-Q hơn (0.9354 vs 0.9477; hiệu +0.0123 [0.0015; 0.0253]). MLP-NB và LightGBM dùng cùng 30 đặc trưng (có giá, SNAP, sự kiện, thứ) còn Croston/TSB chỉ dùng lịch sử doanh số, nên lợi thế phần lớn đến từ biến ngoại sinh chứ chưa chứng minh là từ "học sâu"; chưa có đối chứng Croston/TSB + biến ngoại sinh.

## 7. Hạn chế
- Chỉ 28 ngày val và 28 ngày test, một cửa hàng, một phòng ban; không lặp theo nhiều giai đoạn; KTC bootstrap coi các chuỗi độc lập.
- Siêu tham số (kappa, w, alpha Croston/TSB, q*, chính sách theo nhóm) chọn trên val nên số val lạc quan; test mới là ước lượng không chệch.
- Chi phí giữ/thiếu và quy tắc khởi tạo tồn kho là giả định tương đối; q* chạm cận dưới lưới ở tỷ lệ ≤5.
- Không có DeepAR/N-HiTS/TFT/Chronos-2; không có WRMSSE; LightGBM không refit theo `refit_every` trong từng giai đoạn; một seed duy nhất.
- Coverage theo nghĩa đen chưa đạt mục tiêu ở q thấp do tính rời rạc; SeasonalNaive kém hiệu chuẩn ở đuôi.
- Xếp hạng MLP-NB và LightGBM không ổn định giữa val và test.

## 8. Tài liệu tham khảo
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). The M5 accuracy competition: results, findings, and conclusions. International Journal of Forecasting 38(4). Dữ liệu: Kaggle "M5 Forecasting – Accuracy" (https://www.kaggle.com/c/m5-forecasting-accuracy).
- Croston, J. D. (1972). Forecasting and stock control for intermittent demands. Operational Research Quarterly 23(3). Syntetos, A. A., Boylan, J. E. (2005). The accuracy of intermittent demand estimates. International Journal of Forecasting 21(2). Teunter, R. H., Syntetos, A. A., Babai, M. Z. (2011). Intermittent demand: linking forecasting to inventory obsolescence. European Journal of Operational Research 214(3) (TSB).
- Nixtla StatsForecast (https://github.com/Nixtla/statsforecast) – tham chiếu cho Croston/TSB/WindowAverage/SeasonalNaive (mã dự án tự cài đặt phiên bản vector hóa). Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS.
- Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting 36(3) – chỉ làm tham chiếu cho loss NB; GluonTS/neuralforecast không được dùng.
- Hyndman, R. J., Koehler, A. B. (2006). Another look at measures of forecast accuracy (MASE). Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation (pinball/quantile loss).
- Guo, C., Berkhahn, F. (2016). Entity embeddings of categorical variables – chỉ liên quan nếu mở rộng MLP-NB bằng embedding mặt hàng; chưa dùng trong bản này.
