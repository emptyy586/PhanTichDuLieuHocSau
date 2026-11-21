# Dự báo nhu cầu xác suất và mô phỏng chính sách bổ sung hàng tồn kho (Đề tài 17) – Bản nháp báo cáo (cập nhật GĐ7)

## 1. Giới thiệu
Mục tiêu: dự báo phân phối xác suất của nhu cầu từng mặt hàng và dùng phân vị dự báo để chọn mức đặt hàng (order-up-to) cân bằng giữa thiếu hàng và chi phí tồn kho. Ba câu hỏi nghiên cứu: (1) dự báo phân vị có chọn mức tồn kho tốt hơn dự báo điểm không; (2) kết quả khác nhau thế nào giữa hàng bán nhanh và hàng bán chậm/gián đoạn; (3) học sâu xác suất có hơn Croston/TSB không. Toàn bộ mã, kết quả và các báo cáo trung gian (`results/reports/00…07`) nằm trong repo; mọi con số dưới đây lấy từ file kết quả đã sinh (số liệu trước GĐ7 được lưu ở `archive/pre_gd7/`).

## 2. Dữ liệu và chia dữ liệu
- M5 (Walmart), cửa hàng CA_1 × phòng ban FOODS_3: 823 chuỗi, 1941 ngày (2011-01-29 → 2016-05-22). Train 2011-01-29 → 2016-03-27 (1885 ngày), validation 2016-03-28 → 2016-04-24 (28 ngày), test 2016-04-25 → 2016-05-22 (28 ngày). Giá bán, sự kiện, SNAP của các ngày tương lai coi là đã biết (19.8606% giá khuyết – trước khi mặt hàng được niêm yết).
- Phân loại chuỗi (chỉ tính trên train): demand_type theo ADI/CV² (ngưỡng 1.32/0.49) và speed theo tam phân vị doanh số trung bình. Số chuỗi: smooth 120, erratic 45, intermittent 419, lumpy 239; fast 326, medium 321, slow 176 (bảng đầy đủ: `results/reports/01_eda.md`).
- EDA (train, sau ngày ra mắt): mùa vụ tuần rõ (Chủ nhật 4.1099 vs Thứ 4 2.5597 đơn vị/chuỗi/ngày), ngày SNAP cao hơn 11.9%, sự kiện phân hóa mạnh (SuperBowl chỉ số 1.3460, Thanksgiving 0.6986, Christmas 0 vì đóng cửa).
- Chống rò rỉ: đặc trưng tại gốc t chỉ dùng doanh số đến hết ngày t (có test đơn vị); các chỉ số hiệu ứng thứ/sự kiện chỉ tính đến hết giai đoạn đã biết (xem hạn chế về mã hóa mục tiêu trong mẫu); phân loại và ngưỡng speed chỉ từ train; test chỉ dùng để đánh giá sau khi mọi siêu tham số, mô hình, q*, k* và chính sách đã chốt bằng validation (hparams.json sha256 `8bc7d7507bf0167e…` dùng cho lần chạy test cuối).

## 3. Phương pháp
**Mục tiêu dự báo**: tổng nhu cầu cửa sổ D_H(t) = y[t+1] + … + y[t+H], H = lead time + chu kỳ đặt (1) ∈ {3, 4}, đánh giá bằng rolling-origin thật (26 gốc với H=3, 25 gốc với H=4 trong mỗi giai đoạn 28 ngày).
**Mô hình** (7): WindowAverage(28), SeasonalNaive(lag 7), CrostonSBA, TSB – mỗi mô hình cho trung bình cửa sổ và được bọc bằng phân phối Negative Binomial (độ phân tán alpha ước lượng cuốn chiếu từ phần dư quá khứ, trộn với nhóm demand_type × speed, nhân hệ số hiệu chuẩn kappa chọn trên val; phân vị chính xác bằng `scipy.stats.nbinom.ppf`); LightGBM phân vị (một mô hình/phân vị, huấn luyện trực tiếp trên D_H, 30 đặc trưng gồm lag/rolling, giá, thay đổi giá, SNAP, sự kiện, thứ, speed/demand_type); LightGBM-Poisson + NB; MLP-NB (mạng MLP hai đầu ra NB(mu, alpha), NLL trực tiếp trên D_H). Lưu ý: MLP-NB không phải DeepAR/N-HiTS/TFT (các thư viện neuralforecast/gluonts chỉ cho phân vị biên theo ngày, mà tổng phân vị ngày khác phân vị của tổng); ở GĐ8 đã bổ sung một DeepAR-style tự cài đặt bằng PyTorch (lấy mẫu tự hồi quy để có phân phối của tổng cửa sổ). Chronos-2 không làm. Lưới phân vị chính: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99; lưới phụ cho mô phỏng: 0.3, 0.4. Lưới kappa: 0.25, 0.35, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0.
**Chỉ số dự báo**: pinball (từng q và trung bình), coverage P(y≤Q) và P(y<Q), WAPE (trên trung bình cửa sổ), MASE (chia cho MAE trong mẫu train của dự báo naive theo cửa sổ). WRMSSE không làm vì định nghĩa M5 cần dự báo ngày 28 bước và 12 mức gộp, không tái hiện được ở một cửa hàng × một phòng ban với dự báo cửa sổ.

## 4. Mô phỏng tồn kho và giả định chi phí
Chính sách order-up-to hằng ngày: cuối ngày t đặt hàng với S = mức order-up-to, đặt max(0, S − tồn kho vị trí); hàng về cuối ngày t+L và dùng được từ ngày t+L+1; L ∈ {2, 3}; thiếu hàng bị mất (lost sales); tồn đầu = S ở gốc đầu tiên, không có hàng đang về. **Làm tròn (GĐ7)**: phân vị NB đã là số nguyên nên dùng nguyên giá trị; phân vị LightGBM-Q là số thực nên làm tròn về số nguyên gần nhất, S = max(0, ⌊q̂+0.5⌋) (trước GĐ7 dùng ceil cho mọi mô hình và làm LGBM-Q giữ nhiều tồn kho hơn các mô hình khác ở cùng q). Chính sách điểm: S = ceil(trung bình dự báo của D_H) (k=1) và S = ceil(k × trung bình) (k∈{0.5,…,3.0}); ceil được giữ ở chính sách điểm cho MỌI mô hình (nhất quán giữa các mô hình, nhưng ceil cộng trung bình +0.5 đơn vị so với làm tròn gần nhất nên làm điểm THUẦN k=1 hơi bảo thủ hơn). Chi phí **tương đối, không phải chi phí thực**: giữ 1 đơn vị-ngày = 1.0; thiếu 1 đơn vị = cost_ratio (cơ sở 10; quét 1, 2, 5, 10, 20, 50). Chỉ số: fill rate (tổng bán/tổng cầu), số ngày hết hàng, tồn kho cuối ngày trung bình, chi phí giữ, thiếu, tổng; tách theo speed, demand_type. Chi tiết quy ước: `results/reports/03_inventory_sim.md`. Mô phỏng được kiểm bằng một bản tham chiếu viết độc lập (`tests/test_inventory_reference.py`, L=2 và 3, khớp).

## 5. Kết quả (test; val ở các báo cáo 02–04, 07)
Dự báo, H=3, tổng thể (validation vs test; pinball_avg, WAPE, MASE; không phụ thuộc tỷ lệ chi phí/L/q; MLP-NB = seed 42, mô hình chính chốt từ GĐ2; trung bình 3 seed ở phần Bổ sung GĐ8):
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

Coverage P(y≤Q) test (H=3):
| model | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 | 0.975 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.5694 | 0.6488 | 0.7303 | 0.8092 | 0.8499 | 0.8906 | 0.9318 | 0.9539 | 0.9688 |
| SeasonalNaive | 0.5518 | 0.5944 | 0.6655 | 0.7356 | 0.7677 | 0.7974 | 0.8354 | 0.8634 | 0.8805 |
| CrostonSBA | 0.5165 | 0.6050 | 0.6960 | 0.7882 | 0.8346 | 0.8804 | 0.9278 | 0.9522 | 0.9691 |
| TSB | 0.5751 | 0.6607 | 0.7407 | 0.8189 | 0.8598 | 0.9017 | 0.9415 | 0.9617 | 0.9769 |
| LGBM-Q | 0.5127 | 0.6069 | 0.7062 | 0.8064 | 0.8553 | 0.9027 | 0.9476 | 0.9720 | 0.9874 |
| LGBM-PoisNB | 0.5872 | 0.6704 | 0.7552 | 0.8401 | 0.8790 | 0.9176 | 0.9533 | 0.9724 | 0.9844 |
| MLP-NB | 0.6004 | 0.6787 | 0.7571 | 0.8359 | 0.8739 | 0.9105 | 0.9493 | 0.9702 | 0.9834 |

**Hiệu chuẩn rời rạc** (q ≤ 0.95, 7 mức, tổng thể): số q mà q nằm trong [P(y<q̂), P(y≤q̂)] ±0.05: 7/7 ở mọi mô hình trừ SeasonalNaive (3–3/7); số q thỏa |P(y≤q̂) − q| ≤ 0.05 (test, H=3): CrostonSBA 7, LGBM-Q 7, WindowAverage 6, TSB 5, LGBM-PoisNB 4, MLP-NB 4, SeasonalNaive 2. LGBM-Q cho phân vị liên tục nên khoảng suy biến thành một điểm. SeasonalNaive kém phủ ở đuôi (coverage 0.880 tại q=0.99, H=3). Chi tiết: `06_bo_sung.md`, `07_sua_loi.md`.

Mô phỏng, cost_ratio=10, q*(val) áp dụng cho test, L=2 (L=3 trong `05_final_test.md`):
| model | q_star_val | fill_val | fill_test | cost_val | cost_test | cost_point_test | gain_vs_point_val | gain_vs_point_test |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.8000 | 0.8740 | 0.8683 | 271.4933 | 283.2892 | 306.8226 | -0.0674 | -0.0767 |
| SeasonalNaive | 0.7000 | 0.8347 | 0.8377 | 305.1470 | 308.2199 | 320.7995 | -0.0207 | -0.0392 |
| CrostonSBA | 0.8000 | 0.8529 | 0.8522 | 301.4739 | 304.2539 | 335.2916 | -0.0751 | -0.0926 |
| TSB | 0.8000 | 0.8827 | 0.8835 | 269.6221 | 278.0923 | 301.0486 | -0.0719 | -0.0763 |
| LGBM-Q | 0.8000 | 0.9069 | 0.9028 | 254.4751 | 258.5018 | 283.9842 | -0.0783 | -0.0897 |
| LGBM-PoisNB | 0.8000 | 0.9014 | 0.9035 | 256.5772 | 260.0826 | 283.9842 | -0.0707 | -0.0842 |
| MLP-NB | 0.8000 | 0.9062 | 0.9034 | 252.4508 | 258.5030 | 275.8068 | -0.0546 | -0.0627 |

Chính sách theo nhóm (chốt từ val; cost_ratio=10, L=2; q = q_star trong từng dòng) – bảng đề xuất:
| group | n_series_group | model | q_star | fill_rate | stockout_days | avg_inventory | total_cost | ghi_chu |
|---|---|---|---|---|---|---|---|---|
| erratic_fast | 45 | LGBM-Q | 0.8000 | 0.8936 | 3.1111 | 11.8063 | 476.3556 |  |
| intermittent_fast | 74 | LGBM-Q | 0.8000 | 0.9298 | 2.9054 | 9.4643 | 354.5946 |  |
| intermittent_medium | 203 | MLP-NB | 0.8000 | 0.8680 | 2.5665 | 3.5422 | 146.7685 |  |
| intermittent_slow | 142 | MLP-NB | 0.8000 | 0.8270 | 1.9085 | 1.9298 | 80.5141 |  |
| lumpy_fast | 89 | MLP-NB | 0.7000 | 0.8407 | 3.4382 | 7.8752 | 354.5506 |  |
| lumpy_medium | 116 | MLP-NB | 0.8000 | 0.8354 | 2.5172 | 3.9544 | 169.6897 |  |
| lumpy_slow | 34 | TSB | 0.7000 | 0.6884 | 2.7941 | 1.5977 | 100.9118 |  |
| smooth_fast | 118 | MLP-NB | 0.8000 | 0.9388 | 2.9068 | 14.7022 | 537.0000 |  |
| smooth_medium | 2 | MLP-NB | 0.8000 | 0.8568 | 2.5607 | 3.6943 | 155.2461 | dùng nhóm speed=medium |

So sánh chính sách (val vs test; cost_ratio=10; A = theo nhóm, B = toàn cục, C = điểm thuần):
| L | policy | split | fill_rate | stockout_days | avg_inventory | total_cost |
|---|---|---|---|---|---|---|
| 2 | A_theo_nhom | val | 0.8991 | 2.6634 | 6.2960 | 251.8773 |
| 2 | B_toan_cuc (MLP-NB, q=0.8) | val | 0.9062 | 2.5152 | 6.5066 | 252.4508 |
| 2 | C_diem (MLP-NB) | val | 0.8240 | 4.5176 | 4.8282 | 267.0243 |
| 2 | A_theo_nhom | test | 0.8948 | 2.7132 | 6.2469 | 259.2746 |
| 2 | B_toan_cuc (MLP-NB, q=0.8) | test | 0.9034 | 2.5358 | 6.4662 | 258.5030 |
| 2 | C_diem (MLP-NB) | test | 0.8210 | 4.6209 | 4.7235 | 275.8068 |
| 3 | A_theo_nhom | val | 0.8707 | 3.2576 | 6.1720 | 269.6926 |
| 3 | B_toan_cuc (MLP-NB, q=0.7) | val | 0.8695 | 3.4241 | 6.1531 | 270.0267 |
| 3 | C_diem (MLP-NB) | val | 0.8276 | 4.2661 | 5.3586 | 279.1786 |
| 3 | A_theo_nhom | test | 0.8632 | 3.2807 | 6.0981 | 280.4557 |
| 3 | B_toan_cuc (MLP-NB, q=0.7) | test | 0.8633 | 3.4058 | 6.0336 | 278.5772 |
| 3 | C_diem (MLP-NB) | test | 0.8183 | 4.3888 | 5.1852 | 290.9210 |

Biểu đồ trong `results/figures/`: `policy_tradeoff_fill_vs_inventory.png`, `policy_total_cost_by_q.png`, `policy_qstar_heatmap.png`, `policy_forecast_error_by_group.png`, `final_val_vs_test_forecast.png`, `final_cost_val_vs_test.png`, `mlp_nll_curves.png`.

## 6. Trả lời các câu hỏi nghiên cứu
Các KTC 95% là bootstrap theo chuỗi (2000 lần, seed 42) và chỉ phản ánh sai số lấy mẫu giữa các chuỗi trong cùng 28 ngày. Trong mục này: "điểm thuần" = S = ceil(mean) (k=1); "điểm an toàn" = S = ceil(k·mean) với k chọn trên val; "phân vị" = q* chọn trên val (lưới 0.3–0.99).

**Câu 1 – Phân vị hay điểm?** Phần lớn lợi thế của phân vị so với điểm thuần là lợi thế của việc có hệ số an toàn. So với điểm an toàn, phân vị cho chênh lệch nhỏ và hỗn hợp ở tỷ lệ chi phí thiếu/giữ 2–10 (chênh có ý nghĩa tối đa 5.42 đơn vị chi phí trên mức ≈250), điểm an toàn rẻ hơn ở tỷ lệ 1, và phân vị chỉ hơn lớn ở tỷ lệ ≥20.
- So với điểm THUẦN, phân vị (q*) rẻ hơn rõ trên test: KTC dưới 0 ở 39/42 ô (7 mô hình × 6 tỷ lệ chi phí) với L=2 và 40/42 với L=3 (ví dụ L=2, tỷ lệ 10: CrostonSBA -31.04 [-38.22; -24.52], MLP-NB -17.30 [-21.74; -13.44]; tỷ lệ 50: -660.25 … -429.60).
- Điểm an toàn (k* chọn trên val) cũng rẻ hơn điểm thuần ở 35/42 ô (L=2) và 37/42 ô (L=3).
- Điểm an toàn so với phân vị (k* − q*; dương = phân vị rẻ hơn), tính trên 42 ô mỗi L: L=2 có 9 ô điểm an toàn rẻ hơn, 10 ô không phân biệt được, 23 ô phân vị rẻ hơn; L=3: 14 / 13 / 15.
  - **Tỷ lệ 1**: điểm an toàn rẻ hơn ở 7/7 mô hình (L=2; ví dụ SeasonalNaive -14.61 [-16.59; -12.68]).
  - **Tỷ lệ 2 (L=2)**: phân vị rẻ hơn điểm an toàn rõ ở 6/7 mô hình (WindowAverage, CrostonSBA, TSB, LGBM-Q, LGBM-PoisNB, MLP-NB; ví dụ WindowAverage 2.46 [0.91; 4.08], MLP-NB 3.08 [1.13; 5.14]), điểm an toàn rẻ hơn ở: SeasonalNaive. (L=3 tỷ lệ 2: điểm an toàn rẻ hơn ở WindowAverage, SeasonalNaive, TSB; không phân biệt được ở CrostonSBA, LGBM-Q, LGBM-PoisNB, MLP-NB.)
  - **Tỷ lệ 10 (L=2)**: không phân biệt được ở 4/7 mô hình (WindowAverage, CrostonSBA, TSB, LGBM-PoisNB); phân vị rẻ hơn ở LGBM-Q 2.77 [0.82; 4.99]; MLP-NB 2.11 [0.68; 3.77]; điểm an toàn rẻ hơn ở SeasonalNaive -2.59 [-3.81; -1.34].
  - **Tỷ lệ ≥20 (L=2)**: phân vị rẻ hơn rõ ở 5/7 mô hình (tỷ lệ 20) và 6/7 mô hình (tỷ lệ 50; chênh 16.21 … 40.01 đơn vị chi phí).
  - Ở tỷ lệ 1–2 cả k*=0.5 (cận lưới k) lẫn q*=0.3 (cận lưới q) đều chạm cận lưới nên so sánh bị chặn ở cả hai phía.
Số mô hình (trên 7) theo kết quả so sánh k* − q* (KTC 95%) cho từng tỷ lệ chi phí:
| cost_ratio | L2: điểm an toàn rẻ hơn | L2: không phân biệt | L2: phân vị rẻ hơn | L3: điểm an toàn rẻ hơn | L3: không phân biệt | L3: phân vị rẻ hơn |
|---|---|---|---|---|---|---|
| 1 | 7 | 0 | 0 | 7 | 0 | 0 |
| 2 | 1 | 0 | 6 | 3 | 4 | 0 |
| 5 | 0 | 3 | 4 | 0 | 1 | 6 |
| 10 | 1 | 4 | 2 | 3 | 3 | 1 |
| 20 | 0 | 2 | 5 | 1 | 4 | 2 |
| 50 | 0 | 1 | 6 | 0 | 1 | 6 |

Test, L=2: chi phí[điểm an toàn k*] − chi phí[phân vị q*] [KTC 95%] (dương = phân vị rẻ hơn; toàn bộ 823 chuỗi; cột = tỷ lệ chi phí; k*, q* chọn trên val cho từng ô):
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -7.48 [-9.26; -5.81] | 2.46 [0.91; 4.08] | -0.09 [-1.22; 0.84] | 0.17 [-0.97; 1.40] | 10.04 [5.80; 14.82] | 36.70 [26.27; 48.45] |
| SeasonalNaive | -14.61 [-16.59; -12.68] | -2.92 [-4.80; -0.91] | 5.00 [2.65; 7.63] | -2.59 [-3.81; -1.34] | -3.29 [-6.54; 0.06] | -5.36 [-18.21; 5.58] |
| CrostonSBA | -3.20 [-4.61; -1.93] | 3.29 [2.00; 4.72] | -0.09 [-0.74; 0.54] | -0.03 [-1.63; 1.53] | 6.88 [2.90; 11.30] | 40.01 [25.61; 54.99] |
| TSB | -6.80 [-8.61; -5.14] | 3.20 [1.72; 4.68] | 0.84 [-0.06; 1.61] | 0.52 [-0.79; 1.95] | 7.27 [3.42; 11.82] | 16.21 [5.05; 27.92] |
| LGBM-Q | -6.49 [-8.19; -4.82] | 5.42 [3.53; 7.41] | 2.10 [0.94; 3.32] | 2.77 [0.82; 4.99] | 0.76 [-4.05; 5.96] | 23.39 [12.15; 35.23] |
| LGBM-PoisNB | -6.30 [-8.00; -4.70] | 4.90 [3.15; 6.78] | 1.76 [0.68; 2.57] | 1.19 [-1.09; 3.76] | 5.07 [0.61; 10.14] | 21.16 [9.89; 33.30] |
| MLP-NB | -10.98 [-13.11; -8.91] | 3.08 [1.13; 5.14] | 3.79 [2.73; 4.88] | 2.11 [0.68; 3.77] | 8.67 [5.26; 12.60] | 26.52 [16.68; 37.17] |

Test, L=2: chi phí[phân vị q*] − chi phí[điểm thuần]:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -37.02 [-39.11; -35.01] | -22.97 [-24.56; -21.36] | 0.09 [-0.84; 1.22] | -23.53 [-29.59; -18.08] | -134.36 [-156.95; -115.18] | -561.10 [-638.70; -493.37] |
| SeasonalNaive | -41.59 [-43.93; -39.31] | -29.65 [-31.28; -27.97] | -3.15 [-4.00; -2.17] | -12.58 [-16.05; -9.36] | -96.14 [-109.83; -83.21] | -466.20 [-517.28; -416.74] |
| CrostonSBA | -41.32 [-44.62; -38.26] | -24.60 [-27.47; -21.91] | 0.09 [-0.54; 0.74] | -31.04 [-38.22; -24.52] | -160.66 [-185.02; -139.04] | -660.25 [-745.09; -586.25] |
| TSB | -39.86 [-42.02; -37.80] | -25.51 [-27.14; -23.84] | -0.84 [-1.61; 0.06] | -22.96 [-29.08; -17.50] | -129.34 [-151.36; -110.03] | -537.32 [-618.04; -466.32] |
| LGBM-Q | -37.91 [-39.55; -36.37] | -24.32 [-25.67; -22.96] | -2.10 [-3.32; -0.94] | -25.48 [-31.20; -20.37] | -117.52 [-133.93; -103.35] | -501.46 [-561.30; -448.74] |
| LGBM-PoisNB | -38.11 [-39.90; -36.48] | -23.80 [-25.16; -22.49] | -1.76 [-2.57; -0.68] | -23.90 [-29.79; -18.79] | -121.84 [-140.82; -105.20] | -499.24 [-561.83; -445.81] |
| MLP-NB | -36.80 [-38.60; -35.14] | -25.16 [-26.64; -23.72] | -2.00 [-2.88; -1.10] | -17.30 [-21.74; -13.44] | -98.63 [-113.73; -85.35] | -429.60 [-479.66; -385.23] |

Test, L=2: chi phí[điểm an toàn k*] − chi phí[điểm thuần]:
| model | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 | 50.0 |
|---|---|---|---|---|---|---|
| WindowAverage | -44.49 [-48.06; -41.06] | -20.51 [-22.82; -18.26] | 0.00 [0.00; 0.00] | -23.37 [-28.97; -18.16] | -124.32 [-144.45; -106.35] | -524.40 [-597.66; -460.19] |
| SeasonalNaive | -56.19 [-59.91; -52.49] | -32.57 [-35.34; -29.91] | 1.85 [-0.83; 4.88] | -15.17 [-18.03; -12.64] | -99.42 [-113.22; -86.26] | -471.56 [-524.87; -423.04] |
| CrostonSBA | -44.51 [-48.80; -40.51] | -21.31 [-24.81; -18.14] | 0.00 [0.00; 0.00] | -31.07 [-37.36; -25.43] | -153.78 [-178.16; -133.22] | -620.24 [-699.65; -549.85] |
| TSB | -46.65 [-50.35; -43.07] | -22.30 [-24.57; -20.10] | 0.00 [0.00; 0.00] | -22.43 [-27.96; -17.12] | -122.07 [-142.25; -104.31] | -521.12 [-594.69; -456.33] |
| LGBM-Q | -44.40 [-47.43; -41.44] | -18.90 [-21.18; -16.57] | 0.00 [0.00; 0.00] | -22.71 [-27.18; -18.78] | -116.77 [-133.84; -101.17] | -478.08 [-536.20; -426.83] |
| LGBM-PoisNB | -44.40 [-47.43; -41.44] | -18.90 [-21.18; -16.57] | 0.00 [0.00; 0.00] | -22.71 [-27.18; -18.78] | -116.77 [-133.84; -101.17] | -478.08 [-536.20; -426.83] |
| MLP-NB | -47.78 [-51.37; -44.34] | -22.08 [-24.36; -19.81] | 1.79 [0.61; 3.15] | -15.20 [-18.74; -11.98] | -89.96 [-103.57; -78.35] | -403.09 [-452.87; -359.89] |

Hạn chế: k và q* cùng chọn trên val; k chung cho mọi chuỗi (xem câu 2 cho k theo nhóm tốc độ); chi phí là giả định; chỉ 28 ngày; KTC bootstrap coi chuỗi độc lập; điểm thuần dùng ceil (xem mục 4).

**Câu 2 – Nhanh vs chậm/gián đoạn.** Sai số tương đối lớn hơn nhiều ở hàng chậm (WAPE test, H=3 ở bảng trên). Về chính sách, lợi thế của phân vị ở hàng nhanh phần lớn biến mất khi so với điểm an toàn. Bảng dưới (test, **tỷ lệ chi phí 10, L=2**; k* và q* chọn trên val RIÊNG trong từng nhóm tốc độ; hiệu chi phí TB/chuỗi, [KTC 95% bootstrap theo chuỗi trong nhóm]):
| group | model | n | k*_val | q*_val | chi_phi_diem_thuan | chi_phi_diem_an_toan_k* | chi_phi_phan_vi_q* | k* - q* [KTC] | q* - điểm thuần [KTC] |
|---|---|---|---|---|---|---|---|---|---|
| fast | CrostonSBA | 326 | 1.30 | 0.80 | 614.68 | 548.46 | 547.75 | 0.72 [-3.27; 4.61] | -66.94 [-83.88; -51.31] |
| fast | LGBM-PoisNB | 326 | 1.20 | 0.80 | 504.73 | 454.85 | 452.15 | 2.71 [-2.69; 8.61] | -52.58 [-66.02; -40.06] |
| fast | LGBM-Q | 326 | 1.20 | 0.80 | 504.73 | 454.85 | 448.17 | 6.68 [2.29; 11.78] | -56.55 [-69.80; -44.50] |
| fast | MLP-NB | 326 | 1.20 | 0.80 | 486.50 | 455.58 | 450.33 | 5.26 [2.04; 8.70] | -36.18 [-46.12; -26.61] |
| fast | SeasonalNaive | 326 | 1.10 | 0.70 | 562.58 | 530.76 | 529.46 | 1.29 [-1.45; 4.16] | -33.11 [-41.33; -25.78] |
| fast | TSB | 326 | 1.30 | 0.80 | 547.38 | 495.33 | 492.60 | 2.73 [-0.33; 6.08] | -54.78 [-68.37; -41.82] |
| fast | WindowAverage | 326 | 1.30 | 0.80 | 558.14 | 505.73 | 503.41 | 2.32 [-0.37; 5.12] | -54.73 [-67.96; -41.79] |
| medium | CrostonSBA | 321 | 1.40 | 0.80 | 181.26 | 169.52 | 169.54 | -0.02 [-0.76; 0.73] | -11.72 [-15.03; -8.45] |
| medium | LGBM-PoisNB | 321 | 1.30 | 0.80 | 164.59 | 155.20 | 155.87 | -0.67 [-1.52; 0.15] | -8.72 [-12.04; -5.50] |
| medium | LGBM-Q | 321 | 1.30 | 0.80 | 164.59 | 155.20 | 155.45 | -0.25 [-1.53; 1.05] | -9.13 [-12.59; -5.72] |
| medium | MLP-NB | 321 | 1.30 | 0.80 | 161.96 | 154.75 | 154.86 | -0.11 [-0.96; 0.72] | -7.10 [-10.04; -4.18] |
| medium | SeasonalNaive | 321 | 1.10 | 0.70 | 190.73 | 182.73 | 188.48 | -5.75 [-6.84; -4.69] | -2.24 [-4.23; -0.14] |
| medium | TSB | 321 | 1.30 | 0.80 | 165.59 | 159.65 | 160.29 | -0.64 [-1.72; 0.45] | -5.30 [-8.23; -2.46] |
| medium | WindowAverage | 321 | 1.30 | 0.80 | 168.10 | 161.17 | 162.04 | -0.88 [-1.74; 0.01] | -6.06 [-8.77; -3.27] |
| slow | CrostonSBA | 176 | 1.30 | 0.80 | 98.70 | 97.31 | 98.94 | -1.63 [-2.86; -0.48] | 0.23 [-1.65; 2.29] |
| slow | LGBM-PoisNB | 176 | 1.40 | 0.80 | 92.87 | 95.35 | 94.39 | 0.95 [0.05; 1.93] | 1.52 [-0.78; 3.63] |
| slow | LGBM-Q | 176 | 1.40 | 0.80 | 92.87 | 95.35 | 95.12 | 0.23 [-1.41; 1.78] | 2.25 [-0.30; 4.65] |
| slow | MLP-NB | 176 | 1.30 | 0.80 | 93.18 | 93.05 | 92.23 | 0.82 [-0.07; 1.77] | -0.95 [-3.19; 1.10] |
| slow | SeasonalNaive | 176 | 0.70 | 0.50 | 110.20 | 106.49 | 108.94 | -2.45 [-3.66; -1.32] | -1.26 [-2.76; 0.36] |
| slow | TSB | 176 | 1.30 | 0.70 | 91.84 | 94.19 | 92.66 | 1.52 [-0.16; 3.24] | 0.83 [-0.47; 2.08] |
| slow | WindowAverage | 176 | 1.30 | 0.80 | 94.32 | 94.79 | 96.70 | -1.91 [-3.01; -0.89] | 2.38 [0.45; 4.50] |

- **Hàng nhanh** (n=326): so với điểm thuần, phân vị rẻ hơn rõ ở 7/7 mô hình (q* − điểm thuần: -66.94 … -33.11); nhưng so với điểm an toàn k*, chênh lệch k* − q* chỉ 0.72 … 6.68 (trên mức chi phí 448.2 … 547.7); phân vị rẻ hơn rõ chỉ ở: LGBM-Q 6.68 [2.29; 11.78], MLP-NB 5.26 [2.04; 8.70]; không phân biệt được ở 5/7 mô hình.
- **Hàng trung bình** (n=321): k* − q* không phân biệt được ở 6/7 mô hình; điểm an toàn rẻ hơn rõ ở: SeasonalNaive -5.75 [-6.84; -4.69].
- **Hàng chậm** (n=176): điểm an toàn rẻ hơn hoặc ngang phân vị: điểm an toàn rẻ hơn rõ ở WindowAverage -1.91 [-3.01; -0.89], SeasonalNaive -2.45 [-3.66; -1.32], CrostonSBA -1.63 [-2.86; -0.48]; phân vị rẻ hơn rõ ở LGBM-PoisNB 0.95 [0.05; 1.93]; không phân biệt được ở 3/7 mô hình. Phân vị đắt hơn điểm thuần rõ ở 1/7 mô hình.
- Bảng cũ (q = 0.9 CHƯA tinh chỉnh − điểm thuần, tỷ lệ 10, L=2) chỉ để tham khảo, lẫn lợi ích của hệ số an toàn:
| model | fast | medium | slow |
|---|---|---|---|
| WindowAverage | -45.30 [-65.46; -25.59] | 5.04 [0.34; 9.81] | 11.32 [7.66; 15.33] |
| SeasonalNaive | 21.56 [3.95; 38.73] | 40.26 [34.77; 46.86] | 51.70 [44.37; 58.85] |
| CrostonSBA | -45.47 [-69.61; -21.75] | -2.25 [-7.29; 2.88] | 6.31 [2.89; 9.85] |
| TSB | -39.41 [-59.58; -20.18] | 9.41 [4.68; 14.06] | 16.64 [12.95; 20.38] |
| LGBM-Q | -33.88 [-49.71; -18.96] | 6.30 [1.51; 11.24] | 14.95 [10.98; 18.71] |
| LGBM-PoisNB | -26.85 [-43.60; -11.23] | 5.90 [1.29; 10.42] | 13.91 [10.29; 17.31] |
| MLP-NB | -11.71 [-26.86; 2.39] | 4.26 [-0.01; 8.73] | 9.25 [5.68; 12.60] |

Kết luận câu 2: hàng nhanh được lợi nhiều từ việc đặt mức cao hơn trung bình (cả phân vị lẫn hệ số an toàn k*>1; q* − điểm thuần = -66.94 … -33.11 đơn vị chi phí trên mức 448.2 … 547.7), nhưng phần lợi "riêng của phân vị" trên hàng nhanh chỉ khoảng 0.72 … 6.68 đơn vị chi phí; ở hàng chậm hầu như không có lợi thế của phân vị. Giải thích theo hiệu chuẩn: trên val, fill rate ở q* thấp hơn nhiều ở hàng chậm (intermittent_slow 0.8270, lumpy_slow 0.6884 so với smooth_fast 0.9388, L=2). Chính sách theo nhóm 9 nhóm (cả mô hình lẫn q): trên test theo nhóm − toàn cục (MLP-NB, q=0.8) L=2 = +0.772 [-1.070; 2.698] → chứa 0 (không phân biệt được); L=3 = +1.878 [0.408; 3.364] → dương (theo nhóm đắt hơn) (trên val L=2: -0.574 [-1.726; 0.504]). Khuyến nghị thực tế: một mô hình và một q (hoặc k) chung; q theo nhóm chỉ để tham khảo. Hạn chế: nhóm lumpy_slow chỉ 34 chuỗi, smooth_medium 2 chuỗi; k* và q* theo nhóm chọn trên val của chính nhóm (thiên vị lạc quan).

**Câu 3 – Học sâu xác suất vs Croston/TSB.** MLP-NB (một MLP, không phải DeepAR/N-HiTS/TFT) tốt hơn Croston/TSB ở cả val và test. Test, pinball_avg H=3: MLP-NB 0.9477, TSB 1.0772, CrostonSBA 1.2631; hiệu theo chuỗi MLP-NB − đối thủ [KTC]:
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: pinball_avg MLP-NB - CrostonSBA (H=3) | -0.3154 [-0.4172; -0.2290] | -0.6473 [-0.9000; -0.4365] | -0.1161 [-0.1578; -0.0825] | -0.0643 [-0.0928; -0.0401] |
| Q3: pinball_avg MLP-NB - LGBM-Q (H=3) | 0.0123 [0.0015; 0.0253] | 0.0320 [0.0059; 0.0646] | 0.0016 [-0.0039; 0.0071] | -0.0048 [-0.0097; 0.0001] |
| Q3: pinball_avg MLP-NB - TSB (H=3) | -0.1295 [-0.1670; -0.0968] | -0.2731 [-0.3659; -0.1934] | -0.0422 [-0.0554; -0.0297] | -0.0228 [-0.0326; -0.0134] |

Chi phí mô phỏng (cùng q newsvendor, tỷ lệ 10, L=2), MLP-NB − đối thủ:
| question | overall | fast | medium | slow |
|---|---|---|---|---|
| Q3: MLP-NB - CrostonSBA (cung q_newsvendor) | -42.95 [-57.26; -30.79] | -94.43 [-129.14; -63.92] | -12.80 [-19.80; -6.96] | -2.59 [-6.88; 1.27] |
| Q3: MLP-NB - LGBM-Q (cung q_newsvendor) | -1.41 [-3.53; 1.04] | 3.95 [-0.92; 9.72] | -4.67 [-6.12; -3.27] | -5.39 [-7.57; -3.34] |
| Q3: MLP-NB - TSB (cung q_newsvendor) | -17.86 [-22.63; -13.69] | -33.17 [-44.23; -23.48] | -8.78 [-11.49; -6.15] | -6.05 [-8.88; -3.42] |

LightGBM phân vị (không phải học sâu) ngang hoặc hơn MLP-NB. Thứ hạng đổi giữa val (MLP-NB 0.9025 < LGBM-Q 0.9109, H=3) và test (LGBM-Q 0.9354 < MLP-NB 0.9477; hiệu MLP-NB − LGBM-Q +0.0123 [0.0015; 0.0253]) và không chỉ do nhiễu seed ở H=3. Với 3 seed (42, 43, 44), pinball_avg (trung bình ± độ lệch chuẩn):
| split | H | n_seed | pinball_avg | wape | mase | epochs | kappa |
|---|---|---|---|---|---|---|---|
| test | 3 | 3 | 0.9460 ± 0.0024 | 0.3515 ± 0.0004 | 0.7843 ± 0.0005 | 17/25/20 | 0.5/0.6/0.6 |
| test | 4 | 3 | 1.1433 ± 0.0058 | 0.3170 ± 0.0006 | 0.8031 ± 0.0032 | 30/16/26 | 0.5/0.6/0.6 |
| val | 3 | 3 | 0.9023 ± 0.0008 | 0.3680 ± 0.0029 | 0.7559 ± 0.0043 | 17/25/20 | 0.5/0.6/0.6 |
| val | 4 | 3 | 1.0897 ± 0.0039 | 0.3315 ± 0.0022 | 0.7748 ± 0.0043 | 30/16/26 | 0.5/0.6/0.6 |

So với LGBM-Q: val H=3: MLP-NB 0.9014–0.9029 vs LGBM-Q 0.9109 → cả 3 seed tốt hơn LGBM-Q; val H=4: MLP-NB 1.0853–1.0929 vs LGBM-Q 1.0922 → lẫn lộn (một số seed tốt hơn, một số kém hơn); test H=3: MLP-NB 0.9432–0.9477 vs LGBM-Q 0.9354 → cả 3 seed kém hơn LGBM-Q; test H=4: MLP-NB 1.1374–1.1489 vs LGBM-Q 1.1344 → cả 3 seed kém hơn LGBM-Q. Tức ở H=3 thứ hạng lật val→test là ổn định theo seed; ở H=4 trên val MLP-NB và LGBM-Q gần như ngang nhau (kết quả lẫn lộn giữa các seed), chỉ trên test LGBM-Q hơn ở cả 3 seed.
**Giải thích về biến ngoại sinh (GĐ6)**: ablation bỏ giá, SNAP và sự kiện khỏi LightGBM-Q (30 → 24 đặc trưng) chỉ làm pinball_avg tăng tối đa 0.84% (val H=3 0.9109 → 0.9185; test H=3 0.9354 → 0.9385) và bản không ngoại sinh vẫn hơn hẳn TSB/CrostonSBA. Lợi thế so với Croston/TSB vì vậy KHÔNG đến chủ yếu từ giá/SNAP/sự kiện mà từ đặc trưng lịch sử đa thang, thứ trong tuần (còn trong bản ablation) và việc huấn luyện trực tiếp phân phối của D_H; chưa tách riêng yếu tố lịch và chưa có đối chứng Croston/TSB + biến ngoại sinh, nên chưa chứng minh được phần đóng góp của "học sâu" so với học máy cây.
**Mã hóa mục tiêu trong mẫu (GĐ7)**: hai cột fut_wd_index và fut_event_index là trung bình doanh số theo thứ/tên sự kiện trên giai đoạn huấn luyện, dùng làm đặc trưng cho chính các dòng huấn luyện. Bỏ riêng hai cột này (giữ 28 đặc trưng còn lại): pinball_avg thay đổi -0.17% … +0.27% so với bản đầy đủ (không đáng kể (|chênh| ≤ 1%, và dấu không nhất quán giữa các giai đoạn/H)):
| split | H | pinball_full | pinball_bo_2_cot_hieu_ung | chenh_tuong_doi |
|---|---|---|---|---|
| test | 3 | 0.9354 | 0.9380 | 0.0027 |
| test | 4 | 1.1344 | 1.1325 | -0.0017 |
| val | 3 | 0.9109 | 0.9115 | 0.0007 |
| val | 4 | 1.0922 | 1.0927 | 0.0005 |


## 7. Hạn chế
- Chỉ 28 ngày val và 28 ngày test, một cửa hàng, một phòng ban; không lặp theo nhiều giai đoạn; KTC bootstrap coi các chuỗi độc lập.
- Siêu tham số (kappa, w, alpha Croston/TSB, q*, k*, chính sách theo nhóm) chọn trên val nên số val lạc quan; test mới là ước lượng không chệch.
- Chi phí giữ/thiếu và quy tắc khởi tạo tồn kho là giả định tương đối; q* chạm cận 0.3 và k* chạm cận 0.5 ở tỷ lệ 1–2.
- Chính sách điểm dùng ceil (+0.5 đơn vị trung bình) trong khi phân vị là số nguyên/làm tròn gần nhất: điểm thuần hơi bảo thủ; điểm an toàn k* hấp thụ phần lớn khác biệt này.
- **Mã hóa mục tiêu trong mẫu**: các chỉ số hiệu ứng fut_wd_index, fut_event_index (hàm `effect_tables` trong `src/ml_models.py`) được tính từ doanh số trung bình theo thứ và theo tên sự kiện trên toàn bộ giai đoạn huấn luyện rồi dùng làm đặc trưng cho chính các dòng huấn luyện đó (in-sample target encoding). Không rò rỉ sang val/test (effect_end đúng) nhưng có thể làm mô hình tin quá mức vào đặc trưng sự kiện hiếm. Thử nghiệm bỏ riêng hai cột: tác động không đáng kể (|chênh| ≤ 1%, và dấu không nhất quán giữa các giai đoạn/H) (pinball_avg -0.17% … +0.27%).
- Mô hình học sâu: MLP-NB và DeepAR-style (GĐ8); không có N-HiTS/TFT/Chronos-2; không có WRMSSE; LightGBM không refit theo `refit_every` trong từng giai đoạn; một seed duy nhất cho các mô hình khác MLP-NB.
- Coverage theo nghĩa đen chưa đạt mục tiêu ở q thấp do tính rời rạc; SeasonalNaive kém hiệu chuẩn ở đuôi; LGBM-Q cho phân vị liên tục nên khoảng hiệu chuẩn rời rạc suy biến.
- Xếp hạng MLP-NB và LightGBM đổi chiều giữa val và test (không do nhiễu seed); MLP-NB H=4 dừng ở trần 30 epoch (cực tiểu NLL ở epoch 46, chênh rất nhỏ).


### Bổ sung GĐ8 – mô hình học sâu thứ hai: DeepAR-style (chi tiết ở `08_deepar.md`)
- DeepAR-style (GRU 2 lớp × 64 nút, context 112 ngày, đầu Negative Binomial, lấy mẫu tự hồi quy 500 quỹ đạo, kappa=0.5) chạy đúng giao thức của MLP-NB (cùng chia dữ liệu, cùng gốc dự báo, cùng tiêu chí chọn trên val, refit train+val khi chạy test, 3 seed). Siêu tham số ở `results/hparams_deepar.json` (đóng băng trước khi chạy test; `hparams.json` và số liệu của 7 mô hình cũ không đổi, kiểm bằng sha256).
- pinball_avg (H=3 / H=4): val DeepAR 0.9147 / 1.1016, MLP-NB 0.9025 / 1.0929, LGBM-Q 0.9109 / 1.0922, TSB 1.0382 / 1.2372; test DeepAR 1.0045 / 1.2384, MLP-NB 0.9477 / 1.1489, LGBM-Q 0.9354 / 1.1344, TSB 1.0772 / 1.2991.
- **Seed của các số "chính"**: mọi con số của DeepAR và MLP-NB trong các bảng so sánh là SEED 42 (một seed; không phải trung bình/ensemble), chốt trước khi có kết quả (seed đầu của `mlp_seeds` trong config; MLP-NB dùng seed 42 từ GĐ2). Trung bình ± sd 3 seed, cùng cách tính, pinball_avg (H=3 / H=4): val DeepAR 0.9172 ± 0.0028 / 1.1061 ± 0.0042, MLP-NB 0.9023 ± 0.0008 / 1.0897 ± 0.0039; test DeepAR 1.0001 ± 0.0051 / 1.2313 ± 0.0085, MLP-NB 0.9460 ± 0.0024 / 1.1433 ± 0.0058. Kết luận không đổi: MLP-NB tốt hơn DeepAR ở cả bốn ô theo trung bình seed (xem `08_deepar.md` mục 5.1).
- Mô hình học sâu đề xuất theo val vẫn là **MLP-NB** (DeepAR kém hơn ở cả hai H trên val). Chi phí mô phỏng (tỷ lệ 10, q* riêng chọn trên val, test, L=2): DeepAR − MLP-NB 6.45 [2.29; 11.54], DeepAR − LGBM-Q 6.45 [2.02; 11.89], DeepAR − TSB -13.14 [-17.59; -8.99] (âm = DeepAR rẻ hơn); L=3: − MLP-NB 9.12 [4.59; 14.32], − LGBM-Q 8.35 [3.91; 13.71], − TSB -13.25 [-17.62; -9.20].
- Kết luận câu 3 được củng cố: cả hai mô hình học sâu hơn TSB, nhưng DeepAR không hơn MLP-NB/LightGBM (ngang trên val, kém hơn có ý nghĩa trên test). Câu 1 và 2 không đổi. Hạn chế riêng: huấn luyện CPU ngân sách nhỏ (≤12 epoch × 20 000 cửa sổ), cấu hình tốt nhất ở góc lưới, hiệu chuẩn kém bền hơn từ val sang test.

## 8. Tài liệu tham khảo
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). The M5 accuracy competition: results, findings, and conclusions. International Journal of Forecasting 38(4). Dữ liệu: Kaggle "M5 Forecasting – Accuracy" (https://www.kaggle.com/c/m5-forecasting-accuracy).
- Croston, J. D. (1972). Forecasting and stock control for intermittent demands. Operational Research Quarterly 23(3). Syntetos, A. A., Boylan, J. E. (2005). The accuracy of intermittent demand estimates. International Journal of Forecasting 21(2). Teunter, R. H., Syntetos, A. A., Babai, M. Z. (2011). Intermittent demand: linking forecasting to inventory obsolescence. European Journal of Operational Research 214(3) (TSB).
- Nixtla StatsForecast (https://github.com/Nixtla/statsforecast) – tham chiếu cho Croston/TSB/WindowAverage/SeasonalNaive (mã dự án tự cài đặt phiên bản vector hóa). Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS.
- Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting 36(3) – chỉ làm tham chiếu cho loss NB; GluonTS/neuralforecast không được dùng.
- Hyndman, R. J., Koehler, A. B. (2006). Another look at measures of forecast accuracy (MASE). Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation (pinball/quantile loss).
- Guo, C., Berkhahn, F. (2016). Entity embeddings of categorical variables – chỉ liên quan nếu mở rộng MLP-NB bằng embedding mặt hàng; chưa dùng trong bản này.
