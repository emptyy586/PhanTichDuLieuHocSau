# Báo cáo GĐ8 – DeepAR-style (GRU + Negative Binomial, lấy mẫu tự hồi quy)

> Các mục 0–2 được commit TRƯỚC khi chạy test (commit `GD8: DeepAR-style … đóng băng hparams_deepar.json, cam kết …`); các mục 3 trở đi được dựng sau khi có kết quả test.

## 0. Cam kết trước khi chạy test (ghi trước, không sửa về sau)
1. Báo cáo **cả DeepAR lẫn MLP-NB** (và 5 mô hình còn lại) trên val và test **bất kể ai thắng**.
2. "Mô hình học sâu đề xuất" được chọn **theo pinball trung bình trên VAL** (DeepAR seed 42 vs MLP-NB seed 42, cùng tiêu chí, cùng H=3,4), **không** chọn theo test.
3. Siêu tham số của DeepAR (hidden, số lớp, context, kappa, số epoch từng seed) được chốt trên val và lưu ở `results/hparams_deepar.json` trước khi chạy test; **không tinh chỉnh vòng hai dựa vào test**. Test chạy đúng một lần cho mỗi seed.
4. `results/hparams.json` không được sửa (sha256 phải là `8BC7D7507BF0167E83B1D31F831FB8EDB5EDF60C23A742C67B32B0F86366765B`); số liệu của 7 mô hình hiện có không đổi (kiểm bằng sha256 các file cũ trong `results/gd8_baseline_hashes.txt` và test `test_old_result_files_unchanged_since_baseline`).
5. Không hiệu chuẩn lại DeepAR sau khi đã thấy test (kappa chọn trên val).

## 1. Giao thức (giống MLP-NB)
- Cùng `Panel`, cùng chia train / val / test; cùng gốc dự báo `origins_for_split` (26 gốc với H=3, 25 gốc với H=4 trong 28 ngày); H = lead time + chu kỳ đặt (R=1): L=2 → H=3, L=3 → H=4; cùng lưới phân vị chính {0.5,…,0.99} và lưới phụ {0.3, 0.4}.
- Val: huấn luyện trên train (nhãn đến 2016-03-27), chọn epoch bằng tập kiểm tra nội bộ (56 gốc cuối của train, purge H_max=4 ngày); siêu tham số chọn trên VAL bằng pinball trung bình trên lưới phân vị chính và cả hai H (cùng hàm `avg_pinball` với MLP-NB); 3 seed (42, 43, 44).
- Test: huấn luyện lại trên train+val (nhãn đến 2016-04-24) với đúng số epoch/siêu tham số/kappa đã chốt; dự báo các gốc test.
- Một mô hình DeepAR duy nhất cho cả H=3 và H=4: huấn luyện với horizon 4 ngày, lấy mẫu H bước tương ứng (khác MLP-NB, vốn huấn luyện một mạng cho mỗi H).

## 2. Quyết định đã chốt trên VAL (ghi TRƯỚC khi chạy test)
- Cấu hình DeepAR chọn trên val (pinball_avg val, seed 42, n_samples=300, kappa=1; 8 cấu hình, `results/tuning_log_deepar.csv`): **hidden=64, layers=2, context=112** (pinball_avg 1.0124; cấu hình tốt thứ hai 1.0275). Kappa chọn trên val (lưới 0.25, 0.35, 0.5, 0.75, 1.0, 1.25; `results/tuning_kappa_deepar.csv`): **0.5** (1.0089). Epoch chọn bằng holdout nội bộ: seed 42 → 10, seed 43 → 12, seed 44 → 12 (trần 12).
- `results/hparams_deepar.json` đã được đóng băng: sha256 = `9b2036d39e56bb8d4211a93a251921229549480084fd48445fa9a1c29fdecd34`. `results/hparams.json` không đổi (`8bc7d7507bf0167e83b1d31f831fb8edb5edf60c23a742c67b32b0f86366765b`).
- DeepAR val (seed 42, n_samples=500, kappa=0.5): pinball_avg H=3 **0.9147**, H=4 **1.1016**, trung bình hai H **1.0081**. MLP-NB val (seed 42): H=3 0.9025, H=4 1.0929, tiêu chí tuning `avg_pinball` 0.9977. **Theo tiêu chí val, mô hình học sâu đề xuất là MLP-NB** (DeepAR kém hơn ở cả hai H). Quyết định này đã chốt; test không thay đổi nó.
- Cấu hình tốt nhất nằm ở góc lưới (hidden và context lớn nhất, 2 lớp) nên có thể còn tốt hơn ngoài lưới; không mở rộng vì ngân sách CPU và vì sẽ là tinh chỉnh thêm sau khi đã thấy val (không dựa test).

## 3. Mô tả mô hình (dùng được cho mục 1.3 và 3.1 của báo cáo môn học)
### 3.1 Kiến trúc và hàm mất mát
DeepAR-style: một mạng hồi quy GRU dùng chung cho mọi chuỗi, mô hình hóa phân phối có điều kiện của nhu cầu ngày d bằng Negative Binomial. Với chuỗi i, ngày d, đầu vào của GRU là vectơ x_{i,d} = [ y_{i,d-1}/s_i ; e_i ; c_{i,d} ] trong đó s_i = 1 + mean(y_i trong context đến gốc dự báo) là hệ số co giãn (chỉ dùng doanh số đã quan sát), e_i ∈ R^8 là embedding của chuỗi, và c_{i,d} ∈ R^15 là các covariates ĐÃ BIẾT cả trong tương lai: thứ trong tuần (one-hot 7), tháng và ngày trong tháng (sin/cos), cờ SNAP CA, cờ sự kiện, giá bán tương đối so với giá trung vị của chuỗi trong giai đoạn train, cờ giá thiếu (chuỗi chưa niêm yết). Trạng thái ẩn h_{i,d} = GRU(x_{i,d}, h_{i,d-1}) (2 lớp, 64 nút, dropout 0.1). Đầu ra hai số thô (a, b):

  μ_{i,d} = s_i · softplus(a) ,  α_{i,d} = softplus(b) + 10⁻³ ,  y_{i,d} ~ NB(μ_{i,d}, α_{i,d}) với Var = μ + α μ².

Hàm mất mát là âm log-hợp lý (r = 1/α):

  NLL = − [ lnΓ(y+r) − lnΓ(r) − lnΓ(y+1) + r ln( r/(r+μ) ) + y ln( μ/(r+μ) ) ] ,

được kiểm bằng `scipy.stats.nbinom.logpmf` trong `tests/test_gd8.py`.
### 3.2 Huấn luyện
Mỗi bước huấn luyện lấy một batch (256) cửa sổ ngẫu nhiên (chuỗi i, gốc t) trong 730 ngày gốc gần nhất của giai đoạn huấn luyện (chuỗi phải đã có lần bán tại gốc, nhãn đến gốc+4 nằm trong giai đoạn huấn luyện), gồm C ngày context + 4 ngày dự báo; teacher forcing, NLL trung bình trên mọi bước của cửa sổ; Adam (lr 10⁻³), cắt gradient 1.0; mỗi epoch 20 000 cửa sổ, tối đa 12 epoch. Số epoch chọn bằng NLL trên tập kiểm tra nội bộ (56 gốc cuối của train, purge 4 ngày, 15 000 cửa sổ cố định; chỉ tính trên các bước dự báo) — cùng cách MLP-NB làm. Khi chạy test: huấn luyện lại trên train+val với đúng số epoch đã chọn.
### 3.3 Lấy mẫu tự hồi quy
Với mỗi cặp (chuỗi i, gốc t): (1) chạy encoder (teacher forcing) trên context t−C+1…t để lấy trạng thái h_t; (2) bước k=1: đầu vào (y_t quan sát, c_{t+1}); suy ra (μ, α·κ) và lấy n mẫu ŷ_{t+1}^{(j)} ~ NB (qua Gamma–Poisson: λ ~ Gamma(1/α, 1/(αμ)), y ~ Poisson(λ)); nhân bản trạng thái n lần; (3) bước k=2…H: đầu vào (ŷ_{t+k−1}^{(j)}/s, c_{t+k}), lấy mẫu ŷ_{t+k}^{(j)}; (4) D_H^{(j)} = Σ_{k=1..H} ŷ_{t+k}^{(j)}. n = 500 mẫu (tuning dùng 300), seed cố định. Hệ số hiệu chuẩn κ nhân với α ở mọi bước (chọn trên val, cùng vai trò κ của MLP-NB).
### 3.4 Từ mẫu tới phân vị và chính sách
Phân vị kinh nghiệm kiểu "inverted CDF": Q̂_q(D_H) = mẫu thứ ⌈q·n⌉ sau khi sắp tăng (số nguyên, không âm), cho lưới chính {0.5,…,0.99} và lưới phụ {0.3, 0.4}; "mean" dùng cho chính sách điểm = trung bình mẫu của D_H. Chính sách order-up-to dùng S = Q̂_q (qua `quantile_to_levels`: phân vị đã là số nguyên nên không làm tròn thêm), giống các mô hình NB. Chuỗi chưa có lần bán nào tại gốc → phân vị 0 (cùng quy tắc `age_days<0` của các mô hình khác).
### 3.5 Chống rò rỉ
Đầu vào chỉ gồm doanh số ≤ gốc t (kiểm bằng test: đổi doanh số SAU gốc thì mẫu dự báo không đổi, `test_no_leak_future_values_after_origin_do_not_change_forecast`) và covariates đã biết (lịch, SNAP, sự kiện, giá). Không dùng hai cột mã hóa mục tiêu in-sample (fut_wd_index, fut_event_index). Giá trung vị chỉ từ ngày ≤ hết train (cả lúc chạy test). Lưu ý chung cho MỌI mô hình trong dự án (không riêng DeepAR): quy tắc "chưa có lần bán nào tại gốc → dự báo 0" dùng chỉ số ngày bán đầu tiên tính trên toàn bộ dữ liệu (`age_days`), tức biết trước một chuỗi sẽ ra mắt sau gốc; DeepAR dùng cùng quy tắc để so sánh công bằng.

## 4. Cấu hình, tham số, thời gian
Cấu hình đã chốt (từ `results/hparams_deepar.json`, sha256 `9b2036d39e56bb8d4211a93a251921229549480084fd48445fa9a1c29fdecd34`; bảng cũng ở `results/tables/deepar_hparams.csv`):
| hidden | layers | context | dropout | lr | batch_size | windows_per_epoch | max_epochs | kappa | n_samples | n_params | epochs_seed42/43/44 | train_seconds_seed42/43/44 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 2 | 112 | 0.1000 | 0.0010 | 256 | 20000 | 12 | 0.5000 | 500 | 48954 | 10/12/12 | 775/564/560 |

Số tham số: 48954. Môi trường: CPU 12 luồng, PyTorch 2.15.0+cpu (CPU), numpy 2.5.4, pandas 2.3.3, scipy 1.18.1. Tiến trình chạy song song (3–4 luồng mỗi tiến trình; thử nghiệm cho thấy 12 luồng cho MỘT tiến trình chậm hơn 3–6 luồng). Thời gian huấn luyện trên val (đồng hồ, tiến trình chạy song song): seed 42: 775 s, seed 43: 564 s, seed 44: 560 s; refit trên train+val khi chạy test: seed 42: 513 s, seed 43: 675 s, seed 44: 667 s; lấy mẫu 500 quỹ đạo cho cả hai H: 162–198 s mỗi seed.
Tuning 8 cấu hình trên val (seed 42, n_samples=300, kappa=1; sắp theo pinball_avg_val = trung bình pinball mọi q lưới chính và cả H=3,4):
| cfg_id | hidden | layers | context | best_epoch | n_params | train_seconds | sample_seconds | holdout_nll | pinball_avg_val |
|---|---|---|---|---|---|---|---|---|---|
| 7.0000 | 64.0000 | 2.0000 | 112.0000 | 10.0000 | 48954.0000 | 775.2000 | 117.8000 | 1.5143 | 1.0124 |
| 3.0000 | 32.0000 | 2.0000 | 112.0000 | 11.0000 | 18554.0000 | 456.8000 | 78.9000 | 1.5201 | 1.0275 |
| 5.0000 | 64.0000 | 1.0000 | 112.0000 | 12.0000 | 23994.0000 | 345.3000 | 90.9000 | 1.5227 | 1.0285 |
| 2.0000 | 32.0000 | 2.0000 | 56.0000 | 12.0000 | 18554.0000 | 245.9000 | 69.2000 | 1.5202 | 1.0300 |
| 6.0000 | 64.0000 | 2.0000 | 56.0000 | 11.0000 | 48954.0000 | 345.3000 | 130.6000 | 1.5164 | 1.0312 |
| 4.0000 | 64.0000 | 1.0000 | 56.0000 | 12.0000 | 23994.0000 | 192.3000 | 68.6000 | 1.5232 | 1.0388 |
| 1.0000 | 32.0000 | 1.0000 | 112.0000 | 12.0000 | 12218.0000 | 240.1000 | 50.8000 | 1.5265 | 1.0449 |
| 0.0000 | 32.0000 | 1.0000 | 56.0000 | 12.0000 | 12218.0000 | 125.1000 | 28.8000 | 1.5288 | 1.0476 |

Chọn kappa (cấu hình tốt nhất, n_samples=300):
| kappa | pinball_avg_val |
|---|---|
| 0.2500 | 1.0163 |
| 0.3500 | 1.0139 |
| 0.5000 | 1.0089 |
| 0.7500 | 1.0099 |
| 1.0000 | 1.0124 |
| 1.2500 | 1.0165 |

Cấu hình tốt nhất là cấu hình LỚN NHẤT trong lưới (hidden 64, 2 lớp, context 112) và còn đang giảm theo kích thước, nên có thể còn tốt hơn ngoài lưới (không mở rộng vì ngân sách CPU).

## 5. Kết quả dự báo (val và test, 8 mô hình)
**Con số "chính" trong mọi bảng so sánh của DeepAR và MLP-NB là SEED 42 (một seed, không phải trung bình hay ensemble)**, cho CẢ HAI mô hình. (a)+(b) Seed 42 được chốt TRƯỚC khi thấy kết quả: đây là phần tử đầu của `mlp_seeds` trong `configs/config.yaml` (cố định từ GĐ6, trước DeepAR), MLP-NB dùng seed 42 làm mô hình chính từ GĐ2 (`hparams.json`), và mã DeepAR dùng `seeds[0]` ngay từ lần viết đầu (tuning và mô hình chính đều seed 42); không chọn seed theo val hay test. Seed 43, 44 chỉ để đo độ ổn định. Khác biệt cần nhớ: kappa của MLP-NB chọn riêng cho từng seed (seed 42: 0.5; seed 43, 44: 0.6), kappa của DeepAR chọn một lần theo seed 42 (0.5) rồi dùng cho cả 3 seed; epoch chọn riêng từng seed ở cả hai mô hình. Dưới mỗi bảng có thêm dòng TRUNG BÌNH 3 SEED ± độ lệch chuẩn (ddof=1) cho cả DeepAR và MLP-NB, tính cùng cách (mean ± sd của chỉ số tổng thể trên 3 seed). pinball_avg = trung bình pinball trên 9 mức q (lưới chính); WAPE trên trung bình cửa sổ (DeepAR: trung bình mẫu); MASE scale chỉ train. Các số của 7 mô hình cũ là số cũ, không đổi.
H=3:
| model | pinball_avg_val | pinball_avg_test | wape_val | wape_test | mase_val | mase_test |
|---|---|---|---|---|---|---|
| WindowAverage | 1.0757 | 1.1316 | 0.4129 | 0.3987 | 0.8125 | 0.8474 |
| SeasonalNaive | 1.4002 | 1.4112 | 0.4741 | 0.4399 | 0.9645 | 0.9882 |
| CrostonSBA | 1.2526 | 1.2631 | 0.4948 | 0.4484 | 0.9502 | 0.9174 |
| TSB | 1.0382 | 1.0772 | 0.4058 | 0.3898 | 0.8009 | 0.8287 |
| LGBM-Q | 0.9109 | 0.9354 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| LGBM-PoisNB | 0.9300 | 0.9490 | 0.3697 | 0.3508 | 0.7672 | 0.7870 |
| MLP-NB | 0.9025 | 0.9477 | 0.3701 | 0.3512 | 0.7597 | 0.7839 |
| DeepAR | 0.9147 | 1.0045 | 0.3700 | 0.3628 | 0.7737 | 0.8055 |
| DeepAR – TB 3 seed ± sd | 0.9172 ± 0.0028 | 1.0001 ± 0.0051 | 0.3692 ± 0.0007 | 0.3609 ± 0.0016 | 0.7736 ± 0.0009 | 0.8023 ± 0.0030 |
| MLP-NB – TB 3 seed ± sd | 0.9023 ± 0.0008 | 0.9460 ± 0.0024 | 0.3680 ± 0.0029 | 0.3515 ± 0.0004 | 0.7559 ± 0.0043 | 0.7843 ± 0.0005 |

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
| DeepAR | 1.1016 | 1.2384 | 0.3358 | 0.3306 | 0.7995 | 0.8298 |
| DeepAR – TB 3 seed ± sd | 1.1061 ± 0.0042 | 1.2313 ± 0.0085 | 0.3357 ± 0.0009 | 0.3285 ± 0.0019 | 0.8006 ± 0.0011 | 0.8256 ± 0.0039 |
| MLP-NB – TB 3 seed ± sd | 1.0897 ± 0.0039 | 1.1433 ± 0.0058 | 0.3315 ± 0.0022 | 0.3170 ± 0.0006 | 0.7748 ± 0.0043 | 0.8031 ± 0.0032 |

Ba seed của DeepAR và MLP-NB (trung bình ± độ lệch chuẩn, n=3). DeepAR:
| split | H | n_seed | pinball_avg | wape | mase | epochs | kappa |
|---|---|---|---|---|---|---|---|
| test | 3 | 3 | 1.0001 ± 0.0051 | 0.3609 ± 0.0016 | 0.8023 ± 0.0030 | 10/12/12 | 0.5/0.5/0.5 |
| test | 4 | 3 | 1.2313 ± 0.0085 | 0.3285 ± 0.0019 | 0.8256 ± 0.0039 | 10/12/12 | 0.5/0.5/0.5 |
| val | 3 | 3 | 0.9172 ± 0.0028 | 0.3692 ± 0.0007 | 0.7736 ± 0.0009 | 10/12/12 | 0.5/0.5/0.5 |
| val | 4 | 3 | 1.1061 ± 0.0042 | 0.3357 ± 0.0009 | 0.8006 ± 0.0011 | 10/12/12 | 0.5/0.5/0.5 |

MLP-NB (để so sánh):
| split | H | n_seed | pinball_avg | wape | mase | epochs | kappa |
|---|---|---|---|---|---|---|---|
| test | 3 | 3 | 0.9460 ± 0.0024 | 0.3515 ± 0.0004 | 0.7843 ± 0.0005 | 17/25/20 | 0.5/0.6/0.6 |
| test | 4 | 3 | 1.1433 ± 0.0058 | 0.3170 ± 0.0006 | 0.8031 ± 0.0032 | 30/16/26 | 0.5/0.6/0.6 |
| val | 3 | 3 | 0.9023 ± 0.0008 | 0.3680 ± 0.0029 | 0.7559 ± 0.0043 | 17/25/20 | 0.5/0.6/0.6 |
| val | 4 | 3 | 1.0897 ± 0.0039 | 0.3315 ± 0.0022 | 0.7748 ± 0.0043 | 30/16/26 | 0.5/0.6/0.6 |

Chi tiết từng seed của DeepAR:
| seed | split | H | kappa | epochs | pinball_avg | wape | mase | train_seconds | sample_seconds |
|---|---|---|---|---|---|---|---|---|---|
| 42 | val | 3 | 0.5000 | 10 | 0.9147 | 0.3700 | 0.7737 | 775.2000 | 161.7000 |
| 42 | val | 4 | 0.5000 | 10 | 1.1016 | 0.3358 | 0.7995 | 775.2000 | 162.4000 |
| 43 | val | 3 | 0.5000 | 12 | 0.9202 | 0.3686 | 0.7745 | 563.6000 | 182.1000 |
| 43 | val | 4 | 0.5000 | 12 | 1.1098 | 0.3348 | 0.8017 | 563.6000 | 182.2000 |
| 44 | val | 3 | 0.5000 | 12 | 0.9168 | 0.3691 | 0.7727 | 559.5000 | 183.7000 |
| 44 | val | 4 | 0.5000 | 12 | 1.1069 | 0.3366 | 0.8005 | 559.5000 | 183.9000 |
| 42 | test | 3 | 0.5000 | 10 | 1.0045 | 0.3628 | 0.8055 | 513.2000 | 187.0000 |
| 42 | test | 4 | 0.5000 | 10 | 1.2384 | 0.3306 | 0.8298 | 513.2000 | 187.2000 |
| 43 | test | 3 | 0.5000 | 12 | 1.0012 | 0.3602 | 0.7995 | 675.4000 | 194.2000 |
| 43 | test | 4 | 0.5000 | 12 | 1.2336 | 0.3281 | 0.8220 | 675.4000 | 194.4000 |
| 44 | test | 3 | 0.5000 | 12 | 0.9946 | 0.3598 | 0.8019 | 666.6000 | 198.0000 |
| 44 | test | 4 | 0.5000 | 12 | 1.2218 | 0.3269 | 0.8249 | 666.6000 | 198.2000 |


### 5.1 Làm rõ seed và bảng trung bình 3 seed (GĐ8b)
**Con số "chính" trong mọi bảng so sánh của DeepAR và MLP-NB là SEED 42 (một seed, không phải trung bình hay ensemble)**, cho CẢ HAI mô hình. (b) Seed 42 được chốt TRƯỚC khi thấy kết quả: đây là phần tử đầu của `mlp_seeds` trong `configs/config.yaml` (cố định từ GĐ6, trước DeepAR), MLP-NB dùng seed 42 làm mô hình chính từ GĐ2 (`hparams.json`), và mã DeepAR dùng `seeds[0]` ngay từ lần viết đầu (tuning và mô hình chính đều seed 42); không chọn seed theo val hay test. Seed 43, 44 chỉ để đo độ ổn định. Khác biệt cần nhớ: kappa của MLP-NB chọn riêng cho từng seed (seed 42: 0.5; seed 43, 44: 0.6), kappa của DeepAR chọn một lần theo seed 42 (0.5) rồi dùng cho cả 3 seed; epoch chọn riêng từng seed ở cả hai mô hình.
Trung bình ± sd 3 seed của pinball_avg (cùng cách tính cho hai mô hình; số seed 42 và các số khác trong báo cáo KHÔNG đổi):

| Chỉ số | DeepAR | MLP-NB | DeepAR − MLP-NB (trung bình seed) |
|---|---|---|---|
| pinball_avg val H=3 | 0.9172 ± 0.0028 (từng seed: 0.9147, 0.9202, 0.9168) | 0.9023 ± 0.0008 (từng seed: 0.9025, 0.9014, 0.9029) | +0.0149 |
| pinball_avg val H=4 | 1.1061 ± 0.0042 (từng seed: 1.1016, 1.1098, 1.1069) | 1.0897 ± 0.0039 (từng seed: 1.0929, 1.0909, 1.0853) | +0.0164 |
| pinball_avg test H=3 | 1.0001 ± 0.0051 (từng seed: 1.0045, 1.0012, 0.9946) | 0.9460 ± 0.0024 (từng seed: 0.9477, 0.9471, 0.9432) | +0.0541 |
| pinball_avg test H=4 | 1.2313 ± 0.0085 (từng seed: 1.2384, 1.2336, 1.2218) | 1.1433 ± 0.0058 (từng seed: 1.1489, 1.1435, 1.1374) | +0.0880 |

Kết luận so sánh theo trung bình 3 seed: **không đổi chiều**. MLP-NB có pinball_avg thấp hơn DeepAR ở cả bốn ô (val/test × H=3,4); seed tốt nhất của DeepAR vẫn kém seed kém nhất của MLP-NB ở cả 4 ô (dải seed không chồng lấn). Chênh lệch trung bình seed khoảng 0.015 (val H=3) và 0.054 (test H=3), lớn hơn nhiều sd giữa các seed. Điểm cần nói rõ: ở val, KTC bootstrap THEO CHUỖI của seed 42 chứa 0 (hai mô hình không phân biệt được khi lấy mẫu chuỗi), nhưng theo trục seed DeepAR kém đều đặn; hai loại nhiễu này khác nhau và kết luận "MLP-NB ≥ DeepAR" nhất quán với cả hai (độ chắc chắn trên val thấp hơn trên test).

**Theo nhóm tốc độ** (pinball_avg, WAPE, MASE; H=3; đánh giá dự báo cửa sổ, không phụ thuộc tỷ lệ chi phí/L/q). val:
| model | pinball_avg_fast | pinball_avg_medium | pinball_avg_slow | wape_fast | wape_medium | wape_slow | mase_fast | mase_medium | mase_slow |
|---|---|---|---|---|---|---|---|---|---|
| DeepAR | 1.5005 | 0.6222 | 0.3630 | 0.3118 | 0.5229 | 0.7174 | 0.6510 | 0.8138 | 0.9278 |
| MLP-NB | 1.4806 | 0.6143 | 0.3574 | 0.3150 | 0.5169 | 0.6888 | 0.6495 | 0.8033 | 0.8844 |
| LGBM-Q | 1.4905 | 0.6237 | 0.3609 | 0.3123 | 0.5220 | 0.7039 | 0.6479 | 0.8114 | 0.9074 |
| TSB | 1.7432 | 0.6811 | 0.3839 | 0.3554 | 0.5404 | 0.6964 | 0.7150 | 0.8387 | 0.8909 |

test:
| model | pinball_avg_fast | pinball_avg_medium | pinball_avg_slow | wape_fast | wape_medium | wape_slow | mase_fast | mase_medium | mase_slow |
|---|---|---|---|---|---|---|---|---|---|
| DeepAR | 1.7017 | 0.6196 | 0.4150 | 0.3050 | 0.5183 | 0.7687 | 0.6994 | 0.8131 | 0.9882 |
| MLP-NB | 1.5679 | 0.6154 | 0.4050 | 0.2931 | 0.5141 | 0.7320 | 0.6816 | 0.8059 | 0.9333 |
| LGBM-Q | 1.5358 | 0.6139 | 0.4098 | 0.2916 | 0.5148 | 0.7447 | 0.6773 | 0.8066 | 0.9546 |
| TSB | 1.8410 | 0.6576 | 0.4278 | 0.3372 | 0.5336 | 0.7494 | 0.7524 | 0.8352 | 0.9581 |

(H=4: `results/deepar_forecast_by_speed.csv`.)
**Hiệu chuẩn của DeepAR (chưa hiệu chuẩn lại sau khi thấy test)**: số q (trên 7 mức q ≤ 0.95) nằm trong khoảng [P(y<q̂), P(y≤q̂)] chặt (in_bracket), nới ±0.05 (in_bracket_tol) và thỏa |P(y≤q̂)−q|≤0.05 (abs_dev_ok):
| split | H | model | n_q | in_bracket | in_bracket_tol | abs_dev_ok |
|---|---|---|---|---|---|---|
| test | 3 | DeepAR | 7 | 6 | 7 | 4 |
| test | 4 | DeepAR | 7 | 5 | 7 | 5 |
| val | 3 | DeepAR | 7 | 7 | 7 | 4 |
| val | 4 | DeepAR | 7 | 7 | 7 | 4 |

P(y<q̂) / P(y≤q̂) theo q (cột = q = 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
| index | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| DeepAR val H=3 | 0.404 / 0.625 | 0.496 / 0.702 | 0.596 / 0.773 | 0.708 / 0.844 | 0.769 / 0.880 | 0.845 / 0.914 | 0.923 / 0.951 |

| index | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| DeepAR val H=4 | 0.421 / 0.614 | 0.514 / 0.693 | 0.614 / 0.766 | 0.725 / 0.839 | 0.788 / 0.874 | 0.862 / 0.912 | 0.927 / 0.950 |

| index | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| DeepAR test H=3 | 0.411 / 0.599 | 0.508 / 0.676 | 0.612 / 0.754 | 0.725 / 0.829 | 0.787 / 0.866 | 0.848 / 0.904 | 0.916 / 0.943 |

| index | 0.5 | 0.6 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| DeepAR test H=4 | 0.426 / 0.586 | 0.522 / 0.666 | 0.627 / 0.745 | 0.738 / 0.821 | 0.793 / 0.859 | 0.855 / 0.898 | 0.914 / 0.940 |


## 6. Mô phỏng tồn kho (cost_ratio = 10; chi phí tương đối; q*, fill, chi phí là TB/chuỗi trong 28 ngày; q* chọn trên val trên lưới 0.3–0.99; test dùng q*(val))
L=2 (H=3):
| model | q_star_val | fill_val | cost_val | fill_test | cost_test | inv_test | hang_val | hang_test |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.8000 | 0.8740 | 271.4933 | 0.8683 | 283.2892 | 6.3451 | 6 | 6 |
| SeasonalNaive | 0.7000 | 0.8347 | 305.1470 | 0.8377 | 308.2199 | 6.3589 | 8 | 8 |
| CrostonSBA | 0.8000 | 0.8529 | 301.4739 | 0.8522 | 304.2539 | 6.6326 | 7 | 7 |
| TSB | 0.8000 | 0.8827 | 269.6221 | 0.8835 | 278.0923 | 6.5956 | 5 | 5 |
| LGBM-Q | 0.8000 | 0.9069 | 254.4751 | 0.9028 | 258.5018 | 6.4488 | 2 | 1 |
| LGBM-PoisNB | 0.8000 | 0.9014 | 256.5772 | 0.9035 | 260.0826 | 6.5257 | 4 | 3 |
| MLP-NB | 0.8000 | 0.9062 | 252.4508 | 0.9034 | 258.5030 | 6.4662 | 1 | 2 |
| DeepAR | 0.8000 | 0.9010 | 254.5954 | 0.8809 | 264.9538 | 6.0526 | 3 | 4 |

L=3 (H=4):
| model | q_star_val | fill_val | cost_val | fill_test | cost_test | inv_test | hang_val | hang_test |
|---|---|---|---|---|---|---|---|---|
| WindowAverage | 0.7000 | 0.8408 | 289.8676 | 0.8294 | 306.9745 | 6.0766 | 6 | 6 |
| SeasonalNaive | 0.6000 | 0.8061 | 313.9903 | 0.8052 | 320.0680 | 5.8508 | 7 | 7 |
| CrostonSBA | 0.7000 | 0.8091 | 326.5699 | 0.8035 | 333.0826 | 6.2683 | 8 | 8 |
| TSB | 0.7000 | 0.8494 | 288.3499 | 0.8451 | 300.9453 | 6.3131 | 5 | 5 |
| LGBM-Q | 0.7000 | 0.8663 | 273.0911 | 0.8642 | 279.3499 | 6.0881 | 3 | 2 |
| LGBM-PoisNB | 0.7000 | 0.8637 | 274.4885 | 0.8653 | 279.7606 | 6.1340 | 4 | 3 |
| MLP-NB | 0.7000 | 0.8695 | 270.0267 | 0.8633 | 278.5772 | 6.0336 | 1 | 1 |
| DeepAR | 0.7000 | 0.8701 | 272.7813 | 0.8436 | 287.6950 | 5.7969 | 2 | 4 |

Hạng (1 = chi phí thấp nhất trong 8 mô hình): DeepAR L=2 hạng val 3 / test 4; L=3 hạng val 2 / test 4. Hình cập nhật (lưới chính, như bản cũ, thêm DeepAR): `results/figures/policy_total_cost_by_q.png`, `policy_tradeoff_fill_vs_inventory.png`. Bảng q* mọi tỷ lệ: `results/q_star_with_deepar.csv`. Mô phỏng chỉ cho DeepAR: `results/inventory_sim_deepar_{val,test}[_ext].csv` (các file mô phỏng của 7 mô hình cũ không bị ghi lại; kiểm bằng hash và hàm `old_rows_identical`). `safety_point` (điểm có hệ số an toàn) KHÔNG chạy cho DeepAR vì câu hỏi 1 và 2 là câu hỏi về phương pháp và không đổi theo mô hình bổ sung.

## 7. So sánh có khoảng tin cậy (bootstrap theo chuỗi, 2000 lần, seed 42, ghép cặp)
**Pinball_avg (DeepAR − đối thủ; âm = DeepAR tốt hơn; trung bình theo chuỗi trên các gốc và mọi q lưới chính):**
| so_sanh | H | val | test |
|---|---|---|---|
| DeepAR - LGBM-Q | 3 | 0.0038 [-0.0070; 0.0151] | 0.0691 [0.0322; 0.1149] |
| DeepAR - LGBM-Q | 4 | 0.0094 [-0.0068; 0.0266] | 0.1040 [0.0490; 0.1722] |
| DeepAR - MLP-NB | 3 | 0.0121 [-0.0039; 0.0257] | 0.0568 [0.0241; 0.0999] |
| DeepAR - MLP-NB | 4 | 0.0087 [-0.0163; 0.0292] | 0.0895 [0.0387; 0.1533] |
| DeepAR - TSB | 3 | -0.1236 [-0.1502; -0.0988] | -0.0727 [-0.1025; -0.0446] |
| DeepAR - TSB | 4 | -0.1357 [-0.1668; -0.1062] | -0.0607 [-0.1008; -0.0210] |

Theo nhóm tốc độ (H=3):
| so_sanh | group | val | test |
|---|---|---|---|
| DeepAR - LGBM-Q | fast | 0.0100 [-0.0148; 0.0369] | 0.1659 [0.0716; 0.2842] |
| DeepAR - LGBM-Q | medium | -0.0015 [-0.0089; 0.0066] | 0.0057 [-0.0016; 0.0133] |
| DeepAR - LGBM-Q | slow | 0.0020 [-0.0066; 0.0114] | 0.0052 [-0.0028; 0.0139] |
| DeepAR - MLP-NB | fast | 0.0199 [-0.0192; 0.0535] | 0.1339 [0.0513; 0.2437] |
| DeepAR - MLP-NB | medium | 0.0079 [0.0004; 0.0163] | 0.0042 [-0.0029; 0.0116] |
| DeepAR - MLP-NB | slow | 0.0055 [-0.0009; 0.0120] | 0.0100 [0.0027; 0.0177] |
| DeepAR - TSB | fast | -0.2426 [-0.3049; -0.1864] | -0.1393 [-0.2110; -0.0723] |
| DeepAR - TSB | medium | -0.0590 [-0.0747; -0.0440] | -0.0380 [-0.0513; -0.0262] |
| DeepAR - TSB | slow | -0.0209 [-0.0303; -0.0111] | -0.0127 [-0.0244; -0.0010] |

**Chi phí mô phỏng (DeepAR − đối thủ; âm = DeepAR rẻ hơn), mỗi mô hình ở q* RIÊNG chọn trên val (tổng thể), tổng thể 823 chuỗi, tỷ lệ 10:**
| so_sanh | L | val | test |
|---|---|---|---|
| DeepAR - LGBM-Q | 2 | 0.12 [-2.57; 2.91] | 6.45 [2.02; 11.89] |
| DeepAR - LGBM-Q | 3 | -0.31 [-3.46; 3.04] | 8.35 [3.91; 13.71] |
| DeepAR - MLP-NB | 2 | 2.14 [0.17; 4.11] | 6.45 [2.29; 11.54] |
| DeepAR - MLP-NB | 3 | 2.75 [0.52; 5.21] | 9.12 [4.59; 14.32] |
| DeepAR - TSB | 2 | -15.03 [-19.44; -10.99] | -13.14 [-17.59; -8.99] |
| DeepAR - TSB | 3 | -15.57 [-20.55; -11.13] | -13.25 [-17.62; -9.20] |

tỷ lệ 20:
| so_sanh | L | val | test |
|---|---|---|---|
| DeepAR - LGBM-Q | 2 | -1.07 [-5.27; 3.07] | 8.06 [0.79; 17.87] |
| DeepAR - LGBM-Q | 3 | -1.07 [-6.16; 4.00] | 19.76 [9.65; 31.66] |
| DeepAR - MLP-NB | 2 | 4.04 [1.15; 6.94] | 10.72 [2.76; 20.49] |
| DeepAR - MLP-NB | 3 | 3.46 [-0.13; 7.09] | 18.17 [9.51; 28.44] |
| DeepAR - TSB | 2 | -21.99 [-29.22; -15.15] | -13.53 [-20.49; -6.24] |
| DeepAR - TSB | 3 | -20.72 [-27.82; -13.91] | -9.81 [-17.42; -1.78] |

Theo nhóm tốc độ, L=2, tỷ lệ 10 (q* theo cột q_DeepAR/q_other trong `results/deepar_bootstrap_cost.csv`):
| so_sanh | group | val | test |
|---|---|---|---|
| DeepAR - LGBM-Q | fast | -1.42 [-7.90; 4.75] | 14.79 [3.79; 28.07] |
| DeepAR - LGBM-Q | medium | 0.71 [-1.08; 2.51] | 0.48 [-1.20; 2.23] |
| DeepAR - LGBM-Q | slow | 1.90 [-0.33; 4.37] | 1.90 [-0.32; 4.26] |
| DeepAR - MLP-NB | fast | 1.97 [-2.73; 6.35] | 12.64 [2.48; 25.23] |
| DeepAR - MLP-NB | medium | 1.89 [0.32; 3.53] | 1.07 [-0.50; 2.76] |
| DeepAR - MLP-NB | slow | 2.93 [0.99; 4.99] | 4.79 [2.55; 7.27] |
| DeepAR - TSB | fast | -33.26 [-43.64; -22.75] | -29.63 [-40.23; -19.63] |
| DeepAR - TSB | medium | -5.29 [-7.83; -2.72] | -4.36 [-6.40; -2.35] |
| DeepAR - TSB | slow | 1.00 [-1.31; 3.38] | 1.39 [-1.38; 4.40] |

tỷ lệ 20:
| so_sanh | group | val | test |
|---|---|---|---|
| DeepAR - LGBM-Q | fast | -2.82 [-12.90; 6.62] | 18.96 [0.66; 42.41] |
| DeepAR - LGBM-Q | medium | -1.95 [-5.70; 1.74] | -0.17 [-3.11; 2.85] |
| DeepAR - LGBM-Q | slow | 3.75 [-0.63; 8.01] | 2.88 [-0.97; 6.94] |
| DeepAR - MLP-NB | fast | 5.63 [-1.34; 12.30] | 23.10 [3.38; 47.48] |
| DeepAR - MLP-NB | medium | 1.61 [-1.39; 4.60] | 1.61 [-0.86; 4.13] |
| DeepAR - MLP-NB | slow | 5.52 [2.87; 8.17] | 4.42 [1.36; 7.93] |
| DeepAR - TSB | fast | -44.31 [-61.49; -27.01] | -26.45 [-43.01; -7.44] |
| DeepAR - TSB | medium | -11.77 [-15.75; -7.66] | -7.55 [-11.13; -3.94] |
| DeepAR - TSB | slow | 0.73 [-2.84; 4.39] | -0.49 [-4.69; 4.02] |


## 8. Kết luận (trung thực, theo số liệu)
- **Quyết định chốt trên val (không đổi theo test):** mô hình học sâu đề xuất là **MLP-NB** (pinball_avg val trung bình hai H: MLP-NB 0.9977 vs DeepAR 1.0081). Test cho cùng chiều: trung bình hai H MLP-NB 1.0483 vs DeepAR 1.1215 (MLP-NB tốt hơn).
- **DeepAR vs MLP-NB (dự báo):** val H=3: DeepAR 0.9147 vs MLP-NB 0.9025, hiệu 0.0121 [-0.0039; 0.0257] (không phân biệt được); val H=4: 1.1016 vs 1.0929, hiệu 0.0087 [-0.0163; 0.0292] (không phân biệt được); test H=3: 1.0045 vs 0.9477, hiệu 0.0568 [0.0241; 0.0999] (DeepAR kém hơn rõ); test H=4: 1.2384 vs 1.1489, hiệu 0.0895 [0.0387; 0.1533] (DeepAR kém hơn rõ).
- **DeepAR vs LGBM-Q (dự báo):** val H=3 hiệu 0.0038 [-0.0070; 0.0151] (không phân biệt được), val H=4 0.0094 [-0.0068; 0.0266] (không phân biệt được); test H=3 0.0691 [0.0322; 0.1149] (DeepAR kém hơn rõ), test H=4 0.1040 [0.0490; 0.1722] (DeepAR kém hơn rõ).
- **DeepAR vs TSB (dự báo):** val H=3 hiệu -0.1236 [-0.1502; -0.0988] (DeepAR tốt hơn rõ), test H=3 -0.0727 [-0.1025; -0.0446] (DeepAR tốt hơn rõ), test H=4 -0.0607 [-0.1008; -0.0210] (DeepAR tốt hơn rõ).
- **Chi phí mô phỏng, tỷ lệ 10:** L=2 test: DeepAR − MLP-NB 6.45 [2.29; 11.54] (DeepAR kém hơn rõ), − LGBM-Q 6.45 [2.02; 11.89] (DeepAR kém hơn rõ), − TSB -13.14 [-17.59; -8.99] (DeepAR tốt hơn rõ); L=3 test: − MLP-NB 9.12 [4.59; 14.32] (DeepAR kém hơn rõ), − LGBM-Q 8.35 [3.91; 13.71] (DeepAR kém hơn rõ), − TSB -13.25 [-17.62; -9.20] (DeepAR tốt hơn rõ). Trên val L=2: − MLP-NB 2.14 [0.17; 4.11] (DeepAR kém hơn rõ), − LGBM-Q 0.12 [-2.57; 2.91] (không phân biệt được), − TSB -15.03 [-19.44; -10.99] (DeepAR tốt hơn rõ).
- **Tóm tắt:** DeepAR hơn Croston/TSB-họ thống kê rõ (TSB) ở cả dự báo lẫn chi phí, nhưng KHÔNG hơn MLP-NB hay LightGBM: trên val gần như ngang (KTC chứa 0 ở phần lớn so sánh dự báo), trên test kém hơn có ý nghĩa thống kê cả về pinball lẫn chi phí. Fill rate của DeepAR tại q*(val) giảm từ val sang test nhiều hơn các mô hình khác (L=2: 0.901 → 0.881, trong khi MLP-NB 0.906 → 0.903; bảng mục 6), nhất quán với việc P(y≤q̂) của DeepAR ở test thấp hơn val khoảng 1–3 điểm phần trăm ở q = 0.5–0.9 (ví dụ q=0.5: 0.625 → 0.599; q=0.8: 0.844 → 0.829; q=0.9: 0.914 → 0.904), tức phân vị của DeepAR hơi thấp so với nhu cầu thực ở test; hiệu chuẩn kém bền hơn ở một số mô hình khác.
- **Ảnh hưởng đến ba câu hỏi nghiên cứu:** (1) phân vị vs điểm: không đổi (DeepAR chỉ là một mô hình dự báo thêm; chưa chạy lại so sánh điểm an toàn cho DeepAR). (2) nhanh vs chậm: không đổi về kết luận chung; theo nhóm (pinball_avg test, H=3), DeepAR kém hơn MLP-NB rõ nhất ở hàng nhanh (hiệu +0.1339 [0.0513; 0.2437]; 1.7017 vs 1.5679), nhẹ ở hàng chậm (+0.0100 [0.0027; 0.0177]) và không phân biệt được ở hàng trung bình (+0.0042 [−0.0029; 0.0116]); trên val không nhóm nào khác biệt rõ ngoài hàng trung bình (+0.0079 [0.0004; 0.0163]) (bảng mục 5 và 7). (3) học sâu vs Croston/TSB: **kết luận giữ nguyên và được củng cố**: cả hai mô hình học sâu (MLP-NB, DeepAR) hơn TSB rõ (DeepAR − TSB: dự báo -0.0727 [-0.1025; -0.0446] ở test H=3, chi phí -13.14 [-17.59; -8.99] ở L=2 tỷ lệ 10), nhưng mô hình học sâu thứ hai không hơn LightGBM/MLP-NB; vì vậy lợi thế so với Croston/TSB không phụ thuộc vào kiến trúc sâu cụ thể và vẫn có thể giải thích chủ yếu bởi việc dùng đặc trưng lịch sử đa thang/lịch và huấn luyện trực tiếp phân phối (xem GĐ6, GĐ7).

## 9. Hạn chế
- Chỉ 28 ngày val, 28 ngày test, một cửa hàng, một phòng ban; KTC bootstrap coi các chuỗi độc lập.
- Huấn luyện CPU với ngân sách nhỏ: tối đa 12 epoch × 20 000 cửa sổ (≈ 0.4 lần số cặp huấn luyện mỗi epoch); lưới chỉ 8 cấu hình và cấu hình tốt nhất nằm ở góc lưới; hai trong ba seed chạm trần 12 epoch → DeepAR có thể chưa hội tụ hoàn toàn (xem `results/figures/deepar_nll_curves.png`). Kết luận "DeepAR không hơn MLP-NB/LightGBM" chỉ áp dụng cho cấu hình và ngân sách này.
- Phân vị lấy từ 500 mẫu (sai số lấy mẫu ở q=0.99 lớn); một mô hình dùng cho cả hai H (khác MLP-NB); kappa=0.5 chọn trên val (thu hẹp độ phân tán, lưu ý DeepAR thuần với kappa=1 kém hơn: bảng kappa).
- Quy tắc "chưa có lần bán → dự báo 0" dùng thông tin ngày ra mắt của toàn bộ dữ liệu (chung cho mọi mô hình; mục 3.5).
- Chưa có đối chứng "DeepAR không covariates ngoại sinh" hay GRU không embedding; chưa chạy `safety_point` cho DeepAR.

## 10. Kiểm thử, hash và lệnh
Lệnh: `python -m src.deepar --stage tune_one --cfg-id K` (K=0..7, song song), `--stage select`, `--stage seed_val --seed S`, `--stage freeze`, `--stage seed_test --seed S`, `--stage collect_test`, rồi `python -m src.deepar_analysis --what sim` và `--what tables`.
sha256 `results/hparams.json` = `8bc7d7507bf0167e83b1d31f831fb8edb5edf60c23a742c67b32b0f86366765b` (không đổi so với `8BC7D750…765B`); `results/hparams_deepar.json` = `9b2036d39e56bb8d4211a93a251921229549480084fd48445fa9a1c29fdecd34`.
OUTPUT NGUYÊN VĂN `python -m pytest -v`:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- D:\lt\261\PTDLHS\DA\.venv\Scripts\python.exe
cachedir: .pytest_cache
Fugue tests will be initialized with options:
rootdir: D:\lt\261\PTDLHS\DA
plugins: anyio-4.15.1, fugue-0.9.7, platformdirs-4.14.0
collecting ... collected 64 items

tests/test_eda.py::test_load_train_excludes_val PASSED                   [  1%]
tests/test_eda.py::test_active_rows_drops_pre_launch_days PASSED         [  3%]
tests/test_eda.py::test_weekday_table_has_seven_rows_and_mean_two PASSED [  4%]
tests/test_eda.py::test_demand_speed_table_totals PASSED                 [  6%]
tests/test_final_eval.py::test_group_policy_levels_follow_group_choice PASSED [  7%]
tests/test_final_eval.py::test_case_levels_point_and_quantile PASSED     [  9%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_daily_target_date_tracks_horizon_offset PASSED [ 10%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_evaluate_validation_summary_matches_by_date_and_id PASSED [ 12%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_join_rows_do_not_duplicate PASSED [ 14%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_n_samples_changes_quantile_output PASSED [ 15%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_pinball_known_answer PASSED [ 17%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_poisson_distribution_scaling PASSED [ 18%]
tests/test_forecast_prob.py::ForecastProbEvaluationTest::test_sample_total_window_does_not_blow_up_for_zero_mean PASSED [ 20%]
tests/test_gd6.py::test_k_grid_is_0p5_to_3p0_step_0p1 PASSED             [ 21%]
tests/test_gd6.py::test_series_cost_matches_hold_plus_ratio_times_lost PASSED [ 23%]
tests/test_gd6.py::test_mase_scale_uses_train_only_and_equals_hand_calculation PASSED [ 25%]
tests/test_gd6.py::test_ext_quantiles_below_main_grid_and_monotone PASSED [ 26%]
tests/test_gd6.py::test_lgbm_ext_quantiles_not_above_first_main_quantile PASSED [ 28%]
tests/test_gd6.py::test_exog_mask_drops_only_price_snap_event_columns PASSED [ 29%]
tests/test_gd6.py::test_mlp_seed_changes_weights_but_same_seed_reproduces PASSED [ 31%]
tests/test_gd7.py::test_lgbm_q_levels_are_nonnegative_integers_and_round_to_nearest PASSED [ 32%]
tests/test_gd7.py::test_nb_models_are_not_changed_by_rounding PASSED     [ 34%]
tests/test_gd7.py::test_real_forecasts_levels_used_in_simulation_are_integers PASSED [ 35%]
tests/test_gd7.py::test_smaller_kappa_gives_narrower_upper_tail_and_spread PASSED [ 37%]
tests/test_gd7.py::test_effect_mask_drops_only_the_two_effect_columns PASSED [ 39%]
tests/test_gd8.py::test_nb_nll_matches_scipy_logpmf PASSED               [ 40%]
tests/test_gd8.py::test_no_leak_future_values_after_origin_do_not_change_forecast PASSED [ 42%]
tests/test_gd8.py::test_same_seed_same_result_different_seed_different_weights PASSED [ 43%]
tests/test_gd8.py::test_first_step_sample_mean_matches_mu PASSED         [ 45%]
tests/test_gd8.py::test_forecast_schema_integer_nonnegative_monotone PASSED [ 46%]
tests/test_gd8.py::test_sample_quantile_is_inverted_cdf PASSED           [ 48%]
tests/test_gd8.py::test_old_result_files_unchanged_since_baseline PASSED [ 50%]
tests/test_gd8.py::test_hparams_json_frozen PASSED                       [ 51%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[0-2] PASSED [ 53%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[0-3] PASSED [ 54%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[1-2] PASSED [ 56%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[1-3] PASSED [ 57%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[2-2] PASSED [ 59%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[2-3] PASSED [ 60%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[3-2] PASSED [ 62%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[3-3] PASSED [ 64%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[4-2] PASSED [ 65%]
tests/test_inventory_reference.py::test_vectorized_matches_reference[4-3] PASSED [ 67%]
tests/test_inventory_sim.py::test_constant_demand_matches_hand_calculation PASSED [ 68%]
tests/test_inventory_sim.py::test_order_accounting_and_inventory_position_after_ordering PASSED [ 70%]
tests/test_inventory_sim.py::test_costs_nonnegative_and_fill_rate_range_and_monotone_in_level PASSED [ 71%]
tests/test_inventory_sim.py::test_reproducible_same_inputs PASSED        [ 73%]
tests/test_outputs.py::test_val_npz_shapes_cover_all_series_and_origins_and_monotone PASSED [ 75%]
tests/test_outputs.py::test_val_eval_csv_schema_and_coverage_matches_npz PASSED [ 76%]
tests/test_outputs.py::test_actual_in_npz_equals_window_sum_of_val_demand PASSED [ 78%]
tests/test_outputs.py::test_md_table_rounds_to_four_decimals PASSED      [ 79%]
tests/test_outputs.py::test_test_npz_origins_inside_test_split_and_shapes PASSED [ 81%]
tests/test_prob_models.py::test_pinball_known_answer PASSED              [ 82%]
tests/test_prob_models.py::test_nb_quantiles_match_scipy_and_cdf_definition PASSED [ 84%]
tests/test_prob_models.py::test_nb_quantiles_monotone_in_q_and_zero_mean PASSED [ 85%]
tests/test_prob_models.py::test_legacy_sampler_mean_not_inflated_when_n_below_one PASSED [ 87%]
tests/test_prob_models.py::test_window_targets_values_and_nan_tail PASSED [ 89%]
tests/test_prob_models.py::test_origins_window_inside_split PASSED       [ 90%]
tests/test_prob_models.py::test_croston_tsb_constant_demand PASSED       [ 92%]
tests/test_prob_models.py::test_no_leak_stat_forecast_and_alpha PASSED   [ 93%]
tests/test_prob_models.py::test_no_leak_features PASSED                  [ 95%]
tests/test_prob_models.py::test_estimate_alpha_known_poisson_is_small PASSED [ 96%]
tests/test_prob_models.py::test_evaluate_forecast_coverage_and_shapes PASSED [ 98%]
tests/test_prob_models.py::test_lgbm_quantiles_monotone_and_shape PASSED [100%]

============================== warnings summary ===============================
tests/test_gd8.py::test_first_step_sample_mean_matches_mu
  D:\lt\261\PTDLHS\DA\tests\test_gd8.py:75: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\torch\csrc\autograd\generated\python_variable_methods.cpp:821.)
    assert abs(float(s.mean()) - float(mu)) / float(mu) < 0.05

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 64 passed, 1 warning in 15.16s ========================
```
