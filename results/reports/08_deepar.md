# Báo cáo GĐ8 – DeepAR-style (GRU + Negative Binomial, lấy mẫu tự hồi quy)

> **Trạng thái: bản cam kết trước khi chạy TEST.** File này được commit TRƯỚC khi chạy test; các mục kết quả được điền sau.

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
