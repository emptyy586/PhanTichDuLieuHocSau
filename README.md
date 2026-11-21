# M5 Forecasting - Dự án phân tích dữ liệu và học sâu

## Mục tiêu
Dự án phân tích chuỗi thời gian M5 Walmart theo cửa hàng CA_1 và nhóm hàng FOODS_3, tập trung vào:
- EDA và phân loại chuỗi theo ADI/CV²
- Baseline thống kê theo các mô hình WindowAverage, SeasonalNaive, CrostonSBA, TSB
- Chuẩn bị dữ liệu cho giai đoạn mô phỏng tồn kho và dự báo phân vị sau này

## Cách chạy lại từ đầu
1. Tạo môi trường ảo Python 3.12:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Cài đặt phụ thuộc:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. Chạy pipeline dữ liệu:
   ```powershell
   python src/data.py
   ```
4. Mở notebook EDA và phân loại trong thư mục notebooks/.

## Cấu trúc thư mục chính
- data/raw/: dữ liệu gốc M5
- data/processed/: dữ liệu đã xử lý dạng parquet
- src/: các script Python cho tiền xử lý và mô hình
- notebooks/: notebook phân tích và EDA
- results/: kết quả số liệu và biểu đồ
- configs/: file cấu hình YAML

## Lưu ý quan trọng
- Không đọc dữ liệu bằng Excel; chỉ dùng pandas.
- Các thống kê phân loại chuỗi chỉ tính trên tập train.
- Luôn giữ seed cố định ở một nơi duy nhất trong config hoặc src/utils.py.

## Giai đoạn A (đang phát triển)
- Khởi tạo file mới `src/forecast_prob.py` cho dự báo phân phối và LightGBM xác suất.
- Mục tiêu: ước lượng phân phối nhu cầu rời rạc không âm bằng NB/Poisson, dự báo rolling-origin trên validation, và đánh giá pinball/coverage theo quantile.
- Tham số chính được đặt trong `configs/config.yaml` và có thể điều chỉnh theo `lead_time`, `review_period`, `quantile_levels`, `n_samples`, `refit_every`.
- Giai đoạn A chưa thay thế các output hiện có; file mới lưu dưới `results/` với tên rõ ràng như `forecast_eval_val_smoke.csv` khi chạy smoke test.
