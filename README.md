# Standalone Benchmark Pipeline

Đây là thư mục độc lập (standalone) chứa toàn bộ code, model, và dữ liệu cần thiết để chạy benchmark đo tốc độ các giai đoạn xử lý ảnh của mô hình. Bạn có thể copy toàn bộ thư mục `benchmark_standalone` này sang bất kỳ máy nào để chạy mà không cần phụ thuộc vào API server.

## Cài đặt thư viện

Mở terminal tại thư mục này và cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Chuẩn bị ảnh test

Đã có sẵn thư mục `test_images/` với cấu trúc:
- `test_images/no_face/` (dành cho $N=0$ - ảnh không có khuôn mặt)
- `test_images/single_face/` (dành cho $N=1$ - ảnh có 1 khuôn mặt)
- `test_images/group_photo/` (dành cho $N=3$ - ảnh có 3 khuôn mặt)

Bạn có thể thêm/bớt ảnh vào các thư mục này. Script sẽ tự động quét toàn bộ ảnh trong mỗi thư mục và tính trung bình (mean), độ lệch chuẩn (std) cho từng thư mục (tương ứng từng cột trong bảng kết quả).

## Chạy Benchmark

Chạy script với lệnh sau:
```bash
python benchmark_pipeline.py --model hayao --runs 50 --warmup 5
```

Các tham số tùy chọn:
- `--model`: Tên mô hình (vd: `hayao`, `shinkai`). Mặc định: `hayao`.
- `--runs`: Số lần chạy benchmark trên mỗi ảnh để tính trung bình. Mặc định: `50`.
- `--warmup`: Số lần chạy khởi động trước khi đo (để model load vào RAM/VRAM). Mặc định: `5`.

## Kết quả Output

Sau khi chạy xong, script sẽ:
1. In kết quả trực tiếp ra **Console** với format dạng bảng.
2. Lưu kết quả chi tiết ra file **CSV** (`benchmark_results.csv`).
3. Tạo file **LaTeX** (`benchmark_results.tex`) để bạn có thể copy/paste trực tiếp vào bài báo (paper).
