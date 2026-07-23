# Standalone Benchmark Pipeline

Đây là thư mục độc lập (standalone) chứa toàn bộ code, model, và dữ liệu cần thiết để chạy benchmark đo tốc độ các giai đoạn xử lý ảnh của các mô hình khác nhau. Bạn có thể copy toàn bộ thư mục `benchmark_standalone` này sang bất kỳ máy nào (bao gồm Google Colab) để chạy mà không cần phụ thuộc vào API server.

Pipeline benchmark được thiết kế để đo đạc và so sánh trực tiếp tốc độ của **3 phương pháp** (theo format của Journal Paper):
- **Ours (Lightweight Pipeline)**: RetinaFace + DTGAN + Feathered Blending
- **Baseline 1**: MTCNN + AnimeGANv2 + Poisson Blending
- **Baseline 2**: YOLOv8-Face + CartoonGAN + Alpha Blending

## 1. Cài đặt thư viện

Mở terminal tại thư mục này và cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 2. Tải Models (Tự động)

Để chạy được 3 pipeline thực tế, bạn cần tải các models (MTCNN, AnimeGAN, YOLO, CartoonGAN) về thư mục `onnx_model/`. Chạy script tự động tải:
```bash
python download_baselines.py
```
*Lưu ý: Script sẽ tự động download các models ONNX từ HuggingFace về thư mục `onnx_model`.*

## 3. Chuẩn bị ảnh test

Đã có sẵn thư mục `test_images/` với cấu trúc:
- `test_images/no_face/` (dành cho $N=0$ - ảnh không có khuôn mặt)
- `test_images/single_face/` (dành cho $N=1$ - ảnh có 1 khuôn mặt)
- `test_images/group_photo/` (dành cho $N=3$ - ảnh có ít nhất 3 khuôn mặt)

Bạn có thể thêm/bớt ảnh vào các thư mục này. Script sẽ tự động quét toàn bộ ảnh trong mỗi thư mục và tính trung bình (mean), độ lệch chuẩn (std) cho từng thư mục (tương ứng từng cột trong bảng kết quả).

## 4. Chạy Benchmark

Chạy script với lệnh sau:
```bash
# Chạy đánh giá cho toàn bộ các mô hình trên GPU
python benchmark_pipeline.py --runs 5 --warmup 5 --device gpu --pipeline all
```

Các tham số tùy chọn:
- `--pipeline`: Pipeline cần đánh giá (`ours`, `baseline1`, `baseline2`, `all`). Mặc định: `all`.
- `--device`: Môi trường tính toán (`cpu`, `gpu`). Mặc định: `gpu`. Nếu chạy trên Google Colab T4/T8, hãy để `gpu`.
- `--runs`: Số lần chạy benchmark trên mỗi ảnh để tính trung bình. Mặc định: `50`.
- `--warmup`: Số lần chạy khởi động trước khi đo (để model load vào RAM/VRAM). Mặc định: `5`.

## 5. Kết quả Output

Sau khi chạy xong, script sẽ:
1. In kết quả trực tiếp ra **Console** với format dạng bảng.
2. Lưu kết quả chi tiết ra file **CSV** (`benchmark_results.csv`).
3. Tạo file **LaTeX** (`benchmark_results.tex`) để bạn có thể copy/paste trực tiếp mã LaTeX vào bài báo (paper).
