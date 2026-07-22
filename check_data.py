import os
import sys
import numpy as np
from PIL import Image

# Thêm path hiện tại để có thể import face_det
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_det import detect_face

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def get_image_files(folder):
    if not os.path.exists(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    return files

def main():
    print("="*60)
    print(" KIỂM TRA ĐIỀU KIỆN ẢNH TEST BENCHMARK")
    print("="*60)

    base_dir = "test_images"
    folders = {
        "no_face": 0,
        "single_face": 1,
        "group_photo": 3
    }

    if not os.path.exists(base_dir):
        print(f"❌ THẤT BẠI: Không tìm thấy thư mục gốc '{base_dir}'.")
        sys.exit(1)

    print("⏳ Đang tải mô hình nhận diện khuôn mặt (RetinaFace) để kiểm tra chéo...")

    all_resolutions = set()
    has_error = False

    for folder_name, expected_faces in folders.items():
        folder_path = os.path.join(base_dir, folder_name)
        print(f"\n📂 Đang kiểm tra: {folder_name}/ (Yêu cầu: {expected_faces} khuôn mặt)")
        
        if not os.path.exists(folder_path):
            print(f"  ❌ Thiếu thư mục: {folder_name}")
            has_error = True
            continue
            
        images = get_image_files(folder_path)
        if not images:
            print(f"  ❌ Thư mục trống. Hãy thêm ít nhất 1 ảnh.")
            has_error = True
            continue

        print(f"  ✅ Tìm thấy {len(images)} ảnh.")
        
        for img_path in images:
            img_name = os.path.basename(img_path)
            
            # 1. Kiểm tra kích thước và định dạng
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    w, h = img.size
                    resolution = f"{w}x{h}"
                    all_resolutions.add(resolution)
                    max_edge = max(w, h)
                    
                    if max_edge > 1280:
                        print(f"  ⚠️ CẢNH BÁO [{img_name}]: Kích thước {resolution} > 1280. Cạnh lớn nhất sẽ bị scale xuống 1280 trong lúc benchmark (tạo ra thời gian trễ overhead).")
            except Exception as e:
                print(f"  ❌ Lỗi đọc ảnh [{img_name}]: {e}")
                has_error = True
                continue

            # 2. Kiểm tra số lượng khuôn mặt bằng AI
            try:
                img_np = np.array(img)
                # cv2 format is BGR, so if face_det expects BGR, we should convert or test it.
                # Since face_det works with PIL image converted to numpy array (RGB) typically, we use it directly as done in pipeline.
                bboxes, _ = detect_face(img_np)
                detected_faces = len(bboxes) if bboxes is not None else 0
                
                if detected_faces != expected_faces:
                    print(f"  ❌ LỖI [{img_name}]: Phát hiện {detected_faces} khuôn mặt, nhưng bắt buộc phải là {expected_faces}.")
                    has_error = True
                else:
                    print(f"    ✔️ [{img_name}]: {resolution}, số mặt: {detected_faces}")
            except Exception as e:
                print(f"  ❌ Lỗi chạy RetinaFace [{img_name}]: {e}")
                has_error = True

    print("\n" + "="*60)
    print(" KẾT LUẬN")
    print("="*60)
    
    if len(all_resolutions) > 1:
        print(f"⚠️ CẢNH BÁO: ĐỘ PHÂN GIẢI KHÔNG ĐỒNG NHẤT!")
        print(f"   Bộ dữ liệu đang chứa nhiều kích thước khác nhau: {', '.join(list(all_resolutions))}")
        print("   -> Lời khuyên: Để bài báo công bằng (chỉ khác nhau ở số lượng mặt), bạn nên resize tất cả ảnh về cùng 1 độ phân giải trước khi test.")
    elif len(all_resolutions) == 1:
        print(f"✅ ĐỘ PHÂN GIẢI ĐỒNG NHẤT: {list(all_resolutions)[0]} (rất tốt cho bài báo).")
        
    if has_error:
        print("❌ DỮ LIỆU CHƯA ĐẠT CHUẨN. Bạn cần sửa các lỗi bên trên trước khi chạy benchmark.")
    else:
        print("✅ DỮ LIỆU ĐÃ ĐẠT CHUẨN! Bạn có thể tự tin chạy benchmark ngay.")
        print("   Lệnh: python benchmark_pipeline.py --model hayao --runs 50")

if __name__ == '__main__':
    main()
