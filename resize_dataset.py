import os
import sys

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image

def resize_images_in_folder(folder_path, target_size=(256, 256)):
    if not os.path.exists(folder_path):
        print(f"[X] Thu muc khong ton tai: {folder_path}")
        return 0

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
    
    count = 0
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                if img.size != target_size:
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    resized_img.save(file_path, quality=95)
                    count += 1
                    print(f"  [OK] Scaled {filename}: {img.size[0]}x{img.size[1]} -> {target_size[0]}x{target_size[1]}")
                else:
                    print(f"  [SKIP] {filename} da at kich thuoc {target_size[0]}x{target_size[1]}")
        except Exception as e:
            print(f"  [ERROR] Loi khi xu ly {filename}: {e}")

    return count

if __name__ == "__main__":
    base_dir = "test_images"
    target_res = (256, 256)

    print("=" * 60)
    print(f"DANG SCALE ANH TRONG THU MUC TEST VE {target_res[0]}x{target_res[1]}")
    print("=" * 60)

    folders_to_scale = ["no_face", "group_photo"]
    total_scaled = 0

    for folder in folders_to_scale:
        folder_path = os.path.join(base_dir, folder)
        print(f"\nThu muc: {folder}/")
        scaled = resize_images_in_folder(folder_path, target_res)
        total_scaled += scaled

    print("\n" + "=" * 60)
    print(f"Da scale thanh cong {total_scaled} anh ve kich thuoc {target_res[0]}x{target_res[1]}!")
    print("=" * 60)
