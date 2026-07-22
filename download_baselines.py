import os
import urllib.request
from tqdm import tqdm

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        response = urllib.request.urlopen(url)
        total_size = int(response.info().get('Content-Length').strip())
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                while True:
                    buffer = response.read(8192)
                    if not buffer:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))
        print(f"Successfully downloaded {output_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    os.makedirs("onnx_model", exist_ok=True)
    
    # 1. AnimeGANv2
    download_file(
        "https://huggingface.co/akhaliq/AnimeGANv2-ONNX/resolve/main/AnimeGANv2_Hayao.onnx",
        "onnx_model/AnimeGANv2_Hayao.onnx"
    )
    
    # 2. YOLOv8-Face
    download_file(
        "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx",
        "onnx_model/yolov8n-face.onnx"
    )
    
    # 3. CartoonGAN (Placeholder using another style of AnimeGANv2 since CartoonGAN ONNX is rare)
    download_file(
        "https://huggingface.co/vumichien/AnimeGANv2_Shinkai/resolve/main/AnimeGANv2_Shinkai.onnx",
        "onnx_model/CartoonGAN_Placeholder.onnx"
    )
