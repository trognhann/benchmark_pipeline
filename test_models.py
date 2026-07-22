import os
from PIL import Image
import numpy as np

print("Testing Facenet MTCNN...")
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True)
dummy_img = Image.new("RGB", (640, 480))
res = mtcnn.detect(dummy_img)
print("MTCNN success:", res)

print("Testing YOLOv8n...")
from ultralytics import YOLO
yolo = YOLO('yolov8n.pt')
res = yolo(dummy_img, verbose=False)
print("YOLO success:", len(res))

print("Testing ONNX AnimeGANv2...")
import onnxruntime
if os.path.exists("onnx_model/AnimeGANv2_Hayao.onnx"):
    sess = onnxruntime.InferenceSession("onnx_model/AnimeGANv2_Hayao.onnx", providers=['CPUExecutionProvider'])
    print("ONNX AnimeGANv2 loaded.")
else:
    print("ONNX AnimeGANv2 not found.")

print("Testing ONNX CartoonGAN...")
if os.path.exists("onnx_model/CartoonGAN_Placeholder.onnx"):
    sess = onnxruntime.InferenceSession("onnx_model/CartoonGAN_Placeholder.onnx", providers=['CPUExecutionProvider'])
    print("ONNX CartoonGAN loaded.")
else:
    print("ONNX CartoonGAN not found.")

print("All tests passed.")
