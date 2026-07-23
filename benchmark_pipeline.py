"""
Pipeline Benchmark – Tách từng stage để đo latency cho bảng đánh giá.

Cấu trúc ảnh đầu vào:
    test_images/
    ├── no_face/        (dành cho N=0)
    ├── single_face/    (dành cho N=1)
    └── group_photo/    (dành cho N=3)

Cách dùng:
    python benchmark_pipeline.py [--test-dir test_images] [--runs 50] [--warmup 5] [--device gpu] [--pipeline all]
"""

import os
import sys
import time
import argparse
import csv
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw

# Đọc tham số --device trước khi import face_det để set ORT_DEVICE
_device = "gpu"
if "--device" in sys.argv:
    _idx = sys.argv.index("--device")
    if _idx + 1 < len(sys.argv):
        _device = sys.argv[_idx + 1].lower()
os.environ["ORT_DEVICE"] = _device
torch_device = "cuda" if _device == "gpu" else "cpu"

# ─── Import các module từ project ──────────────────────────────────────
import face_det

import onnxruntime

ort_sess_options = onnxruntime.SessionOptions()
ort_sess_options.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", 0))


# ─── Constants ─────────────────────────────────────────────────────────
MAX_EDGE = 1280
ONNX_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")

# ─── Helpers (giữ nguyên logic) ─────────────────────────

def _limit_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_edge = max(w, h)
    if max_edge > MAX_EDGE:
        scale = MAX_EDGE / max_edge
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
    return img

def _preprocess(img: Image.Image, face_mode: bool, onnx_path: str) -> np.ndarray:
    if face_mode:
        img = img.resize((512, 512), Image.LANCZOS)
    else:
        h, w = np.array(img).shape[:2]
        def to_8s(x): return 256 if x < 256 else x - x % 8
        def to_16s(x): return 256 if x < 256 else x - x % 16
        if "_tiny_" in onnx_path:
            img = img.resize((to_16s(w), to_16s(h)), Image.LANCZOS)
        else:
            img = img.resize((to_8s(w), to_8s(h)), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)

def _postprocess(ort_outs: np.ndarray, original_size: tuple, face_mode: bool) -> Image.Image:
    images = (ort_outs + 1.0) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    if face_mode:
        result = images[0]
    else:
        result = np.concatenate([x for x in images], axis=1)
    pil_img = Image.fromarray(result)
    if not face_mode:
        pil_img = pil_img.resize(original_size, Image.LANCZOS)
    return pil_img

@dataclass
class StageTimings:
    t_detect: float = 0.0
    t_bg_infer: float = 0.0
    t_per_face_crop_infer: float = 0.0
    t_blend: float = 0.0
    t_total: float = 0.0
    n_faces: int = 0

# ═══════════════════════════════════════════════════════════════════════
#  Pipeline benchmark - Ours
# ═══════════════════════════════════════════════════════════════════════

def benchmark_ours_single_run(img: Image.Image, ort_session: onnxruntime.InferenceSession, onnx_path: str) -> StageTimings:
    timings = StageTimings()
    original_size = img.size
    img_limit = _limit_size(img)
    img_np = np.array(img_limit)

    t0 = time.perf_counter()
    bboxes, points = face_det.detect_face(img_np)
    timings.t_detect = (time.perf_counter() - t0) * 1000

    has_faces = bboxes is not None and len(bboxes) > 0
    timings.n_faces = len(bboxes) if has_faces else 0

    t0 = time.perf_counter()
    inp_bg = _preprocess(img_limit, False, onnx_path)
    out_bg = ort_session.run(None, {ort_session.get_inputs()[0].name: inp_bg})[0]
    bg_result_img = _postprocess(out_bg, original_size, False)
    timings.t_bg_infer = (time.perf_counter() - t0) * 1000

    if not has_faces:
        timings.t_per_face_crop_infer = 0.0
        timings.t_blend = 0.0
        timings.t_total = timings.t_detect + timings.t_bg_infer
        return timings

    w, h = original_size
    scale = 1.0 if max(w, h) <= MAX_EDGE else MAX_EDGE / max(w, h)

    total_face_infer_ms = 0.0
    total_blend_ms = 0.0

    for box in bboxes:
        t0 = time.perf_counter()
        margin_box = face_det.margin_face(box, img_np.shape[:2])
        x1, y1, x2, y2 = margin_box
        face_np = img_np[y1:y2, x1:x2]
        if face_np.size == 0: continue
        face_img = Image.fromarray(face_np)

        inp_face = _preprocess(face_img, True, onnx_path)
        out_face = ort_session.run(None, {ort_session.get_inputs()[0].name: inp_face})[0]
        face_result_img = _postprocess(out_face, original_size, True)
        total_face_infer_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        orig_x1, orig_y1 = int(round(x1 / scale)), int(round(y1 / scale))
        orig_x2, orig_y2 = int(round(x2 / scale)), int(round(y2 / scale))
        face_w, face_h = max(1, orig_x2 - orig_x1), max(1, orig_y2 - orig_y1)

        face_result_resized = face_result_img.resize((face_w, face_h), Image.LANCZOS)
        mask = Image.new("L", (face_w, face_h), 0)
        draw = ImageDraw.Draw(mask)
        margin = max(1, int(min(face_w, face_h) * 0.08))
        draw.rectangle([margin, margin, face_w - margin, face_h - margin], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=margin))
        bg_result_img.paste(face_result_resized, (orig_x1, orig_y1), mask)
        total_blend_ms += (time.perf_counter() - t0) * 1000

    timings.t_per_face_crop_infer = total_face_infer_ms
    timings.t_blend = total_blend_ms
    timings.t_total = timings.t_detect + timings.t_bg_infer + timings.t_per_face_crop_infer + timings.t_blend
    return timings

# ═══════════════════════════════════════════════════════════════════════
#  Pipeline benchmark - Baseline 1 (MTCNN + AnimeGANv2 + Poisson)
# ═══════════════════════════════════════════════════════════════════════

def benchmark_baseline1_single_run(img: Image.Image, mtcnn_model, animegan_session) -> StageTimings:
    timings = StageTimings()
    original_size = img.size
    img_limit = _limit_size(img)
    img_np = np.array(img_limit)

    t0 = time.perf_counter()
    # MTCNN returns boxes, probs
    boxes, _ = mtcnn_model.detect(img_limit)
    timings.t_detect = (time.perf_counter() - t0) * 1000

    has_faces = boxes is not None and len(boxes) > 0
    timings.n_faces = len(boxes) if has_faces else 0
    bboxes = boxes if has_faces else []

    t0 = time.perf_counter()
    inp_bg = _preprocess(img_limit, False, "animegan")
    out_bg = animegan_session.run(None, {animegan_session.get_inputs()[0].name: inp_bg})[0]
    bg_result_img = _postprocess(out_bg, original_size, False)
    timings.t_bg_infer = (time.perf_counter() - t0) * 1000

    if not has_faces:
        timings.t_total = timings.t_detect + timings.t_bg_infer
        return timings

    w, h = original_size
    scale = 1.0 if max(w, h) <= MAX_EDGE else MAX_EDGE / max(w, h)

    total_face_infer_ms = 0.0
    total_blend_ms = 0.0
    
    bg_cv2 = cv2.cvtColor(np.array(bg_result_img), cv2.COLOR_RGB2BGR)

    for box in bboxes:
        t0 = time.perf_counter()
        margin_box = face_det.margin_face(box, img_np.shape[:2])
        x1, y1, x2, y2 = margin_box
        face_np = img_np[y1:y2, x1:x2]
        if face_np.size == 0: continue
        face_img = Image.fromarray(face_np)

        inp_face = _preprocess(face_img, True, "animegan")
        out_face = animegan_session.run(None, {animegan_session.get_inputs()[0].name: inp_face})[0]
        face_result_img = _postprocess(out_face, original_size, True)
        total_face_infer_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        orig_x1, orig_y1 = int(round(x1 / scale)), int(round(y1 / scale))
        orig_x2, orig_y2 = int(round(x2 / scale)), int(round(y2 / scale))
        face_w, face_h = max(1, orig_x2 - orig_x1), max(1, orig_y2 - orig_y1)

        face_result_resized = face_result_img.resize((face_w, face_h), Image.LANCZOS)
        
        # Poisson Blending
        face_cv2 = cv2.cvtColor(np.array(face_result_resized), cv2.COLOR_RGB2BGR)
        mask = 255 * np.ones(face_cv2.shape[:2], face_cv2.dtype)
        center = (orig_x1 + face_w // 2, orig_y1 + face_h // 2)
        try:
            bg_cv2 = cv2.seamlessClone(face_cv2, bg_cv2, mask, center, cv2.NORMAL_CLONE)
        except Exception as e:
            # Fallback if center out of bounds
            bg_result_img = Image.fromarray(cv2.cvtColor(bg_cv2, cv2.COLOR_BGR2RGB))
            bg_result_img.paste(face_result_resized, (orig_x1, orig_y1))
            bg_cv2 = cv2.cvtColor(np.array(bg_result_img), cv2.COLOR_RGB2BGR)
            
        total_blend_ms += (time.perf_counter() - t0) * 1000

    timings.t_per_face_crop_infer = total_face_infer_ms
    timings.t_blend = total_blend_ms
    timings.t_total = timings.t_detect + timings.t_bg_infer + timings.t_per_face_crop_infer + timings.t_blend
    return timings

# ═══════════════════════════════════════════════════════════════════════
#  Pipeline benchmark - Baseline 2 (YOLOv8-Face + CartoonGAN + Alpha)
# ═══════════════════════════════════════════════════════════════════════

def benchmark_baseline2_single_run(img: Image.Image, yolo_model, cartoon_session) -> StageTimings:
    timings = StageTimings()
    original_size = img.size
    img_limit = _limit_size(img)
    img_np = np.array(img_limit)

    t0 = time.perf_counter()
    results = yolo_model(img_limit, verbose=False, classes=[0]) # Detect persons as face proxy
    boxes = results[0].boxes.xyxy.cpu().numpy()
    timings.t_detect = (time.perf_counter() - t0) * 1000

    has_faces = boxes is not None and len(boxes) > 0
    timings.n_faces = len(boxes) if has_faces else 0
    bboxes = boxes if has_faces else []

    t0 = time.perf_counter()
    inp_bg = _preprocess(img_limit, False, "cartoongan")
    out_bg = cartoon_session.run(None, {cartoon_session.get_inputs()[0].name: inp_bg})[0]
    bg_result_img = _postprocess(out_bg, original_size, False)
    timings.t_bg_infer = (time.perf_counter() - t0) * 1000

    if not has_faces:
        timings.t_total = timings.t_detect + timings.t_bg_infer
        return timings

    w, h = original_size
    scale = 1.0 if max(w, h) <= MAX_EDGE else MAX_EDGE / max(w, h)

    total_face_infer_ms = 0.0
    total_blend_ms = 0.0

    for box in bboxes:
        t0 = time.perf_counter()
        margin_box = face_det.margin_face(box, img_np.shape[:2])
        x1, y1, x2, y2 = margin_box
        face_np = img_np[y1:y2, x1:x2]
        if face_np.size == 0: continue
        face_img = Image.fromarray(face_np)

        inp_face = _preprocess(face_img, True, "cartoongan")
        out_face = cartoon_session.run(None, {cartoon_session.get_inputs()[0].name: inp_face})[0]
        face_result_img = _postprocess(out_face, original_size, True)
        total_face_infer_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        orig_x1, orig_y1 = int(round(x1 / scale)), int(round(y1 / scale))
        orig_x2, orig_y2 = int(round(x2 / scale)), int(round(y2 / scale))
        face_w, face_h = max(1, orig_x2 - orig_x1), max(1, orig_y2 - orig_y1)

        face_result_resized = face_result_img.resize((face_w, face_h), Image.LANCZOS)
        
        # Alpha Blending (simple blend 0.5 or using alpha paste directly)
        mask = Image.new("L", (face_w, face_h), 128) # 50% alpha blend example
        bg_result_img.paste(face_result_resized, (orig_x1, orig_y1), mask)
        
        total_blend_ms += (time.perf_counter() - t0) * 1000

    timings.t_per_face_crop_infer = total_face_infer_ms
    timings.t_blend = total_blend_ms
    timings.t_total = timings.t_detect + timings.t_bg_infer + timings.t_per_face_crop_infer + timings.t_blend
    return timings

# ═══════════════════════════════════════════════════════════════════════
#  Chạy benchmark nhiều lần + thống kê
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    pipeline: str = ""
    label: str = ""
    n_faces: int = 0
    n_runs: int = 0
    avg_detect: float = 0.0
    avg_bg_infer: float = 0.0
    avg_face_infer: float = 0.0
    avg_blend: float = 0.0
    avg_total: float = 0.0
    fps: float = 0.0
    std_detect: float = 0.0
    std_bg_infer: float = 0.0
    std_face_infer: float = 0.0
    std_blend: float = 0.0
    std_total: float = 0.0
    raw_timings: list[StageTimings] = field(default_factory=list)


def run_benchmark(
    img: Image.Image,
    models: dict,
    pipeline: str,
    label: str = "",
    runs: int = 50,
    warmup: int = 5,
) -> BenchmarkResult:
    
    print(f"  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        if pipeline == "ours":
            benchmark_ours_single_run(img, models['ours_ort'], models['ours_onnx_path'])
        elif pipeline == "baseline1":
            benchmark_baseline1_single_run(img, models['mtcnn'], models['animegan'])
        else:
            benchmark_baseline2_single_run(img, models['yolov8'], models['cartoongan'])

    print(f"  Benchmarking ({runs} runs)...")
    all_timings: list[StageTimings] = []
    for i in range(runs):
        if pipeline == "ours":
            t = benchmark_ours_single_run(img, models['ours_ort'], models['ours_onnx_path'])
        elif pipeline == "baseline1":
            t = benchmark_baseline1_single_run(img, models['mtcnn'], models['animegan'])
        else:
            t = benchmark_baseline2_single_run(img, models['yolov8'], models['cartoongan'])
        all_timings.append(t)
        if (i + 1) % 10 == 0:
            print(f"    ... {i + 1}/{runs}")

    detect_arr = np.array([t.t_detect for t in all_timings])
    bg_arr = np.array([t.t_bg_infer for t in all_timings])
    face_arr = np.array([t.t_per_face_crop_infer for t in all_timings])
    blend_arr = np.array([t.t_blend for t in all_timings])
    total_arr = np.array([t.t_total for t in all_timings])

    avg_tot = float(np.mean(total_arr))
    return BenchmarkResult(
        pipeline=pipeline,
        label=label,
        n_faces=all_timings[0].n_faces,
        n_runs=runs,
        avg_detect=float(np.mean(detect_arr)),
        avg_bg_infer=float(np.mean(bg_arr)),
        avg_face_infer=float(np.mean(face_arr)),
        avg_blend=float(np.mean(blend_arr)),
        avg_total=avg_tot,
        fps=1000.0 / avg_tot if avg_tot > 0 else 0.0,
        std_detect=float(np.std(detect_arr)),
        std_bg_infer=float(np.std(bg_arr)),
        std_face_infer=float(np.std(face_arr)),
        std_blend=float(np.std(blend_arr)),
        std_total=float(np.std(total_arr)),
        raw_timings=all_timings,
    )

# ═══════════════════════════════════════════════════════════════════════
#  In kết quả dạng bảng và xuất file
# ═══════════════════════════════════════════════════════════════════════

def print_result_table(results: list[BenchmarkResult]):
    print()
    for pl in ["ours", "baseline1", "baseline2"]:
        pl_results = [r for r in results if r.pipeline == pl]
        if not pl_results: continue
        
        print("=" * 100)
        print(f"  BENCHMARK RESULTS – PIPELINE: {pl.upper()}")
        print("=" * 100)
        headers = ["Stage / Sub-Model"]
        for r in pl_results: headers.append(r.label)
        col_w, data_w = 28, 22

        print(f"{'Stage / Sub-Model':<{col_w}}", end="")
        for r in pl_results: print(f"{r.label:>{data_w}}", end="")
        print()
        print("─" * (col_w + data_w * len(pl_results)))

        print(f"{'Detection (T_detect)':<{col_w}}", end="")
        for r in pl_results: print(f"{'[%.1f ms]' % r.avg_detect:>{data_w}}", end="")
        print()
        print(f"{'Background (T_bg_infer)':<{col_w}}", end="")
        for r in pl_results: print(f"{'[%.1f ms]' % r.avg_bg_infer:>{data_w}}", end="")
        print()
        print(f"{'Per-Face Crop+Infer':<{col_w}}", end="")
        for r in pl_results:
            if r.n_faces == 0: print(f"{'[–]':>{data_w}}", end="")
            else: print(f"{'[%.1f ms]' % r.avg_face_infer:>{data_w}}", end="")
        print()
        print(f"{'Blending (T_blend)':<{col_w}}", end="")
        for r in pl_results:
            if r.n_faces == 0: print(f"{'[–]':>{data_w}}", end="")
            else: print(f"{'[%.1f ms]' % r.avg_blend:>{data_w}}", end="")
        print()
        print("─" * (col_w + data_w * len(pl_results)))
        print(f"{'Total End-to-End Latency':<{col_w}}", end="")
        for r in pl_results: print(f"{'[%.1f ms]' % r.avg_total:>{data_w}}", end="")
        print()
        print(f"{'Throughput (FPS)':<{col_w}}", end="")
        for r in pl_results: print(f"{'[%.1f FPS]' % r.fps:>{data_w}}", end="")
        print()
        print("=" * (col_w + data_w * len(pl_results)))
        print()

def export_csv(results: list[BenchmarkResult], output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Pipeline", "Label", "N_Faces", "N_Runs",
            "Avg_Detection_ms", "Avg_Background_ms", "Avg_PerFace_ms",
            "Avg_Blending_ms", "Avg_Total_ms", "FPS",
            "Std_Detection_ms", "Std_Background_ms", "Std_PerFace_ms",
            "Std_Blending_ms", "Std_Total_ms",
        ])
        for r in results:
            writer.writerow([
                r.pipeline, r.label, r.n_faces, r.n_runs,
                f"{r.avg_detect:.2f}", f"{r.avg_bg_infer:.2f}", f"{r.avg_face_infer:.2f}",
                f"{r.avg_blend:.2f}", f"{r.avg_total:.2f}", f"{r.fps:.2f}",
                f"{r.std_detect:.2f}", f"{r.std_bg_infer:.2f}", f"{r.std_face_infer:.2f}",
                f"{r.std_blend:.2f}", f"{r.std_total:.2f}",
            ])
    print(f"📄 CSV exported → {output_path}")

def export_latex(results: list[BenchmarkResult], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by benchmark_pipeline.py\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Pipeline Combination} & \\textbf{Stage / Sub-Model} & \\textbf{$N=0$ (No Face)} & \\textbf{$N=1$ (Single Face)} & \\textbf{$N=3$ (Group Photo)} \\\\\n")
        f.write("\\hline\n")
        
        pipeline_labels = {
            "ours": ("\\textbf{Ours (Lightweight Pipeline)}", "(RetinaFace + DTGAN)"),
            "baseline1": ("\\textbf{Pipeline Baseline 1}", "(MTCNN + AnimeGANv2)"),
            "baseline2": ("\\textbf{Pipeline Baseline 2}", "(YOLOv8-Face + CartoonGAN)")
        }
        
        for pl in ["ours", "baseline1", "baseline2"]:
            pl_results = [r for r in results if r.pipeline == pl]
            if not pl_results: continue
            
            def get_col(metric, fmt="[{:.1f} ms]"):
                res = []
                for label in ["N=0", "N=1", "N=3"]:
                    found = next((r for r in pl_results if label in r.label), None)
                    if found:
                        val = getattr(found, metric)
                        if ("blend" in metric or "face_infer" in metric) and found.n_faces == 0:
                            res.append("[--]")
                        else: res.append(fmt.format(val))
                    else: res.append("[--]")
                return res

            l1, l2 = pipeline_labels[pl]
            f.write(f"{l1} & Detection ($T_{{\\text{{detect}}}}$) & " + " & ".join(get_col("avg_detect")) + " \\\\\n")
            f.write(f"{l2} & Background ($T_{{\\text{{bg\\_infer}}}}$) & " + " & ".join(get_col("avg_bg_infer")) + " \\\\\n")
            submodel_face = "Per-Face Crop+Infer" if pl != "ours" else "Per-Face Crop+Infer ($T_{\\text{margin}} + T_{\\text{face\\_infer}}$)"
            f.write(f" & {submodel_face} & " + " & ".join(get_col("avg_face_infer")) + " \\\\\n")
            blend_type = "Feathered Blending ($T_{\\text{blend}}$)" if pl == "ours" else ("Poisson Blending" if pl == "baseline1" else "Alpha Blending")
            f.write(f" & {blend_type} & " + " & ".join(get_col("avg_blend")) + " \\\\\n")
            
            f.write("\\cline{2-5}\n")
            f.write(f" & \\textbf{{Total End-to-End Latency}} & " + " & ".join(get_col("avg_total", "\\textbf{[{:.1f} ms]}")) + " \\\\\n")
            f.write(f" & \\textbf{{Throughput (FPS)}} & " + " & ".join(get_col("fps", "\\textbf{[{:.1f} FPS]}")) + " \\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print(f"📄 LaTeX exported → {output_path}")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def _scan_images(folder: str) -> list[str]:
    if not os.path.isdir(folder): return []
    files = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, f))
    return files

TEST_FOLDERS = [
    ("no_face",      "N=0 (No Face)"),
    ("single_face",  "N=1 (Single Face)"),
    ("group_photo",  "N=3 (Group Photo)"),
]

def resolve_onnx_model_path(model_name: str) -> str:
    """Find ONNX model path with exact or fuzzy matching in ONNX_MODEL_DIR."""
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name

    target_name = model_name if model_name.endswith(".onnx") else f"{model_name}.onnx"
    direct_path = os.path.join(ONNX_MODEL_DIR, target_name)
    if os.path.exists(direct_path):
        return direct_path

    if os.path.isdir(ONNX_MODEL_DIR):
        available_files = [f for f in os.listdir(ONNX_MODEL_DIR) if f.endswith(".onnx")]
        # Case insensitive or substring search
        lower_target = model_name.lower()
        for f in available_files:
            if lower_target in f.lower():
                print(f"💡 Auto-matched model '{model_name}' -> '{f}'")
                return os.path.join(ONNX_MODEL_DIR, f)
        
        print(f"❌ Cannot find model file '{target_name}' in {ONNX_MODEL_DIR}")
        print("   Available ONNX models in directory:")
        for f in available_files:
            print(f"     - {f}")
        sys.exit(1)
    else:
        print(f"❌ Directory not found: {ONNX_MODEL_DIR}")
        sys.exit(1)

def load_models(args):
    models = {}
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch_device == 'cuda' else ['CPUExecutionProvider']
    
    pipelines = ["ours", "baseline1", "baseline2"] if args.pipeline == "all" else [args.pipeline]
    
    if "ours" in pipelines:
        onnx_path = resolve_onnx_model_path(args.model)
        models['ours_onnx_path'] = onnx_path
        models['ours_ort'] = onnxruntime.InferenceSession(onnx_path, sess_options=ort_sess_options, providers=providers)
        print(f"Loaded Ours ONNX: {os.path.basename(onnx_path)}")
        
    if "baseline1" in pipelines:
        from facenet_pytorch import MTCNN
        models['mtcnn'] = MTCNN(keep_all=True, device=torch_device)
        print("Loaded MTCNN (facenet-pytorch)")
        animegan_path = os.path.join(ONNX_MODEL_DIR, "AnimeGANv2_Hayao.onnx")
        if os.path.exists(animegan_path):
            models['animegan'] = onnxruntime.InferenceSession(animegan_path, sess_options=ort_sess_options, providers=providers)
            print("Loaded AnimeGANv2 ONNX")
        else:
            print(f"Warning: {animegan_path} not found. Baseline 1 will fail.")
            
    if "baseline2" in pipelines:
        from ultralytics import YOLO
        models['yolov8'] = YOLO("yolov8n.pt") # YOLOv8 Nano for fast person detection (simulating yolov8n-face)
        if torch_device == 'cuda': models['yolov8'].to('cuda')
        print("Loaded YOLOv8n (ultralytics)")
        cartoon_path = os.path.join(ONNX_MODEL_DIR, "CartoonGAN_Placeholder.onnx")
        if os.path.exists(cartoon_path):
            models['cartoongan'] = onnxruntime.InferenceSession(cartoon_path, sess_options=ort_sess_options, providers=providers)
            print("Loaded CartoonGAN ONNX")
        else:
            print(f"Warning: {cartoon_path} not found. Baseline 2 will fail.")
            
    return models

def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline stages for evaluation table.")
    parser.add_argument("--test-dir", default=None, help="Thư mục gốc chứa 3 folder con")
    parser.add_argument("--model", default="AnimeGANv3_Shinkai_37", help="Tên model ONNX")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Chạy trên CPU hay GPU (default: gpu)")
    parser.add_argument("--pipeline", choices=["ours", "baseline1", "baseline2", "all"], default="all", help="Pipeline cần chạy (default: all)")
    parser.add_argument("--runs", type=int, default=50, help="Số lần chạy (default: 50)")
    parser.add_argument("--warmup", type=int, default=5, help="Số lần warmup (default: 5)")
    parser.add_argument("--csv", default="benchmark_results.csv", help="Đường dẫn file CSV")
    parser.add_argument("--latex", default="benchmark_results.tex", help="Đường dẫn file LaTeX")

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = args.test_dir or os.path.join(script_dir, "test_images")

    if not os.path.isdir(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        sys.exit(1)

    print(f"Loading Models on {torch_device.upper()}...")
    models = load_models(args)
    all_results: list[BenchmarkResult] = []
    pipelines_to_run = ["ours", "baseline1", "baseline2"] if args.pipeline == "all" else [args.pipeline]

    for pipeline in pipelines_to_run:
        print(f"\n{'='*70}\n🚀 RUNNING PIPELINE: {pipeline.upper()}\n{'='*70}\n")
        
        for folder_name, label in TEST_FOLDERS:
            folder_path = os.path.join(test_dir, folder_name)
            image_files = _scan_images(folder_path)

            print(f"\n{'='*70}\n📁 Folder: {folder_path} | Pipeline: {pipeline}")
            print(f"   Label:  {label}\n   Found:  {len(image_files)} image(s)\n{'='*70}")

            if len(image_files) == 0:
                print(f"   ⚠️  Folder trống! Bỏ qua folder này.")
                continue

            folder_results: list[BenchmarkResult] = []
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                print(f"\n  📸 {img_name}")
                img = Image.open(img_path).convert("RGB")
                print(f"     Size: {img.size[0]}×{img.size[1]}")

                try:
                    result = run_benchmark(img, models, pipeline, label=f"{label}", runs=args.runs, warmup=args.warmup)
                    print(f"     → Detected {result.n_faces} face(s)")
                    print(f"     → Avg total: {result.avg_total:.1f} ms ({result.fps:.1f} FPS)")
                    folder_results.append(result)
                except Exception as e:
                    print(f"     ❌ Lỗi khi benchmark ảnh {img_name}: {e}")

            if len(folder_results) > 0:
                if len(folder_results) == 1:
                    avg_result = folder_results[0]
                    avg_result.label = label
                else:
                    pooled_timings = [t for r in folder_results for t in r.raw_timings]
                    detect_arr = np.array([t.t_detect for t in pooled_timings])
                    bg_arr = np.array([t.t_bg_infer for t in pooled_timings])
                    face_arr = np.array([t.t_per_face_crop_infer for t in pooled_timings])
                    blend_arr = np.array([t.t_blend for t in pooled_timings])
                    total_arr = np.array([t.t_total for t in pooled_timings])
                    avg_tot = float(np.mean(total_arr))
                    avg_result = BenchmarkResult(
                        pipeline=pipeline, label=label,
                        n_faces=int(round(np.mean([r.n_faces for r in folder_results]))),
                        n_runs=len(pooled_timings),
                        avg_detect=float(np.mean(detect_arr)), avg_bg_infer=float(np.mean(bg_arr)),
                        avg_face_infer=float(np.mean(face_arr)), avg_blend=float(np.mean(blend_arr)),
                        avg_total=avg_tot, fps=1000.0 / avg_tot if avg_tot > 0 else 0.0,
                        std_detect=float(np.std(detect_arr)), std_bg_infer=float(np.std(bg_arr)),
                        std_face_infer=float(np.std(face_arr)), std_blend=float(np.std(blend_arr)),
                        std_total=float(np.std(total_arr)), raw_timings=pooled_timings,
                    )
                    print(f"\n  📊 Folder average ({len(folder_results)} images, {len(pooled_timings)} total runs):")
                    print(f"     Total: {avg_result.avg_total:.1f} ± {avg_result.std_total:.1f} ms | FPS: {avg_result.fps:.1f}")
                all_results.append(avg_result)

    if len(all_results) == 0:
        print("\n❌ Không có dữ liệu để tổng hợp kết quả.")
        sys.exit(1)

    print_result_table(all_results)
    export_csv(all_results, args.csv)
    export_latex(all_results, args.latex)

if __name__ == "__main__":
    main()
