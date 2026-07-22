"""
Pipeline Benchmark – Tách từng stage để đo latency cho bảng đánh giá.

Bảng mục tiêu (Ours – Lightweight Pipeline: RetinaFace + DTGAN):
┌──────────────────────────┬──────────────┬──────────────┬──────────────────┐
│ Stage / Sub‑Model        │ N=0 (No Face)│ N=1 (Single) │ N=3 (Group Photo)│
├──────────────────────────┼──────────────┼──────────────┼──────────────────┤
│ Detection (T_detect)     │              │              │                  │
│ Background (T_bg_infer)  │              │              │                  │
│ Per-Face Crop+Infer      │ [–]          │              │                  │
│ Feathered Blending       │ [–]          │              │                  │
├──────────────────────────┼──────────────┼──────────────┼──────────────────┤
│ Total End-to-End Latency │              │              │                  │
│ Throughput (FPS)         │              │              │                  │
└──────────────────────────┴──────────────┴──────────────┴──────────────────┘

Cấu trúc ảnh đầu vào:
    test_images/
    ├── no_face/        (dành cho N=0)
    ├── single_face/    (dành cho N=1)
    └── group_photo/    (dành cho N=3)

Cách dùng:
    python benchmark_pipeline.py [--test-dir test_images] [--model hayao] [--runs 50] [--warmup 5]

Kết quả sẽ in ra bảng console + xuất CSV (benchmark_results.csv) và LaTeX (benchmark_results.tex).
"""

import os
import sys
import time
import argparse
import csv
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

# ─── Import các module từ project ──────────────────────────────────────
import face_det

import onnxruntime

ort_sess_options = onnxruntime.SessionOptions()
ort_sess_options.intra_op_num_threads = int(
    os.environ.get("ORT_INTRA_OP_NUM_THREADS", 0))


# ─── Constants ─────────────────────────────────────────────────────────
MAX_EDGE = 1280
ONNX_MODEL_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "onnx_model")


# ─── Helpers (giữ nguyên logic từ fast-api.py) ─────────────────────────

def _limit_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_edge = max(w, h)
    if max_edge > MAX_EDGE:
        scale = MAX_EDGE / max_edge
        img = img.resize(
            (int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
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


# ═══════════════════════════════════════════════════════════════════════
#  Dataclass lưu kết quả benchmark cho 1 lần chạy
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StageTimings:
    """Thời gian (ms) của từng stage trong 1 lần chạy."""
    t_detect: float = 0.0           # Detection (T_detect)
    t_bg_infer: float = 0.0        # Background inference (T_bg_infer)
    # Per-Face Crop + Infer (T_margin + T_face_infer) × N
    t_per_face_crop_infer: float = 0.0
    t_blend: float = 0.0           # Feathered Blending (T_blend)
    t_total: float = 0.0           # Total End-to-End
    n_faces: int = 0               # Số khuôn mặt phát hiện được


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline benchmark – tách từng stage
# ═══════════════════════════════════════════════════════════════════════

def benchmark_single_run(
    img: Image.Image,
    ort_session: onnxruntime.InferenceSession,
    onnx_path: str,
) -> StageTimings:
    """
    Chạy full hybrid pipeline 1 lần, đo từng stage riêng biệt.

    Các stage:
    1. Detection (T_detect)       – RetinaFace detect
    2. Background (T_bg_infer)    – ONNX inference toàn bộ ảnh (landscape mode)
    3. Per-Face Crop+Infer        – Crop margin + ONNX inference cho mỗi khuôn mặt
    4. Feathered Blending         – Tạo mask + paste cho mỗi khuôn mặt
    """
    timings = StageTimings()
    original_size = img.size
    img_limit = _limit_size(img)
    img_np = np.array(img_limit)

    # ────── Stage 1: Detection (T_detect) ──────
    t0 = time.perf_counter()
    bboxes, points = face_det.detect_face(img_np)
    timings.t_detect = (time.perf_counter() - t0) * 1000

    has_faces = bboxes is not None and len(bboxes) > 0
    timings.n_faces = len(bboxes) if has_faces else 0

    # ────── Stage 2: Background Inference (T_bg_infer) ──────
    t0 = time.perf_counter()
    inp_bg = _preprocess(img_limit, False, onnx_path)
    out_bg = ort_session.run(
        None, {ort_session.get_inputs()[0].name: inp_bg})[0]
    bg_result_img = _postprocess(out_bg, original_size, False)
    timings.t_bg_infer = (time.perf_counter() - t0) * 1000

    if not has_faces:
        # N=0: không có khuôn mặt → Per-Face & Blending = N/A
        timings.t_per_face_crop_infer = 0.0
        timings.t_blend = 0.0
        timings.t_total = timings.t_detect + timings.t_bg_infer
        return timings

    # ────── Stage 3 & 4: Per-Face (cho mỗi khuôn mặt) ──────
    w, h = original_size
    max_edge = max(w, h)
    scale = 1.0
    if max_edge > MAX_EDGE:
        scale = MAX_EDGE / max_edge

    total_face_infer_ms = 0.0
    total_blend_ms = 0.0

    for box in bboxes:
        # ── Stage 3: Per-Face Crop + Infer ──
        t0 = time.perf_counter()

        margin_box = face_det.margin_face(box, img_np.shape[:2])
        x1, y1, x2, y2 = margin_box

        face_np = img_np[y1:y2, x1:x2]
        if face_np.size == 0:
            continue
        face_img = Image.fromarray(face_np)

        inp_face = _preprocess(face_img, True, onnx_path)
        out_face = ort_session.run(
            None, {ort_session.get_inputs()[0].name: inp_face})[0]
        face_result_img = _postprocess(out_face, original_size, True)

        total_face_infer_ms += (time.perf_counter() - t0) * 1000

        # ── Stage 4: Feathered Blending ──
        t0 = time.perf_counter()

        orig_x1 = int(round(x1 / scale))
        orig_y1 = int(round(y1 / scale))
        orig_x2 = int(round(x2 / scale))
        orig_y2 = int(round(y2 / scale))

        face_w = max(1, orig_x2 - orig_x1)
        face_h = max(1, orig_y2 - orig_y1)

        face_result_resized = face_result_img.resize(
            (face_w, face_h), Image.LANCZOS)

        mask = Image.new("L", (face_w, face_h), 0)
        draw = ImageDraw.Draw(mask)
        margin = int(min(face_w, face_h) * 0.08)
        if margin <= 0:
            margin = 1
        draw.rectangle([margin, margin, face_w - margin,
                       face_h - margin], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=margin))

        bg_result_img.paste(face_result_resized, (orig_x1, orig_y1), mask)

        total_blend_ms += (time.perf_counter() - t0) * 1000

    timings.t_per_face_crop_infer = total_face_infer_ms
    timings.t_blend = total_blend_ms
    timings.t_total = (
        timings.t_detect
        + timings.t_bg_infer
        + timings.t_per_face_crop_infer
        + timings.t_blend
    )

    return timings


# ═══════════════════════════════════════════════════════════════════════
#  Chạy benchmark nhiều lần + thống kê
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Kết quả trung bình và độ lệch chuẩn sau nhiều lần chạy."""
    label: str = ""
    n_faces: int = 0
    n_runs: int = 0

    # Trung bình (ms)
    avg_detect: float = 0.0
    avg_bg_infer: float = 0.0
    avg_face_infer: float = 0.0
    avg_blend: float = 0.0
    avg_total: float = 0.0
    fps: float = 0.0

    # Std (ms) – tính trên tập mẫu gộp (pooled raw samples)
    std_detect: float = 0.0
    std_bg_infer: float = 0.0
    std_face_infer: float = 0.0
    std_blend: float = 0.0
    std_total: float = 0.0

    # Lưu mảng timing thô để hỗ trợ gộp tập dữ liệu đúng phương sai
    raw_timings: list[StageTimings] = field(default_factory=list)


def run_benchmark(
    img: Image.Image,
    ort_session: onnxruntime.InferenceSession,
    onnx_path: str,
    label: str = "",
    runs: int = 50,
    warmup: int = 5,
) -> BenchmarkResult:
    """Chạy benchmark nhiều lần, trả về kết quả trung bình và mảng sample thô."""

    # Warmup
    print(f"  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        benchmark_single_run(img, ort_session, onnx_path)

    # Benchmark
    print(f"  Benchmarking ({runs} runs)...")
    all_timings: list[StageTimings] = []
    for i in range(runs):
        t = benchmark_single_run(img, ort_session, onnx_path)
        all_timings.append(t)
        if (i + 1) % 10 == 0:
            print(f"    ... {i + 1}/{runs}")

    # Thống kê
    detect_arr = np.array([t.t_detect for t in all_timings])
    bg_arr = np.array([t.t_bg_infer for t in all_timings])
    face_arr = np.array([t.t_per_face_crop_infer for t in all_timings])
    blend_arr = np.array([t.t_blend for t in all_timings])
    total_arr = np.array([t.t_total for t in all_timings])

    result = BenchmarkResult(
        label=label,
        n_faces=all_timings[0].n_faces,
        n_runs=runs,
        avg_detect=float(np.mean(detect_arr)),
        avg_bg_infer=float(np.mean(bg_arr)),
        avg_face_infer=float(np.mean(face_arr)),
        avg_blend=float(np.mean(blend_arr)),
        avg_total=float(np.mean(total_arr)),
        fps=1000.0 / float(np.mean(total_arr)
                           ) if np.mean(total_arr) > 0 else 0.0,
        std_detect=float(np.std(detect_arr)),
        std_bg_infer=float(np.std(bg_arr)),
        std_face_infer=float(np.std(face_arr)),
        std_blend=float(np.std(blend_arr)),
        std_total=float(np.std(total_arr)),
        raw_timings=all_timings,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  In kết quả dạng bảng
# ═══════════════════════════════════════════════════════════════════════

def print_result_table(results: list[BenchmarkResult]):
    """In bảng kết quả giống format trong paper."""
    print()
    print("=" * 100)
    print("  BENCHMARK RESULTS – Ours (Lightweight Pipeline): RetinaFace + DTGAN")
    print("=" * 100)

    # Header
    headers = ["Stage / Sub-Model"]
    for r in results:
        headers.append(r.label)

    col_w = 28
    data_w = 22

    # Print header row
    print(f"{'Stage / Sub-Model':<{col_w}}", end="")
    for r in results:
        print(f"{r.label:>{data_w}}", end="")
    print()
    print("─" * (col_w + data_w * len(results)))

    # Detection
    print(f"{'Detection (T_detect)':<{col_w}}", end="")
    for r in results:
        print(f"{'[%.1f ms]' % r.avg_detect:>{data_w}}", end="")
    print()

    # Background
    print(f"{'Background (T_bg_infer)':<{col_w}}", end="")
    for r in results:
        print(f"{'[%.1f ms]' % r.avg_bg_infer:>{data_w}}", end="")
    print()

    # Per-Face Crop+Infer
    print(f"{'Per-Face Crop+Infer':<{col_w}}", end="")
    for r in results:
        if r.n_faces == 0:
            print(f"{'[–]':>{data_w}}", end="")
        else:
            print(f"{'[%.1f ms]' % r.avg_face_infer:>{data_w}}", end="")
    print()

    # Feathered Blending
    print(f"{'Feathered Blending (T_blend)':<{col_w}}", end="")
    for r in results:
        if r.n_faces == 0:
            print(f"{'[–]':>{data_w}}", end="")
        else:
            print(f"{'[%.1f ms]' % r.avg_blend:>{data_w}}", end="")
    print()

    # Separator
    print("─" * (col_w + data_w * len(results)))

    # Total
    print(f"{'Total End-to-End Latency':<{col_w}}", end="")
    for r in results:
        print(f"{'[%.1f ms]' % r.avg_total:>{data_w}}", end="")
    print()

    # Throughput
    print(f"{'Throughput (FPS)':<{col_w}}", end="")
    for r in results:
        print(f"{'[%.1f FPS]' % r.fps:>{data_w}}", end="")
    print()

    print("=" * (col_w + data_w * len(results)))
    print()

    # Chi tiết std
    print("── Standard Deviation (σ) ──")
    print(f"{'Stage':<{col_w}}", end="")
    for r in results:
        print(f"{r.label:>{data_w}}", end="")
    print()
    print(f"{'σ Detection':<{col_w}}", end="")
    for r in results:
        print(f"{'±%.2f ms' % r.std_detect:>{data_w}}", end="")
    print()
    print(f"{'σ Background':<{col_w}}", end="")
    for r in results:
        print(f"{'±%.2f ms' % r.std_bg_infer:>{data_w}}", end="")
    print()
    print(f"{'σ Per-Face':<{col_w}}", end="")
    for r in results:
        if r.n_faces == 0:
            print(f"{'–':>{data_w}}", end="")
        else:
            print(f"{'±%.2f ms' % r.std_face_infer:>{data_w}}", end="")
    print()
    print(f"{'σ Blending':<{col_w}}", end="")
    for r in results:
        if r.n_faces == 0:
            print(f"{'–':>{data_w}}", end="")
        else:
            print(f"{'±%.2f ms' % r.std_blend:>{data_w}}", end="")
    print()
    print(f"{'σ Total':<{col_w}}", end="")
    for r in results:
        print(f"{'±%.2f ms' % r.std_total:>{data_w}}", end="")
    print()
    print()


def export_csv(results: list[BenchmarkResult], output_path: str):
    """Xuất kết quả ra file CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Label", "N_Faces", "N_Runs",
            "Avg_Detection_ms", "Avg_Background_ms", "Avg_PerFace_ms",
            "Avg_Blending_ms", "Avg_Total_ms", "FPS",
            "Std_Detection_ms", "Std_Background_ms", "Std_PerFace_ms",
            "Std_Blending_ms", "Std_Total_ms",
        ])
        for r in results:
            writer.writerow([
                r.label, r.n_faces, r.n_runs,
                f"{r.avg_detect:.2f}", f"{r.avg_bg_infer:.2f}", f"{r.avg_face_infer:.2f}",
                f"{r.avg_blend:.2f}", f"{r.avg_total:.2f}", f"{r.fps:.2f}",
                f"{r.std_detect:.2f}", f"{r.std_bg_infer:.2f}", f"{r.std_face_infer:.2f}",
                f"{r.std_blend:.2f}", f"{r.std_total:.2f}",
            ])
    print(f"📄 CSV exported → {output_path}")


def export_latex(results: list[BenchmarkResult], output_path: str):
    """Xuất kết quả ra LaTeX table (copy-paste vào paper)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by benchmark_pipeline.py\n")
        f.write("\\begin{tabular}{l" + "c" * len(results) + "}\n")
        f.write("\\hline\n")

        # Header
        f.write("\\textbf{Stage / Sub-Model}")
        for r in results:
            f.write(f" & \\textbf{{{r.label}}}")
        f.write(" \\\\\n\\hline\n")

        # Detection
        f.write("Detection ($T_{\\text{detect}}$)")
        for r in results:
            f.write(f" & [{r.avg_detect:.1f} ms]")
        f.write(" \\\\\n")

        # Background
        f.write("Background ($T_{\\text{bg\\_infer}}$)")
        for r in results:
            f.write(f" & [{r.avg_bg_infer:.1f} ms]")
        f.write(" \\\\\n")

        # Per-Face
        f.write(
            "Per-Face Crop+Infer ($T_{\\text{margin}} + T_{\\text{face\\_infer}}$)")
        for r in results:
            if r.n_faces == 0:
                f.write(" & [--]")
            else:
                f.write(f" & [{r.avg_face_infer:.1f} ms]")
        f.write(" \\\\\n")

        # Blending
        f.write("Feathered Blending ($T_{\\text{blend}}$)")
        for r in results:
            if r.n_faces == 0:
                f.write(" & [--]")
            else:
                f.write(f" & [{r.avg_blend:.1f} ms]")
        f.write(" \\\\\n\\hline\n")

        # Total
        f.write("\\textbf{Total End-to-End Latency}")
        for r in results:
            f.write(f" & \\textbf{{[{r.avg_total:.1f} ms]}}")
        f.write(" \\\\\n")

        # FPS
        f.write("\\textbf{Throughput (FPS)}")
        for r in results:
            f.write(f" & \\textbf{{[{r.fps:.1f} FPS]}}")
        f.write(" \\\\\n\\hline\n")

        f.write("\\end{tabular}\n")

    print(f"📄 LaTeX exported → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _scan_images(folder: str) -> list[str]:
    """Quét tất cả file ảnh trong folder, trả về danh sách path đã sort."""
    if not os.path.isdir(folder):
        return []
    files = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, f))
    return files


# 3 folder cố định tương ứng 3 cột trong bảng đánh giá
TEST_FOLDERS = [
    ("no_face",      "N=0 (No Face)"),
    ("single_face",  "N=1 (Single Face)"),
    ("group_photo",  "N=3 (Group Photo)"),
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark pipeline stages for evaluation table.\n"
            "Tự động quét ảnh từ 3 folder trong test_images/:\n"
            "  test_images/no_face/       → N=0 (No Face)\n"
            "  test_images/single_face/   → N=1 (Single Face)\n"
            "  test_images/group_photo/   → N=3 (Group Photo)\n"
            "\nBỏ ảnh test vào folder tương ứng rồi chạy script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-dir", default=None,
        help="Thư mục gốc chứa 3 folder con (default: test_images/ trong cùng thư mục script)"
    )
    parser.add_argument(
        "--model", default="hayao",
        help="Tên model ONNX (default: hayao)"
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Số lần chạy benchmark mỗi ảnh (default: 50)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Số lần warmup mỗi ảnh (default: 5)"
    )
    parser.add_argument(
        "--csv", default="benchmark_results.csv",
        help="Đường dẫn file CSV output (default: benchmark_results.csv)"
    )
    parser.add_argument(
        "--latex", default="benchmark_results.tex",
        help="Đường dẫn file LaTeX output (default: benchmark_results.tex)"
    )

    args = parser.parse_args()

    # Xác định thư mục test_images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = args.test_dir or os.path.join(script_dir, "test_images")

    if not os.path.isdir(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        print(f"   Hãy tạo folder test_images/ và 3 folder con: no_face/, single_face/, group_photo/")
        sys.exit(1)

    # Load model
    onnx_filename = args.model if args.model.endswith(
        ".onnx") else f"{args.model}.onnx"
    onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_filename)

    if not os.path.isfile(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        sys.exit(1)

    print(f"Loading ONNX model: {onnx_path}")
    ort_session = onnxruntime.InferenceSession(
        onnx_path, sess_options=ort_sess_options)

    # Quét 3 folder và chạy benchmark
    all_results: list[BenchmarkResult] = []

    for folder_name, label in TEST_FOLDERS:
        folder_path = os.path.join(test_dir, folder_name)
        image_files = _scan_images(folder_path)

        print(f"\n{'='*70}")
        print(f"📁 Folder: {folder_path}")
        print(f"   Label:  {label}")
        print(f"   Found:  {len(image_files)} image(s)")
        print(f"{'='*70}")

        if len(image_files) == 0:
            print(f"   ⚠️  Folder trống! Bỏ ảnh test vào: {folder_path}")
            print(f"   ⏭️  Bỏ qua folder này.")
            continue

        # Benchmark từng ảnh trong folder, rồi lấy trung bình tổng hợp
        folder_results: list[BenchmarkResult] = []

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            print(f"\n  📸 {img_name}")

            img = Image.open(img_path).convert("RGB")
            print(f"     Size: {img.size[0]}×{img.size[1]}")

            result = run_benchmark(
                img, ort_session, onnx_path,
                label=f"{label} – {img_name}",
                runs=args.runs,
                warmup=args.warmup,
            )
            print(f"     → Detected {result.n_faces} face(s)")
            print(
                f"     → Avg total: {result.avg_total:.1f} ms ({result.fps:.1f} FPS)")
            folder_results.append(result)

        # Tổng hợp cho cả folder bằng cách gộp toàn bộ sample thô (pooled raw samples)
        if len(folder_results) == 1:
            avg_result = folder_results[0]
            avg_result.label = label
        else:
            pooled_timings = [t for r in folder_results for t in r.raw_timings]
            detect_arr = np.array([t.t_detect for t in pooled_timings])
            bg_arr = np.array([t.t_bg_infer for t in pooled_timings])
            face_arr = np.array(
                [t.t_per_face_crop_infer for t in pooled_timings])
            blend_arr = np.array([t.t_blend for t in pooled_timings])
            total_arr = np.array([t.t_total for t in pooled_timings])

            avg_tot = float(np.mean(total_arr))
            avg_result = BenchmarkResult(
                label=label,
                n_faces=int(
                    round(np.mean([r.n_faces for r in folder_results]))),
                n_runs=len(pooled_timings),
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
                raw_timings=pooled_timings,
            )
            print(
                f"\n  📊 Folder average ({len(folder_results)} images, {len(pooled_timings)} total runs):")
            print(
                f"     Total: {avg_result.avg_total:.1f} ± {avg_result.std_total:.1f} ms | FPS: {avg_result.fps:.1f}")

        all_results.append(avg_result)

    if len(all_results) == 0:
        print("\n❌ Không có folder nào có ảnh! Hãy bỏ ảnh vào:")
        for folder_name, label in TEST_FOLDERS:
            print(f"   📁 {os.path.join(test_dir, folder_name)}/  ← {label}")
        sys.exit(1)

    # Output
    print_result_table(all_results)
    export_csv(all_results, args.csv)
    export_latex(all_results, args.latex)


if __name__ == "__main__":
    main()
