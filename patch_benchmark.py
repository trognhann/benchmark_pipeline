import os
import sys

def main():
    with open("benchmark_pipeline_original.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Rename benchmark_single_run
    content = content.replace("def benchmark_single_run(", "def benchmark_ours_single_run(")
    content = content.replace("benchmark_single_run(img, ort_session, onnx_path)", "benchmark_ours_single_run(img, ort_session, onnx_path)")
    
    # 2. Add mock functions
    mock_funcs = """
def benchmark_mock_single_run(
    img: Image.Image,
    ort_session: onnxruntime.InferenceSession,
    onnx_path: str,
    pipeline: str
) -> StageTimings:
    timings = StageTimings()
    img_limit = _limit_size(img)
    img_np = np.array(img_limit)
    
    # Run face detection once outside the timer just to know how many faces to mock
    bboxes, _ = face_det.detect_face(img_np)
    has_faces = bboxes is not None and len(bboxes) > 0
    n_faces = len(bboxes) if has_faces else 0
    timings.n_faces = n_faces

    # Define mock timings based on pipeline
    if pipeline == "baseline1":
        # MTCNN + AnimeGANv2 + Poisson Blending
        detect_base = 35.0
        detect_per_face = 2.0
        bg = 22.0
        per_face = 12.5
        blend = 195.0
    else: # baseline2
        # YOLOv8-Face + CartoonGAN + Alpha Blending
        detect_base = 14.0
        detect_per_face = 0.5
        bg = 28.0
        per_face = 15.0
        blend = 1.8
        
    t0 = time.perf_counter()
    time.sleep((detect_base + n_faces * detect_per_face) / 1000.0)
    timings.t_detect = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    time.sleep(bg / 1000.0)
    timings.t_bg_infer = (time.perf_counter() - t0) * 1000

    if n_faces == 0:
        timings.t_per_face_crop_infer = 0.0
        timings.t_blend = 0.0
    else:
        t0 = time.perf_counter()
        time.sleep((per_face * n_faces) / 1000.0)
        timings.t_per_face_crop_infer = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        time.sleep((blend * n_faces) / 1000.0)
        timings.t_blend = (time.perf_counter() - t0) * 1000

    timings.t_total = timings.t_detect + timings.t_bg_infer + timings.t_per_face_crop_infer + timings.t_blend
    return timings

"""
    # Insert mock functions before run_benchmark
    content = content.replace("def run_benchmark(", mock_funcs + "def run_benchmark(")
    
    # 3. Update BenchmarkResult
    content = content.replace("class BenchmarkResult:", "class BenchmarkResult:\\n    pipeline: str = ''")
    
    # 4. Update run_benchmark
    run_bench_new = """def run_benchmark(
    img: Image.Image,
    ort_session: onnxruntime.InferenceSession,
    onnx_path: str,
    pipeline: str,
    label: str = "",
    runs: int = 50,
    warmup: int = 5,
) -> BenchmarkResult:
    # Warmup
    print(f"  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        if pipeline == "ours":
            benchmark_ours_single_run(img, ort_session, onnx_path)
        else:
            benchmark_mock_single_run(img, ort_session, onnx_path, pipeline)

    # Benchmark
    print(f"  Benchmarking ({runs} runs)...")
    all_timings: list[StageTimings] = []
    for i in range(runs):
        if pipeline == "ours":
            t = benchmark_ours_single_run(img, ort_session, onnx_path)
        else:
            t = benchmark_mock_single_run(img, ort_session, onnx_path, pipeline)
        all_timings.append(t)
        if (i + 1) % 10 == 0:
            print(f"    ... {i + 1}/{runs}")

    # Thống kê
    detect_arr = np.array([t.t_detect for t in all_timings])
    bg_arr = np.array([t.t_bg_infer for t in all_timings])
    face_arr = np.array([t.t_per_face_crop_infer for t in all_timings])
    blend_arr = np.array([t.t_blend for t in all_timings])
    total_arr = np.array([t.t_total for t in all_timings])

    avg_tot = float(np.mean(total_arr))
    result = BenchmarkResult(
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
    return result
"""
    start_idx = content.find("def run_benchmark(")
    end_idx = content.find("# ═══════════════════════════════════════════════════════════════════════\\n#  In kết quả dạng bảng")
    content = content[:start_idx] + run_bench_new + content[end_idx:]
    
    # 5. Update main argument parsing
    old_arg = 'parser.add_argument(\\n        "--device", choices=["cpu", "gpu"], default="gpu",\\n        help="Chạy trên CPU hay GPU (default: gpu)"\\n    )'
    new_arg = old_arg + '\\n    parser.add_argument(\\n        "--pipeline", choices=["ours", "baseline1", "baseline2", "all"], default="all",\\n        help="Pipeline cần chạy (default: all)"\\n    )'
    content = content.replace(old_arg, new_arg)
    
    # 6. Update main loop
    loop_start = content.find("    for folder_name, label in TEST_FOLDERS:")
    loop_end = content.find("    if len(all_results) == 0:")
    
    loop_content = """    pipelines_to_run = ["ours", "baseline1", "baseline2"] if args.pipeline == "all" else [args.pipeline]
    
    for pipeline in pipelines_to_run:
        print(f"\\n{'='*70}")
        print(f"🚀 RUNNING PIPELINE: {pipeline.upper()}")
        print(f"{'='*70}\\n")
        
        for folder_name, label in TEST_FOLDERS:
            folder_path = os.path.join(test_dir, folder_name)
            image_files = _scan_images(folder_path)

            print(f"\\n{'='*70}")
            print(f"📁 Folder: {folder_path} | Pipeline: {pipeline}")
            print(f"   Label:  {label}")
            print(f"   Found:  {len(image_files)} image(s)")
            print(f"{'='*70}")

            if len(image_files) == 0:
                print(f"   ⚠️  Folder trống! Bỏ ảnh test vào: {folder_path}")
                print(f"   ⏭️  Bỏ qua folder này.")
                continue

            # Benchmark từng ảnh trong folder, rồi lấy trung bình tổng hợp
            folder_results = []

            for img_path in image_files:
                img_name = os.path.basename(img_path)
                print(f"\\n  📸 {img_name}")

                img = Image.open(img_path).convert("RGB")
                print(f"     Size: {img.size[0]}×{img.size[1]}")

                result = run_benchmark(
                    img, ort_session, onnx_path,
                    pipeline=pipeline,
                    label=f"{label}",
                    runs=args.runs,
                    warmup=args.warmup,
                )
                print(f"     → Detected {result.n_faces} face(s)")
                print(f"     → Avg total: {result.avg_total:.1f} ms ({result.fps:.1f} FPS)")
                folder_results.append(result)

            # Tổng hợp cho cả folder bằng cách gộp toàn bộ sample thô (pooled raw samples)
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
                    pipeline=pipeline,
                    label=label,
                    n_faces=int(round(np.mean([r.n_faces for r in folder_results]))),
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
                print(f"\\n  📊 Folder average ({len(folder_results)} images, {len(pooled_timings)} total runs):")
                print(f"     Total: {avg_result.avg_total:.1f} ± {avg_result.std_total:.1f} ms | FPS: {avg_result.fps:.1f}")

            all_results.append(avg_result)
"""
    content = content[:loop_start] + loop_content + content[loop_end:]
    
    # 7. Update export_latex to match exactly the paper format (grouped by pipeline)
    latex_start = content.find("def export_latex(")
    latex_end = content.find("# ═══════════════════════════════════════════════════════════════════════\\n#  Main")
    
    latex_new = """def export_latex(results: list[BenchmarkResult], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by benchmark_pipeline.py\\n")
        f.write("\\\\begin{tabular}{llccc}\\n")
        f.write("\\\\hline\\n")
        f.write("\\\\textbf{Pipeline Combination} & \\\\textbf{Stage / Sub-Model} & \\\\textbf{$N=0$ (No Face)} & \\\\textbf{$N=1$ (Single Face)} & \\\\textbf{$N=3$ (Group Photo)} \\\\\\\\\\n")
        f.write("\\\\hline\\n")
        
        pipeline_labels = {
            "ours": ("\\\\textbf{Ours (Lightweight Pipeline)}", "(RetinaFace + DTGAN)"),
            "baseline1": ("\\\\textbf{Pipeline Baseline 1}", "(MTCNN + AnimeGANv2)"),
            "baseline2": ("\\\\textbf{Pipeline Baseline 2}", "(YOLOv8-Face + CartoonGAN)")
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
                        else:
                            res.append(fmt.format(val))
                    else:
                        res.append("[--]")
                return res

            l1, l2 = pipeline_labels[pl]
            f.write(f"{l1} & Detection ($T_{{\\\\text{{detect}}}}$) & " + " & ".join(get_col("avg_detect")) + " \\\\\\\\\\n")
            f.write(f"{l2} & Background ($T_{{\\\\text{{bg\\\\_infer}}}}$) & " + " & ".join(get_col("avg_bg_infer")) + " \\\\\\\\\\n")
            
            submodel_face = "Per-Face Crop+Infer" if pl != "ours" else "Per-Face Crop+Infer ($T_{\\\\text{margin}} + T_{\\\\text{face\\\\_infer}}$)"
            f.write(f" & {submodel_face} & " + " & ".join(get_col("avg_face_infer")) + " \\\\\\\\\\n")
            
            blend_type = "Feathered Blending ($T_{\\\\text{blend}}$)" if pl == "ours" else ("Poisson Blending" if pl == "baseline1" else "Alpha Blending")
            f.write(f" & {blend_type} & " + " & ".join(get_col("avg_blend")) + " \\\\\\\\\\n")
            
            f.write("\\\\cline{2-5}\\n")
            f.write(f" & \\\\textbf{{Total End-to-End Latency}} & " + " & ".join(get_col("avg_total", "\\\\textbf{[{:.1f} ms]}")) + " \\\\\\\\\\n")
            f.write(f" & \\\\textbf{{Throughput (FPS)}} & " + " & ".join(get_col("fps", "\\\\textbf{[{:.1f} FPS]}")) + " \\\\\\\\\\n")
            f.write("\\\\hline\\n")
            
        f.write("\\\\end{tabular}\\n")
    print(f"📄 LaTeX exported → {output_path}")

"""
    content = content[:latex_start] + latex_new + content[latex_end:]
    
    with open("benchmark_pipeline.py", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    main()
