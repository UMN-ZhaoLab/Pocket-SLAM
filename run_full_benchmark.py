#!/usr/bin/env python3
"""Full-length baseline vs Pocket-SLAM with memory/ATE/FPS tracking."""
import importlib.util
import multiprocessing as mp
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(ROOT, "configs/euroc/full_benchmark.py")
OUT_DIR = os.environ.get(
    "POCKET_SLAM_RESULTS",
    os.path.join(ROOT, "results", "full_benchmark"),
)


def load_base_config():
    spec = importlib.util.spec_from_file_location("full_cfg", CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def worker(gpu_id: int, label: str, pocket_enable: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, ".torch_cache"))
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    os.chdir(ROOT)

    import io
    import numpy as np
    import torch
    from contextlib import redirect_stdout, redirect_stderr
    from utils.common_utils import seed_everything
    from scripts.loop_closure import rgbd_slam

    config = load_base_config()
    config = dict(config)
    config["primary_device"] = "cuda:0"
    config["pocket_slam"] = dict(config["pocket_slam"])
    config["pocket_slam"]["enable"] = pocket_enable
    config["run_name"] = f"{config['run_name']}_{label}"
    config["workdir"] = f"euroc_full_{label}"

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f"{label}.log")
    metrics_path = os.path.join(OUT_DIR, f"{label}_metrics.txt")

    seed_everything(seed=config["seed"])
    os.makedirs(os.path.join(config["workdir"], config["run_name"]), exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    buf = io.StringIO()
    with open(log_path, "w", encoding="utf-8") as logf:
        with redirect_stdout(buf), redirect_stderr(buf):
            rgbd_slam(config, loop=[0, 0])
        logf.write(buf.getvalue())
    elapsed = time.time() - t0
    text = buf.getvalue()

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    metrics = {
        "label": label,
        "gpu": gpu_id,
        "pocket": pocket_enable,
        "elapsed_s": round(elapsed, 1),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }

    ate_vals = re.findall(r"Final Average ATE RMSE:\s*([\d.]+)\s*cm", text)
    if ate_vals:
        metrics["ate_cm"] = float(ate_vals[-1])

    m = re.search(r"Average Tracking/Frame Time:\s*([\d.]+)\s*s", text)
    if m:
        metrics["tracking_frame_s"] = float(m.group(1))
    m = re.search(r"Average Mapping/Frame Time:\s*([\d.]+)\s*s", text)
    if m:
        metrics["mapping_frame_s"] = float(m.group(1))
    if "tracking_frame_s" in metrics and "mapping_frame_s" in metrics:
        total = metrics["tracking_frame_s"] + metrics["mapping_frame_s"]
        metrics["fps"] = round(1.0 / total, 4) if total > 0 else 0.0

    events = re.findall(r"Pruned (\d+) Gaussians → remaining: (\d+)", text)
    if events:
        before = [int(a) + int(b) for a, b in events]
        after = [int(b) for a, b in events]
        metrics["peak_gaussians"] = max(before + after)
        metrics["final_gaussians_log"] = after[-1]

    npz_path = os.path.join(
        config["workdir"], config["run_name"], "eval_0_0", "params.npz"
    )
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        metrics["final_gaussians"] = int(d["means3D"].shape[0])
        metrics["map_size_mb"] = round(os.path.getsize(npz_path) / 1024 / 1024, 2)

    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(
        f"[GPU{gpu_id}] {label} done in {elapsed/60:.1f}min | "
        f"ATE={metrics.get('ate_cm')}cm FPS={metrics.get('fps')} "
        f"VRAM_peak={peak_vram_gb:.2f}GB Gaussians={metrics.get('final_gaussians')}",
        flush=True,
    )
    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = mp.get_context("spawn")
    jobs = [
        (0, "baseline_lsg_v2", False),
        (1, "pocket_slam_v2", True),
    ]
    with ctx.Pool(2) as pool:
        results = pool.starmap(worker, jobs)

    base, pocket = results
    summary = os.path.join(OUT_DIR, "summary.txt")
    with open(summary, "w") as f:
        f.write("Full V2_01_easy benchmark (0-2200, stride=5, 100 iters)\n\n")
        for r in results:
            f.write(f"=== {r['label']} ===\n")
            for k, v in r.items():
                if k != "label":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("=== COMPARISON (Pocket vs Baseline) ===\n")
        if "ate_cm" in base and "ate_cm" in pocket:
            delta = pocket["ate_cm"] - base["ate_cm"]
            pct = (delta / base["ate_cm"]) * 100 if base["ate_cm"] else 0
            f.write(f"ATE: {base['ate_cm']:.2f} -> {pocket['ate_cm']:.2f} cm ({delta:+.2f} cm, {pct:+.1f}%)\n")
        if "fps" in base and "fps" in pocket and base["fps"] > 0:
            ratio = pocket["fps"] / base["fps"]
            f.write(f"FPS: {base['fps']:.4f} -> {pocket['fps']:.4f} ({ratio:.2f}x, {(ratio-1)*100:+.1f}%)\n")
        if "peak_vram_gb" in base and "peak_vram_gb" in pocket:
            delta = pocket["peak_vram_gb"] - base["peak_vram_gb"]
            pct = (1 - pocket["peak_vram_gb"] / base["peak_vram_gb"]) * 100 if base["peak_vram_gb"] else 0
            f.write(f"Peak VRAM: {base['peak_vram_gb']:.3f} -> {pocket['peak_vram_gb']:.3f} GB ({pct:+.1f}% reduction)\n")
        if "final_gaussians" in base and "final_gaussians" in pocket:
            pct = (1 - pocket["final_gaussians"] / base["final_gaussians"]) * 100
            f.write(f"Final Gaussians: {base['final_gaussians']:,} -> {pocket['final_gaussians']:,} ({pct:.1f}% reduction)\n")
        if "map_size_mb" in base and "map_size_mb" in pocket:
            pct = (1 - pocket["map_size_mb"] / base["map_size_mb"]) * 100
            f.write(f"Map size: {base['map_size_mb']:.1f} -> {pocket['map_size_mb']:.1f} MB ({pct:.1f}% reduction)\n")
        if "peak_gaussians" in pocket and "final_gaussians" in base:
            pct = (1 - pocket["peak_gaussians"] / base["final_gaussians"]) * 100
            f.write(f"Pocket peak gaussians vs baseline final: {pocket['peak_gaussians']:,} vs {base['final_gaussians']:,} ({pct:.1f}% lower)\n")

    print(open(summary).read())


if __name__ == "__main__":
    main()
