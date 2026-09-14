#!/usr/bin/env python3
"""Run paper-aligned baseline vs Pocket-SLAM on the V2 validation subset."""
import importlib.util
import io
import multiprocessing as mp
import os
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(ROOT, "configs/euroc/paper_val_v2.py")
OUT_DIR = os.environ.get(
    "POCKET_SLAM_RESULTS",
    os.path.join("/mnt/hdd/lls/pocket_slam_results", "paper_val_v2"),
)


def load_base_config():
    spec = importlib.util.spec_from_file_location("paper_val_cfg", CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def worker(gpu_id: int, label: str, pocket_enable: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    os.chdir(ROOT)

    import numpy as np
    import torch
    from utils.common_utils import seed_everything
    from scripts.loop_closure import rgbd_slam

    config = dict(load_base_config())
    config["primary_device"] = "cuda:0"
    config["pocket_slam"] = dict(config["pocket_slam"])
    config["pocket_slam"]["enable"] = pocket_enable
    config["run_name"] = f"{config['run_name']}_{label}"
    config["workdir"] = f"euroc_paper_val_{label}"

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f"{label}.log")

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

    metrics = {
        "label": label,
        "gpu": gpu_id,
        "pocket": pocket_enable,
        "elapsed_s": round(elapsed, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
    }
    ate = re.findall(r"Final Average ATE RMSE:\s*([\d.]+)\s*cm", text)
    if ate:
        metrics["ate_cm"] = float(ate[-1])
        metrics["ate_m"] = float(ate[-1]) / 100.0
    psnr = re.findall(r"Average PSNR:\s*([\d.]+)", text)
    if not psnr:
        psnr = re.findall(r"PSNR[:\s]+([\d.]+)", text)
    if psnr:
        metrics["psnr"] = float(psnr[-1])
    m = re.search(r"Average Tracking/Frame Time:\s*([\d.]+)\s*s", text)
    if m:
        metrics["tracking_frame_s"] = float(m.group(1))
    m = re.search(r"Average Mapping/Frame Time:\s*([\d.]+)\s*s", text)
    if m:
        metrics["mapping_frame_s"] = float(m.group(1))
    if "tracking_frame_s" in metrics and "mapping_frame_s" in metrics:
        total = metrics["tracking_frame_s"] + metrics["mapping_frame_s"]
        metrics["fps"] = round(1.0 / total, 4) if total > 0 else 0.0

    npz = os.path.join(config["workdir"], config["run_name"], "eval_0_0", "params.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        metrics["final_gaussians"] = int(d["means3D"].shape[0])
        metrics["map_size_mb"] = round(os.path.getsize(npz) / 1024 / 1024, 2)

    with open(os.path.join(OUT_DIR, f"{label}_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"[GPU{gpu_id}] {label} done | {metrics}", flush=True)
    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = mp.get_context("spawn")
    jobs = [
        (0, "baseline_lsg", False),
        (1, "pocket_slam", True),
    ]
    with ctx.Pool(2) as pool:
        results = pool.starmap(worker, jobs)

    summary = os.path.join(OUT_DIR, "summary.txt")
    with open(summary, "w") as f:
        f.write("Paper-aligned V2_01_easy_val (IGEV depth, 0-275 stride=5)\n\n")
        for r in results:
            f.write(f"=== {r['label']} ===\n")
            for k, v in r.items():
                if k != "label":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
        if len(results) == 2:
            b, p = results
            f.write("=== COMPARISON ===\n")
            if "ate_m" in b and "ate_m" in p:
                f.write(f"ATE(m): {b['ate_m']:.3f} -> {p['ate_m']:.3f}\n")
            if "psnr" in b and "psnr" in p:
                f.write(f"PSNR: {b['psnr']:.2f} -> {p['psnr']:.2f}\n")
            if "fps" in b and "fps" in p and b["fps"]:
                f.write(f"FPS: {b['fps']:.3f} -> {p['fps']:.3f} ({p['fps']/b['fps']:.2f}x)\n")
            if "peak_vram_gb" in b and "peak_vram_gb" in p:
                f.write(f"Peak VRAM(GB): {b['peak_vram_gb']:.2f} -> {p['peak_vram_gb']:.2f}\n")
    print(open(summary).read())


if __name__ == "__main__":
    main()
