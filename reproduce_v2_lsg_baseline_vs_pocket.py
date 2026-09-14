#!/usr/bin/env python3
"""One-click reproduce: LSG Baseline vs Pocket-SLAM on EuRoC V2_01_easy.

This runs the full LSG pipeline used for our validated numbers:

  chunked odometry (step=200, stride=5)
  → loop_closure
  → pose-graph
  → structure refine (map refine)

Expected final metrics (1× GPU, seed=0; small run-to-run variance OK):

                ATE(loop)   PSNR     MS-SSIM   LPIPS
  LSG Baseline    6.64 cm   32.50    0.984     0.037
  Pocket-SLAM     6.38 cm   32.04    0.982     0.042

Prereqs:
  - euroc/V2_01_easy prepared (data_rect, traj.txt, global_features)
  - IGEV depth_sceneflow/ (auto-generated if missing and weights exist)
  - TransVPR / SuperPoint / LightGlue / IGEV sceneflow.pth weights

Usage:
  python reproduce_v2_lsg_baseline_vs_pocket.py
  python reproduce_v2_lsg_baseline_vs_pocket.py --gpu 0
  python reproduce_v2_lsg_baseline_vs_pocket.py --skip_igev   # if depths already exist
  python reproduce_v2_lsg_baseline_vs_pocket.py --ate_only    # skip map refine (faster)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

EXPECTED = {
    "baseline": {"ate_cm": 6.64, "psnr": 32.50, "ssim": 0.984, "lpips": 0.037},
    "pocket": {"ate_cm": 6.38, "psnr": 32.04, "ssim": 0.982, "lpips": 0.042},
}


def run(cmd, env=None):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    env = dict(env or os.environ)
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not pp else f"{ROOT}:{pp}"
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    print(f"<<< done in {time.time() - t0:.1f}s", flush=True)


def check_prereqs(need_igev: bool):
    cam0 = ROOT / "euroc/V2_01_easy/mav0/cam0"
    missing = []
    for rel in ["data_rect", "traj.txt", "global_features"]:
        p = cam0 / rel
        if not p.exists():
            missing.append(str(p))
    depth = cam0 / "depth_sceneflow"
    n_depth = len(list(depth.glob("*.npy"))) if depth.is_dir() else 0
    n_rect = len(list((cam0 / "data_rect").glob("*.png"))) if (cam0 / "data_rect").is_dir() else 0
    if need_igev and n_depth < n_rect:
        ckpt = ROOT / "third_party/IGEV-Stereo/pretrained_models/sceneflow.pth"
        if not ckpt.exists():
            missing.append(str(ckpt))
    for w in [
        "third_party/TransVPR/TransVPR_MSLS.pth",
        "sp_lg/superpoint_v1.pth",
        "sp_lg/superpoint_lightglue.pth",
    ]:
        if not (ROOT / w).exists():
            missing.append(str(ROOT / w))
    if missing:
        print("Missing prerequisites:")
        for m in missing:
            print("  -", m)
        print("\nPrepare EuRoC V2_01_easy (IGEV depth + features) first. See README.")
        sys.exit(1)
    print(f"[ok] V2_01_easy cam0 ready | rect={n_rect} depth_sceneflow={n_depth}")


def ensure_igev(gpu: str):
    cam0 = ROOT / "euroc/V2_01_easy/mav0/cam0"
    depth = cam0 / "depth_sceneflow"
    n_depth = len(list(depth.glob("*.npy"))) if depth.is_dir() else 0
    n_rect = len(list((cam0 / "data_rect").glob("*.png")))
    if n_depth >= n_rect and n_rect > 0:
        print(f"[skip] IGEV depths already complete ({n_depth})")
        return
    print(f"[igev] generating depths ({n_depth}/{n_rect}) ...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    run(
        [
            sys.executable,
            "tools/euroc_parser/generate_igev_depth.py",
            "--cam0_dir",
            str(cam0),
            "--gpu",
            "0",
        ],
        env=env,
    )


def parse_metrics(workdir: Path):
    ate_path = workdir / "PoseGraphResult" / "ate_summary.txt"
    ate_cm = None
    if ate_path.exists():
        text = ate_path.read_text()
        m = re.search(r"loop_align_ate_m=([0-9.]+)", text)
        if m:
            ate_cm = float(m.group(1)) * 100.0  # m → cm

    render = workdir / "RenderingResult"
    out = {"ate_cm": ate_cm, "psnr": None, "ssim": None, "lpips": None}
    try:
        import numpy as np

        for key, fname in [
            ("psnr", "after_opt_psnr.txt"),
            ("ssim", "after_opt_ms_ssim.txt"),
            ("lpips", "after_opt_lpips.txt"),
        ]:
            p = render / fname
            if p.exists():
                out[key] = float(np.loadtxt(p).mean())
    except Exception as e:
        print(f"[warn] could not load rendering metrics from {render}: {e}")
    return out


def fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def print_table(results: dict, title: str):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"{'Method':<14} {'ATE(cm)':>10} {'PSNR':>8} {'MS-SSIM':>8} {'LPIPS':>8}")
    for name, key in [("LSG Baseline", "baseline"), ("Pocket-SLAM", "pocket")]:
        r = results.get(key, {})
        print(
            f"{name:<14} {fmt(r.get('ate_cm')):>10} {fmt(r.get('psnr')):>8} "
            f"{fmt(r.get('ssim'), 3):>8} {fmt(r.get('lpips'), 3):>8}"
        )
    print("-" * 64)
    print("Expected (reference):")
    print(f"{'LSG Baseline':<14} {EXPECTED['baseline']['ate_cm']:>10.2f} {EXPECTED['baseline']['psnr']:>8.2f} "
          f"{EXPECTED['baseline']['ssim']:>8.3f} {EXPECTED['baseline']['lpips']:>8.3f}")
    print(f"{'Pocket-SLAM':<14} {EXPECTED['pocket']['ate_cm']:>10.2f} {EXPECTED['pocket']['psnr']:>8.2f} "
          f"{EXPECTED['pocket']['ssim']:>8.3f} {EXPECTED['pocket']['lpips']:>8.3f}")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--skip_igev", action="store_true")
    parser.add_argument("--ate_only", action="store_true",
                        help="Skip structure refine (ATE only; no final PSNR)")
    parser.add_argument("--structure_refine_iters", type=int, default=5000)
    parser.add_argument("--fresh", action="store_true",
                        help="Delete previous euroc_lsg_v2_* workdirs before running")
    args = parser.parse_args()

    check_prereqs(need_igev=not args.skip_igev)
    if not args.skip_igev:
        ensure_igev(args.gpu)

    if args.fresh:
        for d in ["euroc_lsg_v2_baseline", "euroc_lsg_v2_pocket"]:
            p = ROOT / d
            if p.exists():
                print(f"[fresh] removing {p}")
                import shutil
                shutil.rmtree(p)

    # Full pipeline for both methods
    cmd = [
        sys.executable, "-u", "run_lsg_pipeline_v2.py",
        "--mode", "both",
        "--stage", "all",
        "--gpu", args.gpu,
        "--structure_refine_iters", str(args.structure_refine_iters),
    ]
    if args.ate_only:
        cmd.append("--skip_structure_refine")
    run(cmd)

    results = {
        "baseline": parse_metrics(ROOT / "euroc_lsg_v2_baseline"),
        "pocket": parse_metrics(ROOT / "euroc_lsg_v2_pocket"),
    }
    print_table(results, "V2_01_easy full LSG pipeline (chunk + loop + pose-graph + refine)")

    out_dir = ROOT / "results" / "v2_lsg_reproduce"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.txt"
    with open(summary, "w") as f:
        f.write("V2_01_easy LSG Baseline vs Pocket-SLAM\n")
        f.write("pipeline: chunks(step=200) + loop_closure + pose_graph + structure_refine\n\n")
        f.write(f"{'Method':<14} {'ATE(cm)':>10} {'PSNR':>8} {'MS-SSIM':>8} {'LPIPS':>8}\n")
        for name, key in [("LSG Baseline", "baseline"), ("Pocket-SLAM", "pocket")]:
            r = results[key]
            f.write(
                f"{name:<14} {fmt(r.get('ate_cm')):>10} {fmt(r.get('psnr')):>8} "
                f"{fmt(r.get('ssim'), 3):>8} {fmt(r.get('lpips'), 3):>8}\n"
            )
        f.write("\nExpected reference:\n")
        f.write("LSG Baseline    6.64 cm   32.50    0.984     0.037\n")
        f.write("Pocket-SLAM     6.38 cm   32.04    0.982     0.042\n")
    print(f"\nWrote {summary}")


if __name__ == "__main__":
    main()
