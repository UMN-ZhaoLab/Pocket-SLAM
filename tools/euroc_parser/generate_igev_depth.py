#!/usr/bin/env python3
"""Generate IGEV SceneFlow depths for a EuRoC cam0 folder that already has data_rect/."""
import argparse
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IGEV_ROOT = os.path.join(ROOT, "third_party/IGEV-Stereo")
sys.path.insert(0, IGEV_ROOT)
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder

# EuRoC stereo baseline * fx (same as operate_euroc_data.py)
baseline_times_fx = 47.90639384423901


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0_dir", required=True, help=".../mav0/cam0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument("--corr_implementation", default="reg")
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--max_disp", type=int, default=192)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"

    rect0 = os.path.join(args.cam0_dir, "data_rect")
    # right rect lives under cam1
    rect1 = os.path.join(os.path.dirname(args.cam0_dir), "cam1", "data_rect")
    if not os.path.isdir(rect1):
        # validation subset may only symlink cam0; fall back to full sequence cam1
        rect1 = "/mnt/hdd/lls/euroc/V2_01_easy/mav0/cam1/data_rect"
    out_disp = os.path.join(args.cam0_dir, "disparity_sceneflow")
    out_depth = os.path.join(args.cam0_dir, "depth_sceneflow")
    os.makedirs(out_disp, exist_ok=True)
    os.makedirs(out_depth, exist_ok=True)

    ckpt = os.path.join(ROOT, "third_party/IGEV-Stereo/pretrained_models/sceneflow.pth")
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(ckpt))
    model = model.module.to(device).eval()

    def load_image(path):
        img = np.array(Image.open(path)).astype(np.uint8)
        if img.ndim == 2:
            img = np.dstack([img, img, img])
        return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

    names = sorted(os.listdir(rect0), key=lambda x: float(x[:-4]))
    with torch.no_grad():
        for name in tqdm(names, desc="IGEV depth"):
            out_npy = os.path.join(out_depth, name[:-4] + ".npy")
            if os.path.exists(out_npy):
                continue
            left = load_image(os.path.join(rect0, name))
            right = load_image(os.path.join(rect1, name))
            padder = InputPadder(left.shape, divis_by=32)
            left_p, right_p = padder.pad(left, right)
            disp = model(left_p, right_p, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp).cpu().numpy().squeeze()
            np.save(os.path.join(out_disp, name[:-4]), disp)
            plt.imsave(os.path.join(out_disp, name), disp, cmap="jet")
            depth = baseline_times_fx / disp
            depth[depth < 0.1] = 0
            np.save(out_npy, depth)

    print("done:", out_depth, "files=", len(os.listdir(out_depth)))


if __name__ == "__main__":
    main()
