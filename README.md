# Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM

Official implementation of Pocket-SLAM (ICRA'26), built on [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM).

Paper: [arXiv:2606.24796](https://arxiv.org/abs/2606.24796)

## Quick start

### 1. Install

```bash
conda create -n pocket-slam python=3.10
conda activate pocket-slam
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
pip install gdown gtsam "numpy<2"

cd diff-gaussian-rasterization-w-depth.git
python setup.py install
pip install .
cd ..
```

### 2. Download weights

```bash
git clone --depth 1 https://github.com/lsg-slam/LSG-SLAM /tmp/LSG-SLAM
cp /tmp/LSG-SLAM/third_party/TransVPR/TransVPR_MSLS.pth third_party/TransVPR/
cp /tmp/LSG-SLAM/sp_lg/superpoint_v1.pth /tmp/LSG-SLAM/sp_lg/superpoint_lightglue.pth sp_lg/

mkdir -p third_party/IGEV-Stereo/pretrained_models
gdown --folder https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz \
  -O third_party/IGEV-Stereo/pretrained_models
# Ensure sceneflow.pth is at:
#   third_party/IGEV-Stereo/pretrained_models/sceneflow.pth
```

### 3. Download & preprocess EuRoC (MH01–MH05)

Paper tables use **Machine Hall** sequences `MH_01_easy` … `MH_05_difficult` (not V2).

```bash
# Download from EuRoC (needs network access to ETHZ):
# http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/

export EUROC_DIR=/path/to/euroc   # contains MH_01_easy/, MH_02_easy/, ...
ln -sfn $EUROC_DIR euroc

# Edit tools/euroc_parser/operate_euroc_data.py → scene_names = ["MH_01_easy", ...]
# Must use IGEV depth (depth_sceneflow/), NOT SGBM.
python tools/euroc_parser/operate_euroc_data.py
```

Preprocessing writes under `euroc/<seq>/mav0/cam0/`:
`data_rect/`, **`depth_sceneflow/`**, `traj.txt`, `global_features/`.

### 4. Run (paper pipeline)

Frame ranges (stride=5), from `bash_scripts/run_euroc_sequence.bash`:

| Sequence | start | end |
|----------|------:|----:|
| MH_01_easy | 900 | 3630 |
| MH_02_easy | 780 | 2990 |
| MH_03_medium | 350 | 2600 |
| MH_04_difficult | 368 | 1970 |
| MH_05_difficult | 400 | 2220 |

Paper hyperparameters (`configs/euroc/lsgslam.py`):

- tracking iters **50**, mapping iters **100**
- tracking depth weight **1.0**, mapping depth weight **1.5**
- `N_tar = 0.4 * N_init`, `B_min=5`, `B_max=200`

```bash
# Frontend (+ loop finding / segment re-runs)
python scripts/loop_closure.py configs/euroc/lsgslam.py

# Backend pose-graph + structure refine (required for paper-level ATE)
python tools/loop_closure/pose_graph_part_optim.py
```

Baseline (no Pocket pruning): set `pocket_slam.enable=False` in the config.

## Paper results (EuRoC Table I/II)

| Metric | LSG-SLAM | Pocket-SLAM (w/ tile budget) |
|--------|----------|------------------------------|
| MH01 ATE (m) | 0.05 | 0.05 |
| MH01 PSNR | 31.23 | 31.05 |
| Avg peak mem (GB) | 25.4 | 10.1 |
| Avg FPS | 1.3 | 3.6 |

## Common reproduction pitfalls

1. **SGBM depth** → ATE/PSNR collapse. Use **IGEV `depth_sceneflow`**.
2. **Skip pose-graph backend** → worse ATE than Table I.
3. **Wrong sequence / frame range** (e.g. V2 or full MH01 from frame 0).
4. Upstream `loop_closure.py` had mapping gated to frame 0; this repo maps every keyframe.

## Acknowledgement

Built on [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM) and [SplaTAM](https://github.com/spla-tam/SplaTAM).
