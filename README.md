# Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM

Official implementation of Pocket-SLAM (ICRA'26), built on [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM).

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
# TransVPR (clone LSG-SLAM or copy weights manually)
git clone --depth 1 https://github.com/lsg-slam/LSG-SLAM /tmp/LSG-SLAM
cp /tmp/LSG-SLAM/third_party/TransVPR/TransVPR_MSLS.pth third_party/TransVPR/
cp /tmp/LSG-SLAM/sp_lg/superpoint_v1.pth /tmp/LSG-SLAM/sp_lg/superpoint_lightglue.pth sp_lg/

# IGEV (optional; preprocessing below uses SGBM depth by default)
mkdir -p third_party/IGEV-Stereo/pretrained_models
gdown --folder https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz \
  -O third_party/IGEV-Stereo/pretrained_models
```

### 3. Download & preprocess EuRoC

Download [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (`V2_01_easy`).

```bash
export EUROC_DIR=/path/to/euroc          # contains V2_01_easy/
ln -sf $EUROC_DIR euroc                  # or edit base_path in operate_euroc_data.py

python tools/euroc_parser/operate_euroc_data.py
```

This generates rectified images, SGBM depth (`depth_sgbm/`), poses (`traj.txt`), and global features under `euroc/V2_01_easy/mav0/cam0/`.

### 4. Run benchmark (baseline vs Pocket-SLAM)

Full-sequence comparison on EuRoC `V2_01_easy`, frames 0–2200, stride 5:

```bash
# 2 GPUs in parallel (GPU 0 = LSG baseline, GPU 1 = Pocket-SLAM)
python run_full_benchmark.py

# Or run Pocket-SLAM only
python scripts/loop_closure.py configs/euroc/full_benchmark.py
```

Config: `configs/euroc/full_benchmark.py` (`N_tar=60000`, `B_max=200`, 100 tracking/mapping iters).

Results are written to `results/full_benchmark/` (`summary.txt`, per-run logs).

## Benchmark results

EuRoC `V2_01_easy`, frames 0–2200, stride 5 (441 keyframes), 100 tracking/mapping iters, 2× RTX A6000.

| Metric | LSG-SLAM (baseline) | Pocket-SLAM | Change |
|--------|---------------------|-------------|--------|
| FPS | 0.072 | 0.393 | **5.5×** |
| Peak VRAM | 11.1 GB | 5.0 GB | **−55%** |
| Final Gaussians | 8,512,714 | 55,908 | **−99.3%** |
| Map size | 422 MB | 2.8 MB | **−99.3%** |
| ATE RMSE | 813 cm | 854 cm | +5% |
| Wall time | ~112 min | ~27 min | **~4× faster** |

Pocket-SLAM trades a small ATE increase for large memory and speed gains on this long outdoor sequence.

## Code changes (vs upstream)

| File | Change |
|------|--------|
| `utils/slam_external.py` | Global cap in `pocket_slam_prune()`; tile-budget pruning |
| `scripts/loop_closure.py` | Fix mapping to run every frame; adaptive mapping/prune |
| `datasets/gradslam_datasets/euroc.py` | Use `depth_sgbm/` depth maps |
| `tools/euroc_parser/operate_euroc_data.py` | SGBM depth by default; `EUROC_DIR` env var |
| `configs/euroc/full_benchmark.py` | Full V2_01_easy benchmark config |
| `run_full_benchmark.py` | Side-by-side baseline vs Pocket benchmark |

Pocket-SLAM config block:

```python
pocket_slam=dict(
    enable=True,
    N_tar=60000,
    B_min=1,
    B_max=200,
    tile_size=16,
),
```

## Acknowledgement

Built on [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM) and [SplaTAM](https://github.com/spla-tam/SplaTAM).
