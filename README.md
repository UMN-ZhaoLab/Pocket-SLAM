# Pocket-SLAM

Pocket-SLAM is built on top of [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM) and introduces memory-efficient Gaussian pruning for large-scale outdoor SLAM via two complementary strategies:

- **Rendering-Area–Aware Pruning**: prunes Gaussians based on their pixel-coverage contribution in the current frame.
- **Tile-Level Budget Mechanism**: allocates per-tile survival budgets from tracking-stage Gaussian gradients, preventing over-pruning in texture-rich regions.

## Changes from LSG-SLAM

| File | Change |
|------|--------|
| `utils/slam_external.py` | Added `compute_tile_budgets()` and `pocket_slam_prune()`; extended `remove_points()` to maintain new accumulators |
| `scripts/loop_closure.py` | Accumulates per-Gaussian gradients during tracking; computes tile budgets after tracking; calls pruning after mapping |
| `configs/kitti/lsgslam.py` | Added `pocket_slam` config block |
| `configs/euroc/lsgslam.py` | Added `pocket_slam` config block |

Enable/disable and tune via the config:
```python
pocket_slam=dict(
    enable=True,
    N_tar=80000,   # target Gaussian count
    B_min=1,       # min budget per tile
    B_max=2000,    # max budget per tile
    tile_size=16,
),
```

## Installation

```bash
# Create conda environment
conda create -n lsgslam python=3.10
conda activate lsgslam

conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt

# Build extension
cd diff-gaussian-rasterization-w-depth.git
python setup.py install
pip install .
```

## Dataset

[EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and [KITTI](https://www.cvlibs.net/datasets/kitti/).

## Run

```bash
# Preprocess data
python tools/euroc_parser/operate_euroc_data.py
python tools/kitti_parser/operate_kitti_data.py

# Front end + loop closure
python scripts/loop_closure.py configs/euroc/lsgslam.py

# Back end
python tools/loop_closure/pose_graph_part_optim.py
```

## Acknowledgement

Built on [LSG-SLAM](https://github.com/lsg-slam/LSG-SLAM) (ICRA'25) and [SplaTAM](https://github.com/spla-tam/SplaTAM).
