import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from math import exp, sqrt
from tqdm import tqdm
import wandb
import open3d as o3d
from sklearn.neighbors import KDTree

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset, KittiDataset, EurocDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval, plot_progress
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import (calc_ssim, build_rotation, prune_gaussians, densify,
                                  compute_tile_budgets, pocket_slam_prune)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from sp_lg.lightglue import LightGlue
from sp_lg.superpoint import SuperPoint
from sp_lg.disk import DISK
from sp_lg.utils import load_image, match_pair
from sp_lg import viz2d
from feature_matching import *


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["kitti"]:
        return KittiDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["euroc"]:
        return EurocDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, config=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose, depth_original, global_feature = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), depth_threshold=config['pixel_gs_depth_threshold'])
    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), depth_threshold=config['pixel_gs_depth_threshold'])
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, grad_mask=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    gt_image = curr_data['im']
    if grad_mask is not None:
        gt_image = grad_mask * curr_data['im']
        im = grad_mask * im
    
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(gt_image - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(gt_image - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, gt_image) + 0.2 * (1.0 - calc_ssim(im, gt_image))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)
        if 'pocket_grad_accum' in variables:
            variables['pocket_grad_accum'] = torch.zeros(num_pts, device="cuda").float()
            variables['pocket_denom'] = torch.zeros(num_pts, device="cuda").float()

    return params, variables


def icp(target, source, init_pose_T12, corr_threshold=0.5):
    # target = o3d.geometry.PointCloud()
    # target.points = o3d.utility.Vector3dVector(submap1_data[::self.skip_points, :3])
    # target.colors = o3d.utility.Vector3dVector(submap1_data[::self.skip_points, 3:6])

    # source = o3d.geometry.PointCloud()
    # source.points = o3d.utility.Vector3dVector(submap2_data[::self.skip_points, :3])
    # source.colors = o3d.utility.Vector3dVector(submap2_data[::self.skip_points, 3:6])

    # print("3-1. Downsample with a voxel size %.2f" % 0.2)
    source = source.voxel_down_sample(0.1)
    target = target.voxel_down_sample(0.1)

    num_points = 20  # 邻域球内的最少点数，低于该值的点为噪声点
    radius = 0.4   # 邻域半径大小
    # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    source, ind = source.remove_radius_outlier(num_points, radius)
    print('filter outliers')
    # source = source.select_by_index(ind,invert = True)
    # print('filter outliers')
    target, ind = target.remove_radius_outlier(num_points, radius)
    print('filter outliers')
    # target = target.select_by_index(ind,invert = True)
    # print('filter outliers')

    # print("3-2. Estimate normal.")
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2 * 2, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2 * 2, max_nn=30))

    # # calculate relative pose from coarse to fine

    # threshold = [5.0, 0.5, 0.1]
    if corr_threshold > 0.5:
        threshold = [corr_threshold, 0.5, 0.1]
    else:
        threshold = [0.5, 0.1]
    print(threshold)
    
    relative_transpose = init_pose_T12

    for i in range(len(threshold)):
        max_correspondence_distance = threshold[i]

        relative_rmse = 0.1
        max_iteration = 100
        if max_correspondence_distance == 0.1:
            relative_rmse = 0.001
            max_iteration = 100
        # print("Apply point-to-plane ICP")
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, relative_transpose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=relative_rmse,
                                                                max_iteration=max_iteration)
        )

        relative_transpose = reg_p2l.transformation

    # print('coarse Transformation is:')
    # print(pose_T12)
    # print(reg_p2l)
    print("refine Transformation is:")
    print(init_pose_T12)
    print(reg_p2l.transformation)
    print(np.array(reg_p2l.correspondence_set).shape)
    # print(submap1_data.shape)
    # print(submap2_data.shape)
    # draw_registration_result(source, target, init_pose_T12)
    # draw_registration_result(source, target, reg_p2l.transformation)
    # exit()
    return reg_p2l.transformation, reg_p2l.fitness, reg_p2l.inlier_rmse


def icp_init_camera_pose(params, curr_time_idx, now_pc, pre_pc, intrinsics, depth_filter_near, depth_filter_far, 
                         init_pose, corr_threshold, device):
    with torch.no_grad():
        # icp
        pre2now, fitness, inlier_rmse = icp(now_pc, pre_pc, init_pose, corr_threshold)
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
        cam_tran = params['cam_trans'][..., curr_time_idx-1].detach()
        w2pre = torch.eye(4).float().to(device=device)
        w2pre[:3, :3] = build_rotation(cam_rot)
        w2pre[:3, 3] = cam_tran
        pre2now_tmp = torch.tensor(pre2now, dtype=torch.float32, device=device)
        w2now = pre2now_tmp @ w2pre
        rel_w2c_rot = w2now[:3, :3].unsqueeze(0).detach()
        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
        rel_w2c_tran = w2now[:3, 3].detach()
        # Update the camera parameters
        params['cam_unnorm_rots'][..., curr_time_idx] = rel_w2c_rot_quat
        params['cam_trans'][..., curr_time_idx] = rel_w2c_tran

    return params, pre2now


def initialize_camera_pose(params, curr_time_idx, forward_prop, gt_w2c):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
       
        # GT pose作为初始值
        # rel_w2c = gt_w2c
        # rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
        # rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
        # rel_w2c_tran = rel_w2c[:3, 3].detach()
        # Update the camera parameters
        # params['cam_unnorm_rots'][..., curr_time_idx] = rel_w2c_rot_quat
        # params['cam_trans'][..., curr_time_idx] = rel_w2c_tran
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def compute_grad_mask(original_image, dataset, edge_threshold=1.1):
    gray_img = original_image.mean(dim=0, keepdim=True)
    gray_grad_v, gray_grad_h = image_gradient(gray_img)
    mask_v, mask_h = image_gradient_mask(gray_img)
    gray_grad_v = gray_grad_v * mask_v
    gray_grad_h = gray_grad_h * mask_h
    img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)
    
    grad_mask = None
    if isinstance(dataset, ReplicaDataset) == isinstance(dataset, ReplicaV2Dataset):
        edge_threshold = 4
        row, col = 32, 32
        multiplier = edge_threshold
        _, h, w = original_image.shape
        for r in range(row):
            for c in range(col):
                block = img_grad_intensity[
                    :,
                    r * int(h / row) : (r + 1) * int(h / row),
                    c * int(w / col) : (c + 1) * int(w / col),
                ]
                th_median = block.median()
                block[block > (th_median * multiplier)] = 1
                block[block <= (th_median * multiplier)] = 0
        grad_mask = img_grad_intensity
    else:
        median_img_grad_intensity = img_grad_intensity.median()
        grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )
    return grad_mask


def rgbd_slam(config: dict, loop):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval_{}_{}".format(loop[0], loop[1]))
    os.makedirs(eval_dir, exist_ok=True)
    plot_progress_dir = os.path.join(output_dir, "plot_progress_{}_{}".format(loop[0], loop[1]))
    os.makedirs(plot_progress_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # SuperPoint+LightGlue
    max_num_keypoints = 1024
    sp_extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    match_conf = {
        "width_confidence": 0.99,  # for point pruning
        "depth_confidence": 0.95,  # for early stopping,
    }
    lg_matcher = LightGlue(pretrained="superpoint", **match_conf).eval().to(device)

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    _, _, _, _, depth_original_first_frame, _ = dataset[0]
    pixel_gs_depth_threshold = config['pixel_gs_depth_gamma'] * torch.max(depth_original_first_frame).item()
    config['pixel_gs_depth_threshold'] = pixel_gs_depth_threshold


    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'], config=config)                                                                                                             
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'], config=config)
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy(), depth_threshold=config['pixel_gs_depth_threshold'])
    
    # Initialize Pocket-SLAM per-Gaussian gradient accumulators (tracking stage)
    if config.get('pocket_slam', {}).get('enable', False):
        variables['pocket_grad_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['pocket_denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()

    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose, depth_original, global_feature = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0

    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose, depth_original, global_feature = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        depth_original = depth_original.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx

        print("use_gt_poses =", config["tracking"]["use_gt_poses"])

        if config["use_warp_loss"]:
            if depth_original is not None:
                mask = (depth_original < 0.1) | (depth_original > np.min([dataset.depth_filter_far, 15.0]))
            else:
                mask = (depth < 0.1) | (depth > np.min([dataset.depth_filter_far, 15.0]))
            color_feature = torch.clone(color)
            color_feature[:, mask[0]] = 0
            curr_feats, curr_desc_all = extract_feature(color_feature, depth_original, sp_extractor, device)
        else:
            curr_feats = None
            curr_desc_all = None
        # print(time_idx, image_name)

        if time_idx == 0:
            last_data = None
        else:
            last_data = curr_data
            # print('init = ', len(last_data['iter_gt_w2c_list']))
            if config["use_warp_loss"]:
                match_save_dir = os.path.join(eval_dir, "match_res")
                os.makedirs(match_save_dir, exist_ok=True)
                match_save_path = os.path.join(match_save_dir, "{}_matches.png".format(time_idx))
                mkpts_cur, mkpts_last, mscores = match_feature(
                    config,
                    color,
                    curr_feats,
                    intrinsics,
                    last_data["im"],
                    last_data["feats"],
                    lg_matcher,
                    device,
                    topk=1024,
                    save_path=match_save_path,
                )
                if mkpts_cur is not None and mkpts_cur.shape[0] > 10:
                    np.savetxt(os.path.join(match_save_dir, "{}_inliers.txt".format(time_idx)), mkpts_cur.shape)


                pnp_prior_pose_w2c, est_T_curr_last, pnp_num_inliers = estimate_pnp(mkpts_cur, mkpts_last, curr_data, last_data, dataset)
            else:
                mkpts_cur = None
                mkpts_last = None
                mscores = None
                pnp_prior_pose_w2c = None

        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'depth_original': depth_original, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                    'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c, "feats": curr_feats, "descs": curr_desc_all}

        grad_mask = None
        if config["use_grad_mask"]:
            print("use_grad mask")
            grad_mask = compute_grad_mask(curr_data['im'], dataset=dataset, edge_threshold=1.1)
            
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']

        color_tmp = color.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
        if depth_original is not None:
            depth_tmp = depth_original.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
        else:
            depth_tmp = depth.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
        depth_tmp = depth_tmp.squeeze(-1)
        x, y = np.meshgrid(
                range(color_tmp.shape[1]), range(color_tmp.shape[0])
            )
        pts = np.vstack(
            (
                x.flatten(),
                y.flatten(),
                np.ones(color_tmp.shape[0] * color_tmp.shape[1]),
            )
        )
        pts = torch.tensor(pts, dtype=torch.float32, device=device)
        depth_tmp = depth_tmp.flatten()
        color_tmp = color_tmp.reshape([-1, 3])
        valid_depth_indices = torch.where((depth_tmp > np.max([0.1, dataset.depth_filter_near])) & (depth_tmp < np.max([30.0, dataset.depth_filter_far])))[0]
        pts = pts[:, valid_depth_indices]
        depth_tmp = depth_tmp[valid_depth_indices]
        color_tmp = color_tmp[valid_depth_indices, :]
        X = torch.multiply(depth_tmp.flatten(), torch.linalg.inv(intrinsics) @ pts) # (3, n)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(X.t().cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(color_tmp.cpu().numpy())
        init_pose = np.eye(4)
        curr_data = {**curr_data, "pc": pc}
            
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            if pnp_num_inliers > 100 and not isinstance(dataset, KittiDataset):
            # if pnp_num_inliers > 100:
                print('use pnp prior pose')
                with torch.no_grad():
                    # use prior pose from pnp
                    rel_w2c_rot = (
                        torch.from_numpy(pnp_prior_pose_w2c[:3, :3])
                        .unsqueeze(0)
                        .detach()
                        .to(device)
                    )
                    rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                    rel_w2c_tran = (
                        torch.from_numpy(pnp_prior_pose_w2c[:3, 3]).detach().to(device)
                    )
                    # Update the camera parameters
                    params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat
                    params["cam_trans"][..., time_idx] = rel_w2c_tran
            else:
                init_pose = est_T_curr_last
                icp_corr_threshold = config['tracking']['icp_corr_threshold']
                if pnp_num_inliers < 50:
                    icp_corr_threshold = np.max([3.0, icp_corr_threshold])
                
                params, init_pose = icp_init_camera_pose(params, time_idx, curr_data['pc'], last_data['pc'], intrinsics, dataset.depth_filter_near, 
                                                            dataset.depth_filter_far, init_pose, icp_corr_threshold, device)
                
                curr_gt_pose_w2c = torch.clone(curr_data['iter_gt_w2c_list'][-1]).detach().cpu().numpy()
                last_gt_pose_w2c = torch.clone(last_data['iter_gt_w2c_list'][-2]).detach().cpu().numpy()
                gt_T_curr_last = curr_gt_pose_w2c @ np.linalg.inv(last_gt_pose_w2c)

                T_error = np.linalg.inv(init_pose) @ gt_T_curr_last
                r_err = cv2.Rodrigues(T_error[:3, :3])[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi
                t_err = np.linalg.norm(T_error[:3, 3], ord=2)
                print("icp T_error = ", r_err, t_err)

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            milestones = [int(config['tracking']['num_iters'] * 0.8),
                        int(config['tracking']['num_iters'] * 0.85),
                        int(config['tracking']['num_iters'] * 0.9),]
            pose_rot_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.9)
            pose_trans_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.9)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            if config["use_warp_loss"] and pnp_prior_pose_w2c is None:
                num_iters_tracking *= 2
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                tracking_iteration=iter, grad_mask=grad_mask)
                if config["use_warp_loss"] and time_idx > 0:
                    warp_loss = get_loss_from_match(
                        time_idx,
                        mkpts_cur,
                        mkpts_last,
                        mscores,
                        params,
                        curr_data,
                        last_data,
                        iter,
                        device,
                        config,
                    )
                    weight_warp = 50
                    warp_loss *= weight_warp
                    loss = loss / weight_warp + warp_loss

                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Pocket-SLAM: accumulate per-Gaussian gradient magnitudes during tracking
                if config.get('pocket_slam', {}).get('enable', False):
                    if variables['means2D'].grad is not None:
                        seen = variables['seen']
                        grad_norms = torch.norm(variables['means2D'].grad[seen, :2], dim=-1)
                        variables['pocket_grad_accum'][seen] += grad_norms.detach()
                        variables['pocket_denom'][seen] += 1
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pose_rot_scheduler.step()
                pose_trans_scheduler.step()
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if config['tracking']['use_l1'] and losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
            # Pocket-SLAM: compute per-tile gradient budgets from tracking stage
            if config.get('pocket_slam', {}).get('enable', False):
                pocket_cfg = config['pocket_slam']
                trk_cam = tracking_curr_data['cam']
                with torch.no_grad():
                    tile_budgets = compute_tile_budgets(
                        variables['pocket_grad_accum'],
                        variables['pocket_denom'],
                        variables['means2D'].detach(),
                        trk_cam.image_height,
                        trk_cam.image_width,
                        N_tar=pocket_cfg.get('N_tar', params['means3D'].shape[0]),
                        tile_size=pocket_cfg.get('tile_size', 16),
                        B_min=pocket_cfg.get('B_min', 1),
                        B_max=pocket_cfg.get('B_max', None),
                    )
                variables['tile_budgets'] = tile_budgets
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                    config['mapping']['sil_thres'], time_idx,
                                                    config['mean_sq_dist_method'], config['gaussian_distribution'])
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

            # Mapping
            if time_idx < 1:
                mapping_start_time = time.time()
                if num_iters_mapping > 0:
                    progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                for iter in range(num_iters_mapping):
                    iter_start_time = time.time()
                    # Randomly select a frame until current time step amongst keyframes
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                    iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                    # Loss for current frame
                    loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                    config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                    config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True, grad_mask=grad_mask)
                    if config['use_wandb']:
                        # Report Loss
                        wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        if config['mapping']['prune_gaussians']:
                            params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                            if config['use_wandb']:
                                wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                            "Mapping/step": wandb_mapping_step})
                        # Gaussian-Splatting's Gradient-based Densification
                        if config['mapping']['use_gaussian_splatting_densification']:
                            params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                            if config['use_wandb']:
                                wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                            "Mapping/step": wandb_mapping_step})
                        # Optimizer Update
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # Report Progress
                        if config['report_iter_progress']:
                            if config['use_wandb']:
                                report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                                wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                                mapping=True, online_time_idx=time_idx)
                            else:
                                report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                                mapping=True, online_time_idx=time_idx)
                        else:
                            progress_bar.update(1)
                    # Update the runtime numbers
                    iter_end_time = time.time()
                    mapping_iter_time_sum += iter_end_time - iter_start_time
                    mapping_iter_time_count += 1
                if num_iters_mapping > 0:
                    progress_bar.close()
                # Update the runtime numbers
                mapping_end_time = time.time()
                mapping_frame_time_sum += mapping_end_time - mapping_start_time
                mapping_frame_time_count += 1

            # Pocket-SLAM: Rendering-Area-Aware Pruning with Tile-Level Budget
            if config.get('pocket_slam', {}).get('enable', False):
                pocket_cfg = config['pocket_slam']
                with torch.no_grad():
                    transformed_gaussians = transform_to_frame(
                        params, time_idx, gaussians_grad=False, camera_grad=False)
                    rendervar = transformed_params2rendervar(params, transformed_gaussians)
                    rendervar['means2D'] = torch.zeros_like(
                        params['means3D'], requires_grad=False, device="cuda")
                    _, pocket_radius, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
                    pocket_means2D = rendervar['means2D']
                    map_cam = curr_data['cam']
                params, variables = pocket_slam_prune(
                    params, variables, optimizer,
                    pocket_means2D, pocket_radius,
                    time_idx,
                    map_cam.image_height,
                    map_cam.image_width,
                    pocket_cfg,
                )
                # Reset pocket accumulators for next tracking round
                N_after = params['means3D'].shape[0]
                variables['pocket_grad_accum'] = torch.zeros(N_after, device="cuda").float()
                variables['pocket_denom']       = torch.zeros(N_after, device="cuda").float()

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')

        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Updata current image pose
        with torch.no_grad():
            # Get the current estimated rotation & translation
            curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran
            curr_data = {**curr_data, "est_w2c": curr_w2c.detach()}

            if time_idx > 0:
                rel_w2curr = curr_data["est_w2c"].to(device)
                rel_w2last = last_data["est_w2c"].to(device)
                T_last_cur = rel_w2last.to(device) @ rel_w2curr.inverse()

                rel_w2curr_gt = curr_data["iter_gt_w2c_list"][curr_data["id"]].to(device)
                rel_w2last_gt = last_data["iter_gt_w2c_list"][last_data["id"]].to(device)
                T_last_cur_gt = rel_w2last_gt.to(device) @ rel_w2curr_gt.inverse()

        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                    "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                    "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                    "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                    "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                eval_every=config['eval_every'], align_traj=False)
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                eval_every=config['eval_every'], align_traj=False)

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, eval_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()


def find_loops(config: dict):
    # Print Config
    print("Loaded Config:")
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "vis_loops")
    os.makedirs(eval_dir, exist_ok=True)

    # Get Device
    device = torch.device(config["primary_device"])

    # SuperPoint+LightGlue
    max_num_keypoints = 1024
    sp_extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    match_conf = {
        "width_confidence": 0.99,  # for point pruning
        "depth_confidence": 0.95,  # for early stopping,
    }
    lg_matcher = LightGlue(pretrained="superpoint", **match_conf).eval().to(device)

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    _, _, _, _, depth_original_first_frame, _ = dataset[0]
    pixel_gs_depth_threshold = config['pixel_gs_depth_gamma'] * torch.max(depth_original_first_frame).item()
    config['pixel_gs_depth_threshold'] = pixel_gs_depth_threshold

    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)


    # Initialize Parameters & Canoncial Camera parameters
    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                    config['scene_radius_depth_ratio'],
                                                                                    config['mean_sq_dist_method'],
                                                                                    gaussian_distribution=config['gaussian_distribution'], config=config)
    
    print("find loops")
    keyframe_idxs = []
    keyframe_features = []
    keyframe_infos = {}
    keyframe_colors = []
    for time_idx in tqdm(range(num_frames)):
        is_keyframe = ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or (time_idx == num_frames-2))
        if not is_keyframe:
            continue
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose, depth_original, global_feature = dataset[time_idx]

        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        depth_original = depth_original.permute(2, 0, 1)
        
        color = color.detach().cpu().numpy()
        depth_original = depth_original.detach().cpu().numpy()
        gt_w2c = gt_w2c.detach().cpu().numpy()

        keyframe_idxs.append(time_idx)
        keyframe_features.append(global_feature)
        keyframe_infos[time_idx] = {"color": color, "depth_original": depth_original, "gt_w2c": gt_w2c}
        
        color_image = cv2.imread(dataset.color_paths[time_idx], 0)
        if len(color_image.shape) == 2:
            color_image = np.dstack([color_image, color_image, color_image])
        # print(color_image.shape)
        keyframe_colors.append(color_image)
    
    keyframe_features = np.array(keyframe_features).reshape([len(keyframe_idxs), -1])
    print('number of feature = {}'.format(keyframe_features.shape))

    loop_detect_interval = 10
    loop_before_interval = 10
    topk = 5
    pnp_inlier_threshold = 100
    adj_rot_error = 1.0  # degree
    adj_trans_error = 0.2  # meter
    if isinstance(dataset, KittiDataset):
        loop_detect_interval = 5
        loop_before_interval = 5
        adj_rot_error = 2.0  # degree
        adj_trans_error = 0.5  # meter


    found_loops = []
    last_found_loop_idx = -1
    for kidx in range(loop_before_interval + 1, keyframe_features.shape[0] - 2, int(loop_detect_interval / 4)):
        if last_found_loop_idx > 0 and kidx - last_found_loop_idx < loop_detect_interval:
            continue
        current_kf_feature = keyframe_features[kidx, :]
        search_kf_features = keyframe_features[:kidx - loop_before_interval, :]
        num_search_kfs = search_kf_features.shape[0]
        if num_search_kfs < 10:
            continue
        print("num_search_kfs = ", num_search_kfs)
        feature_distance_matrix = np.sqrt(np.sum(np.square(current_kf_feature.reshape([1, -1])  - search_kf_features.reshape([num_search_kfs, -1])), axis=-1))
        print(feature_distance_matrix.shape)
        
        search_knn_idx = np.argsort(feature_distance_matrix)

        curr_feats, curr_desc_all = get_local_features(keyframe_infos[kidx], sp_extractor, device, dataset.depth_filter_far)            
        curr_color = torch.from_numpy(keyframe_infos[kidx]['color']).to(device)
        operated_knn_kf = []
        for k in range(topk):
            knn_idx = search_knn_idx[k]

            if len(operated_knn_kf) > 0 and np.min(abs(np.array(operated_knn_kf) - knn_idx)) < loop_detect_interval:
                continue

            knn_feats, knn_desc_all = get_local_features(keyframe_infos[knn_idx], sp_extractor, device, dataset.depth_filter_far) 
            knn_color = torch.from_numpy(keyframe_infos[knn_idx]['color']).to(device)

            mkpts_cur, mkpts_knn, mscores = match_feature(
                config,
                curr_color,
                curr_feats,
                intrinsics,
                knn_color,
                knn_feats,
                lg_matcher,
                device,
                topk=1024,
                save_path=None,
            )
            if mkpts_cur is None:
                continue

            est_T_curr_knn, pnp_num_inliers = estimate_pnp_for_loop(mkpts_cur, mkpts_knn, intrinsics, keyframe_infos[kidx], keyframe_infos[knn_idx], dataset)
            print(pnp_num_inliers)
            if pnp_num_inliers < pnp_inlier_threshold:
                continue
            
            current_match_res = {'mkpts_cur': mkpts_cur, 'mkpts_knn': mkpts_knn, 'mscores': mscores}

            adjacent_keyframe_idxs = [kidx - 1, kidx + 1]
            good_loop_candidate = True
            for adj_idx in adjacent_keyframe_idxs:
                adj_feats, adj_desc_all = get_local_features(keyframe_infos[adj_idx], sp_extractor, device, dataset.depth_filter_far) 
                adj_color = torch.from_numpy(keyframe_infos[adj_idx]['color']).to(device)

                mkpts_adj, mkpts_knn, mscores = match_feature(
                    config,
                    adj_color,
                    adj_feats,
                    intrinsics,
                    knn_color,
                    knn_feats,
                    lg_matcher,
                    device,
                    topk=1024,
                    save_path=None,
                )
                if mkpts_adj is None:
                    continue
                est_T_adj_knn, pnp_num_inliers = estimate_pnp_for_loop(mkpts_adj, mkpts_knn, intrinsics, keyframe_infos[adj_idx], keyframe_infos[knn_idx], dataset)
                print(pnp_num_inliers)

                curr_gt_pose_w2c = np.copy(keyframe_infos[kidx]['gt_w2c'])
                adj_gt_pose_w2c = np.copy(keyframe_infos[adj_idx]['gt_w2c'])
                gt_T_curr_adj = curr_gt_pose_w2c @ np.linalg.inv(adj_gt_pose_w2c)
                est_T_curr_adj = est_T_curr_knn @ np.linalg.inv(est_T_adj_knn)
                T_error = np.linalg.inv(est_T_curr_adj) @ gt_T_curr_adj
                r_err = cv2.Rodrigues(T_error[:3, :3])[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi
                t_err = np.linalg.norm(T_error[:3, 3], ord=2)
                print("adj_T_error = ", r_err, t_err)
                
                if pnp_num_inliers < pnp_inlier_threshold or r_err > adj_rot_error or t_err > adj_trans_error:
                    good_loop_candidate = False
                    print('false loop candidate')
                    break
            if not good_loop_candidate:
                continue
            print('good_loop_candidate:', good_loop_candidate)
            
            # for visualization
            img0 = torch.clone(curr_color)
            img1 = torch.clone(knn_color)
            m_kpts0 = current_match_res['mkpts_cur'].detach().cpu().numpy()
            m_kpts1 = current_match_res['mkpts_knn'].detach().cpu().numpy()
            axes = viz2d.plot_images(
                [img0.permute(1, 2, 0), img1.permute(1, 2, 0)]
            )
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            # plt.show()

            curr_name = os.path.basename(dataset.color_paths[kidx]).split('.')[0]
            knn_name = os.path.basename(dataset.color_paths[knn_idx]).split('.')[0]
            save_path = os.path.join(eval_dir, "{}_{}_{}_{}_matches.png".format(kidx, curr_name, knn_idx, knn_name))
            plt.savefig(save_path)

            found_loops.append([kidx, knn_idx]) 
            last_found_loop_idx = kidx   
            operated_knn_kf.append(knn_idx)
    print(found_loops)
    found_loops = np.array(found_loops)
    np.save(os.path.join(output_dir, 'found_loops'), found_loops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    experiment.config["run_name"] = experiment.config["run_name"] + '_loops'
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
    
    find_loops(experiment.config)
    found_loops = np.load(os.path.join(results_dir, 'found_loops.npy'))
    print(found_loops)

    for loop in found_loops:
        experiment = SourceFileLoader(
            os.path.basename(args.experiment), args.experiment
        ).load_module()

        # Create Results Directory and Copy Config
        experiment.config["run_name"] = experiment.config["run_name"] + '_loops'
        experiment.config["tracking"]['num_iters'] = 300
        experiment.config["mapping"]['num_iters'] = 100

        experiment.config["use_warp_loss"] = True

        start_idx = experiment.config['data']['start']
        stride = experiment.config['data']['stride']
        ref_kf_idx = loop[1]
        query_kf_idx = loop[0]
        start = start_idx + (ref_kf_idx) * stride
        end = start_idx + (query_kf_idx) * stride

        experiment.config['data']['start'] = start
        experiment.config['data']['end'] = end
        experiment.config['data']['stride'] = end - start
        print(experiment.config['data'])
        
        rgbd_slam(experiment.config, loop)
