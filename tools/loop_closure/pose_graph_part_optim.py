import numpy as np
np.set_printoptions(precision=4)
import open3d as o3d
import os 
import torch
import cv2
import copy
import time
import math
import matplotlib.pyplot as plt
import gtsam
import csv
import random
import wandb
from datetime import datetime
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import sys
import glob
import imageio.v2 as imageio
import torchvision.transforms as transforms
import rich
from PIL import Image
current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(current_dir)
from natsort import natsorted
from matplotlib.animation import FFMpegWriter
from utils.slam_external import build_rotation, calc_ssim, calc_psnr, densify, densify_with_bound, densify_use_pixel_gs
from utils.eval_helpers import align
from utils.recon_helpers import setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.slam_helpers import transformed_params2rendervar, l1_loss_v1, transformed_params2depthplussilhouette, matrix_to_quaternion, transform_to_frame
from utils.common_utils import save_params
from datasets.gradslam_datasets import (load_dataset_config, KittiDataset, EurocDataset, GradSLAMDataset)
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def getConstDigitsNumber(val, num_digits):
    return "{:.{}f}".format(val, num_digits)

def getUnixTime():
    return int(time.time())

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def yawdeg2so3(yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    return eulerAnglesToRotationMatrix([0, 0, yaw_rad])

def yawdeg2se3(yaw_deg):
    se3 = np.eye(4)
    se3[:3, :3] = yawdeg2so3(yaw_deg)
    return se3 


def getGraphNodePose(graph, idx):

    pose = graph.atPose3(gtsam.symbol('x', idx))
    pose_trans = np.array([pose.x(), pose.y(), pose.z()])
    pose_rot = pose.rotation().matrix()

    return pose_trans, pose_rot

def saveOptimizedGraphPose(curr_node_idx, graph_optimized, filename):

    for opt_idx in range(curr_node_idx):
        pose_trans, pose_rot = getGraphNodePose(graph_optimized, opt_idx)
        pose_trans = np.reshape(pose_trans, (-1, 3)).squeeze()
        pose_rot = np.reshape(pose_rot, (-1, 9)).squeeze()
        optimized_pose_ith = np.array([ pose_rot[0], pose_rot[1], pose_rot[2], pose_trans[0], 
                                        pose_rot[3], pose_rot[4], pose_rot[5], pose_trans[1], 
                                        pose_rot[6], pose_rot[7], pose_rot[8], pose_trans[2],
                                        0.0, 0.0, 0.0, 0.1 ])
        if(opt_idx == 0):
            optimized_pose_list = optimized_pose_ith
        else:
            optimized_pose_list = np.vstack((optimized_pose_list, optimized_pose_ith))

    np.savetxt(filename, optimized_pose_list, delimiter=",")


class PoseGraphResultSaver:
    def __init__(self, init_pose, save_gap, num_frames, seq_idx, save_dir):
        self.pose_list = np.reshape(init_pose, (-1, 16))
        self.save_gap = save_gap
        self.num_frames = num_frames

        self.seq_idx = seq_idx
        self.save_dir = save_dir

    def saveUnoptimizedPoseGraphResult(self, cur_pose, cur_node_idx):
        # save 
        self.pose_list = np.vstack((self.pose_list, np.reshape(cur_pose, (-1, 16))))

        # write
        if(cur_node_idx % self.save_gap == 0 or cur_node_idx == self.num_frames):        
            # save odometry-only poses
            filename = "pose" + self.seq_idx + "unoptimized_" + str(getUnixTime()) + ".csv"
            filename = os.path.join(self.save_dir, filename)
            np.savetxt(filename, self.pose_list, delimiter=",")

    def saveOptimizedPoseGraphResult(self, cur_node_idx, graph_optimized):
        filename = "pose" + self.seq_idx + "optimized_" + str(getUnixTime()) + ".csv"
        filename = os.path.join(self.save_dir, filename)
        saveOptimizedGraphPose(cur_node_idx, graph_optimized, filename)    
    
        optimized_pose_list = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
        self.pose_list = optimized_pose_list # update with optimized pose 

    def vizCurrentTrajectory(self, fig_idx, fig_save_path):
        x = self.pose_list[:,3]
        y = self.pose_list[:,7]
        z = self.pose_list[:,11]

        fig = plt.figure(fig_idx)
        plt.clf()
        plt.plot(x, z, color='blue') # kitti camera coord for clarity
        plt.axis('equal')
        plt.xlabel('x', labelpad=10)
        plt.ylabel('y', labelpad=10)
        plt.draw()
        plt.pause(0.01) #is necessary for the plot to update for some reason
        plt.savefig(fig_save_path, dpi=150)


class PoseGraphManager:
    def __init__(self):

        self.prior_cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10]))
        # self.const_cov_odo = np.array([2e-1, 2e-1, 2e-1, 1e-1, 1e-1, 1e-1])
        self.const_cov_odo = np.array([2e-1, 2e-1, 2e-1, 1e1, 1e1, 1e1])
        self.const_cov_loop = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]) 
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov_odo)
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov_loop)

        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_initials = gtsam.Values()

        self.opt_param = gtsam.LevenbergMarquardtParams()
        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, self.opt_param)

        self.curr_se3 = None
        self.curr_node_idx = None
        self.prev_node_idx = None

        self.graph_optimized = None

    def addPriorFactor(self):
        self.curr_node_idx = 0
        self.prev_node_idx = 0

        self.curr_se3 = np.eye(4)

        self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), gtsam.Pose3(self.curr_se3))
        self.graph_factors.add(gtsam.PriorFactorPose3(
                                                gtsam.symbol('x', self.curr_node_idx), 
                                                gtsam.Pose3(self.curr_se3), 
                                                self.prior_cov))

    def addOdometryFactor(self, odom_transform):

        self.graph_initials.insert(gtsam.symbol('x', self.curr_node_idx), gtsam.Pose3(self.curr_se3))
        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', self.prev_node_idx), 
                                                gtsam.symbol('x', self.curr_node_idx), 
                                                gtsam.Pose3(odom_transform), 
                                                self.odom_cov))

    def addLoopFactor(self, loop_transform, loop_idx):

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                        gtsam.symbol('x', loop_idx), 
                                        gtsam.symbol('x', self.curr_node_idx), 
                                        gtsam.Pose3(loop_transform), 
                                        self.loop_cov))

    def optimizePoseGraph(self):

        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, self.opt_param)
        self.graph_optimized = self.opt.optimize()

        # correct current pose 
        pose_trans, pose_rot = getGraphNodePose(self.graph_optimized, self.curr_node_idx)
        self.curr_se3[:3, :3] = pose_rot
        self.curr_se3[:3, 3] = pose_trans


def load_params(scene_path, optimize_keys=None):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}

    if optimize_keys is not None:
        keys = [k for k in all_params.keys() if k in optimize_keys]
    else:
        keys = [k for k in all_params.keys() if
                k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'log_scales', 'logit_opacities']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(all_params[k]).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(all_params[k].cuda().float().contiguous().requires_grad_(True))

    all_w2cs = []
    all_gt_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.detach().cpu().numpy())

        gt_w2c = params['gt_w2c_all_frames'][t_i]
        # print(gt_w2c.shape)
        all_gt_w2cs.append(gt_w2c.cpu().numpy())

    return all_w2cs, all_gt_w2cs, params


def load_params_with_overlap(part_path, optimize_keys=None, next_part_path=None, overlap_bound=0, stride=1):
    # Load Scene Data
    all_params = dict(np.load(part_path, allow_pickle=True))
    next_part_params = dict(np.load(next_part_path, allow_pickle=True))
    cat_gs_mask = (next_part_params['timestep'] > 0) & (next_part_params['timestep'] < overlap_bound//stride+1)
    for k in ['rgb_colors', 'unnorm_rotations', 'log_scales', 'logit_opacities']:
        all_params[k] = np.vstack((all_params[k], next_part_params[k][cat_gs_mask]))
    all_params['timestep'] = np.concatenate((all_params['timestep'], next_part_params['timestep'][cat_gs_mask]+all_params['timestep'][-1]))

    cam_rot = F.normalize(torch.tensor(all_params['cam_unnorm_rots'][..., -1]))
    cam_tran = torch.tensor(all_params['cam_trans'][..., -1])
    base_w2c = torch.eye(4).float()
    base_w2c[:3, :3] = build_rotation(cam_rot)
    base_w2c[:3, 3] = cam_tran
    base_c2w = torch.linalg.inv(base_w2c)
    base_gt_w2c = all_params['gt_w2c_all_frames'][-1]

    next_part_means3D = next_part_params['means3D'][cat_gs_mask]
    next_part_means3D = np.hstack((next_part_means3D, np.ones((next_part_means3D.shape[0], 1)))).T # (4, k)
    next_part_means3D = ((base_c2w @ next_part_means3D).T)[:, :3] # (k, 3)
    all_params['means3D'] = np.vstack((all_params['means3D'], next_part_means3D))

    for t_i in range(1, overlap_bound//stride+1):
        cam_rot = F.normalize(torch.tensor(next_part_params['cam_unnorm_rots'][..., t_i]))
        cam_tran = torch.tensor(next_part_params['cam_trans'][..., t_i])
        rel_w2c = torch.eye(4).float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        rel_w2c = rel_w2c @ base_w2c
        rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0)
        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
        rel_w2c_tran = rel_w2c[:3, 3]
        rel_w2c_rot_quat = rel_w2c_rot_quat.numpy().reshape(1, 4, 1)
        all_params['cam_unnorm_rots'] = np.concatenate((all_params['cam_unnorm_rots'], rel_w2c_rot_quat), axis=2)
        rel_w2c_tran = rel_w2c_tran.numpy().reshape(1, 3, 1)
        all_params['cam_trans'] = np.concatenate((all_params['cam_trans'], rel_w2c_tran), axis=2)

        gt_w2c = next_part_params['gt_w2c_all_frames'][t_i]
        gt_w2c = gt_w2c @ base_gt_w2c
        gt_w2c = gt_w2c.reshape(1, 4, 4)
        all_params['gt_w2c_all_frames'] = np.concatenate((all_params['gt_w2c_all_frames'], gt_w2c), axis=0)

    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}

    if optimize_keys is not None:
        keys = [k for k in all_params.keys() if k in optimize_keys]
    else:
        keys = [k for k in all_params.keys() if
                k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'log_scales', 'logit_opacities']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(all_params[k]).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(all_params[k].cuda().float().contiguous().requires_grad_(True))

    all_w2cs = []
    all_gt_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

        gt_w2c = params['gt_w2c_all_frames'][t_i]
        # print(gt_w2c.shape)
        all_gt_w2cs.append(gt_w2c.cpu().numpy())

    return all_w2cs, all_gt_w2cs, params


def load_imgs(image_folder_path, depth_folder_path, start_idx, end_idx, stride):
    color_paths = natsorted(glob.glob(f"{image_folder_path}/*.png"))
    depth_paths = natsorted(glob.glob(f"{depth_folder_path}/*.npy"))
    color_paths = color_paths[start_idx : end_idx : stride]
    depth_paths = depth_paths[start_idx : end_idx : stride]
    assert len(color_paths) == len(depth_paths)
    color_imgs = []
    depth_imgs = []
    for idx in range(len(color_paths)):
        color = np.asarray(imageio.imread(color_paths[idx]), dtype=float)
        depth = np.load(depth_paths[idx])
        if len(color.shape) == 2:
            color = np.dstack([color, color, color])
        color = torch.tensor(color, dtype=torch.float32) / 255.0
        color = torch.clamp(color, 0., 1.)
        color = color.permute(2, 0, 1)
        color_imgs.append(color)
        depth_imgs.append(depth)
    return color_imgs, depth_imgs


def load_imgs_from_path_list(image_paths, depth_paths):
    assert len(image_paths) == len(depth_paths)
    color_imgs = []
    depth_imgs = []
    for idx in range(len(image_paths)):
        color = np.asarray(imageio.imread(image_paths[idx]), dtype=float)
        depth = np.load(depth_paths[idx])
        if len(color.shape) == 2:
            color = np.dstack([color, color, color])
        color = torch.tensor(color, dtype=torch.float32) / 255.0
        color = torch.clamp(color, 0., 1.)
        color = color.permute(2, 0, 1)
        color_imgs.append(color)
        depth_imgs.append(depth)
    return color_imgs, depth_imgs


def align_pose_to_first_frame(c2ws, start_idx, end_idx, stride=1):
    aligned_c2ws = []
    first_w2c = None
    for i in range(start_idx, end_idx, stride):
        if i == start_idx:
            first_w2c = np.linalg.inv(c2ws[i])
            aligned_c2ws.append(np.eye(4))
            continue
        aligned_c2ws.append(first_w2c @ c2ws[i])
    return np.array(aligned_c2ws)


print_style = {
    'info': 'bold green',
    'eval': 'bold red',
    'gaussian': 'bold blue',
}
def Log(*args, tag):
    style = print_style[tag]
    rich.print(f"[{style}]{'Stereo Gaussian SLAM'}:[/{style}]", *args)


def align_pose_2_first_frame(c2ws):
    result_c2ws = torch.ones_like(c2ws, dtype=c2ws.dtype, device=c2ws.device)
    for i in range(c2ws.shape[0]):
        if i == 0:
            first_c2w = c2ws[i]
            first_w2c = torch.linalg.inv(first_c2w)
            result_c2ws[0] = torch.eye(4, dtype=c2ws.dtype, device=c2ws.device)
        else:
            result_c2ws[i] = first_w2c @ c2ws[i]
    return result_c2ws


def compute_min_scale_loss(params):
    row_min_values, _ = torch.min(torch.exp(params['log_scales']), dim=1)  
    sum_of_row_min = row_min_values.mean()  
    return sum_of_row_min


def compute_scene_radius(trajs):
    cam_centers = []
    for i in range(trajs.shape[0]):
        cam_centers.append(trajs[i, :3, 3:4])
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    radius = diagonal * 1.1
    return radius


def fill_holes(source_cloud, target_cloud):
    source_tree = o3d.geometry.KDTreeFlann(source_cloud)  

    new_points = []
    new_points_color = []

    for i in tqdm(range(len(target_cloud.points))):  
        [k, idx, _] = source_tree.search_radius_vector_3d(target_cloud.points[i], 0.1)  
        if k < 3:  
            new_points.append(target_cloud.points[i])
            new_points_color.append(target_cloud.colors[i])

    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(new_points)
    tmp_pcd.colors = o3d.utility.Vector3dVector(new_points_color)
    tmp_pcd = tmp_pcd.random_down_sample(1.0 / 4.0)

    return source_cloud + tmp_pcd, tmp_pcd


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["kitti"]:
        return KittiDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["euroc"]:
        return EurocDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


if __name__ == "__main__":

    base_folder = 'MH_01_easy' # path to folder with loop closure results
    scene_name = 'MH_01_easy'
    dataset_type = 'euroc' # kitti or euroc
    if dataset_type == 'kitti':
        kitti_base_folder = '' 
        image_folder_path = os.path.join(kitti_base_folder, scene_name, 'image_2')
        depth_folder_path = os.path.join(kitti_base_folder, scene_name, 'depth_sceneflow')
    elif dataset_type == 'euroc':
        from configs.euroc.lsgslam import config
        dataset_config = config["data"]
        dataset_config['basedir'] = f"euroc/{scene_name}/mav0/cam0"
        dataset_config['sequence'] = scene_name
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    ba = False
    if ba:
        optimize_keys = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales', 'cam_trans', 'cam_unnorm_rots']
    else:
        optimize_keys = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']

    overlap = False
    overlap_bound = 20

    structure_refine_total_iters = 5000 # 2000, 5000
    structure_refine_lrs=dict(
        means3D=0.0008, # 0.0001 in euroc, 0.0008 in kitti
        rgb_colors=0.0025,
        unnorm_rotations=0.001,
        logit_opacities=0.05,
        log_scales=0.001 * 6, 
        cam_unnorm_rots=0.0000,
        cam_trans=0.000,
    )

    structure_refine_lrs_decay1=dict(
        means3D=0.0001, # 0.0001 in euroc, 0.0008 in kitti
        rgb_colors=0.0025,
        unnorm_rotations=0.001,
        logit_opacities=0.05,
        log_scales=0.001 * 3, 
        cam_unnorm_rots=0.0000,
        cam_trans=0.000,
    )

    structure_refine_lrs_decay2=dict(
        means3D=0.0001, # 0.0001 in euroc, 0.0008 in kitti
        rgb_colors=0.0025,
        unnorm_rotations=0.001,
        logit_opacities=0.05,
        log_scales=0.001, 
        cam_unnorm_rots=0.0000,
        cam_trans=0.000,
    )

    sil_thres = 0.99 
    save_rendering_every = 1

    use_min_scale_loss = True
    min_scale_loss_warmup_iters = 200
    min_scale_loss_weight = 1.0

    pixel_gs_depth_gamma = 0.37 

    depth_filter_near = 0.1
    depth_filter_far = 30.0

    gaussians_distribution = 'anisotropic' # isotropic or anisotropic

    use_densify = False
    split_explore_weight = 1.0
    densify_dict=dict( # Needs to be updated based on the number of mapping iterations
        start_after=0.02*structure_refine_total_iters, 
        remove_big_after=0.1*structure_refine_total_iters, 
        stop_after=0.9*structure_refine_total_iters, 
        densify_every=70, 
        grad_thresh=0.0002, 
        num_to_split_into=2,
        removal_opacity_threshold=0.005,
        final_removal_opacity_threshold=0.005,
        reset_opacities_every=2*structure_refine_total_iters, 
    )

    # Rendering result saver
    rendering_save_dir = os.path.join(base_folder, 'RenderingResult')
    save_render_rgb_dir = os.path.join(rendering_save_dir, 'after_opt_render_rgb')
    save_gt_rgb_dir = os.path.join(rendering_save_dir, 'gt_rgb')
    save_tmp_rgb_dir = os.path.join(rendering_save_dir, 'before_opt_render_rgb')
    os.makedirs(rendering_save_dir, exist_ok=True)
    os.makedirs(save_render_rgb_dir, exist_ok=True)
    os.makedirs(save_gt_rgb_dir, exist_ok=True)
    os.makedirs(save_tmp_rgb_dir, exist_ok=True)

    # Result saver
    save_dir = os.path.join(base_folder, 'PoseGraphResult')
    fig_save_dir = os.path.join(save_dir, 'figures')
    csv_save_dir = os.path.join(save_dir, 'csvs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_save_dir, exist_ok=True)
    os.makedirs(csv_save_dir, exist_ok=True)

    res_folders = os.listdir(base_folder)
    odo_res_folders = []
    loop_res_folders = []
    for res_folder in res_folders:
        if scene_name not in res_folder:
            continue
        if 'loop' in res_folder:
            loop_res_folders.append(res_folder)
        else:
            odo_res_folders.append(res_folder)
    
    assert len(loop_res_folders) == 1, "should only has one loop_res_folder"
    loop_res_folder = loop_res_folders[0]

    odo_res_folders = sorted(odo_res_folders, key=lambda x:int(x.split('_')[-3]))
    odo_res_folders = odo_res_folders[:]
    print(odo_res_folders)

    all_odo_est_w2cs = []
    all_odo_gt_w2cs = []
    for odo_res_folder in odo_res_folders:
        print(odo_res_folder)
        info = odo_res_folder.split('_')
        start_idx = int(info[-3])
        end_idx = int(info[-2])
        odo_scene_path = os.path.join(base_folder, odo_res_folder, 'params.npz')
        odo_est_w2cs, odo_gt_w2cs, _ = load_params(odo_scene_path)
        if len(all_odo_est_w2cs) == 0:
            all_odo_est_w2cs = odo_est_w2cs
            all_odo_gt_w2cs = odo_gt_w2cs
        else:
            last_odo_est_pose = all_odo_est_w2cs[-1]
            last_odo_gt_pose = all_odo_gt_w2cs[-1]
            odo_est_w2cs = [pose @ last_odo_est_pose for pose in odo_est_w2cs[1:]]
            odo_gt_w2cs = [pose @ last_odo_gt_pose for pose in odo_gt_w2cs[1:]]
            all_odo_est_w2cs += odo_est_w2cs
            all_odo_gt_w2cs += odo_gt_w2cs

    print(len(all_odo_est_w2cs))
    print(len(all_odo_gt_w2cs))
    num_frames = len(all_odo_est_w2cs)

    found_loops_path = os.path.join(base_folder, loop_res_folder, 'found_loops.npy')
    found_loops = np.load(found_loops_path)
    loop_infos = {}
    for loop in found_loops:
        query_kf_idx = loop[0]
        ref_kf_idx = loop[1]

        loop_inlier_path = os.path.join(base_folder, loop_res_folder, 'eval_{}_{}'.format(query_kf_idx, ref_kf_idx), 'match_res/1_inliers.txt')
        if not os.path.exists(loop_inlier_path):
            continue
        num_inliers = float(open(loop_inlier_path, 'r').readlines()[0])
        if num_inliers < 100:
            continue
        loop_scene_path = os.path.join(base_folder, loop_res_folder, 'eval_{}_{}'.format(query_kf_idx, ref_kf_idx), 'params.npz')
        loop_est_r2qs, loop_gt_r2qs, _ = load_params(loop_scene_path)

        loop_infos[query_kf_idx] = [ref_kf_idx, loop_est_r2qs[1], loop_gt_r2qs[1]]

    # Pose Graph Manager (for back-end optimization) initialization
    PGM = PoseGraphManager()
    PGM.addPriorFactor()

    ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3, 
                                save_gap=1,
                                num_frames=len(all_odo_est_w2cs),
                                seq_idx=scene_name,
                                save_dir=csv_save_dir)

    # for save the results as a video
    fig_idx = 1
    fig = plt.figure(fig_idx)
    num_frames_to_skip_to_show = 1
    
    for odo_idx, odo_est_w2c in tqdm(enumerate(all_odo_est_w2cs), total=num_frames, mininterval=5.0):
        # save current node
        PGM.curr_node_idx = odo_idx # make start with 0
        if(PGM.curr_node_idx == 0):
            PGM.prev_node_idx = PGM.curr_node_idx
            continue
        
        odo_transform = all_odo_est_w2cs[odo_idx - 1] @ np.linalg.inv(odo_est_w2c)

        # update the current (moved) pose 
        PGM.curr_se3 = PGM.curr_se3 @ odo_transform

        # add the odometry factor to the graph 
        PGM.addOdometryFactor(odo_transform)

        # renewal the prev information 
        PGM.prev_node_idx = PGM.curr_node_idx

        # loop detection and optimize the graph 
        if odo_idx in loop_infos:
            loop_idx = loop_infos[odo_idx][0]
            loop_est_r2qs = loop_infos[odo_idx][1]
            loop_gt_r2qs = loop_infos[odo_idx][2]
            loop_transform = np.linalg.inv(loop_est_r2qs)

            odo_gt_w2c = all_odo_gt_w2cs[odo_idx]
            loop_gt_w2c = all_odo_gt_w2cs[loop_idx]
            gt_loop_transform = loop_gt_w2c @ np.linalg.inv(odo_gt_w2c)

            PGM.addLoopFactor(loop_transform, loop_idx)

            # 2-2/ graph optimization 
            PGM.optimizePoseGraph()

            # 2-2/ save optimized poses
            ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)


        # save the ICP odometry pose result (no loop closure)
        ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx) 
        if(odo_idx % num_frames_to_skip_to_show == 0): 
            fig_save_path = os.path.join(fig_save_dir, '{}.png'.format(odo_idx))
            ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx, fig_save_path=fig_save_path)

    loop_trajs = ResultSaver.pose_list.reshape([-1, 4, 4])
    gt_traj_pts = []
    est_traj_pts = []
    loop_traj_pts = [[0.0, 0.0, 0.0]]
    for i in range(num_frames):
        gt_c2w_trans = np.linalg.inv(all_odo_gt_w2cs[i])[:3, 3].flatten()
        est_c2w_trans = np.linalg.inv(all_odo_est_w2cs[i])[:3, 3].flatten()
        gt_traj_pts.append(gt_c2w_trans)
        est_traj_pts.append(est_c2w_trans)
        if i < num_frames - 1:
            loop_c2w_trans = loop_trajs[i, :3, 3].flatten()
            loop_traj_pts.append(loop_c2w_trans)

    gt_traj_pts = np.array(gt_traj_pts).T
    est_traj_pts = np.array(est_traj_pts).T
    loop_traj_pts = np.array(loop_traj_pts).T
    print(gt_traj_pts.shape)
    print(est_traj_pts.shape)
    print(loop_traj_pts.shape)

    # evaluate optimized poses
    print('Align results')
    _, _, trans_error = align(gt_traj_pts, est_traj_pts, align_traj=True)
    est_align_avg_trans_error = round(trans_error.mean(), 4)
    print(est_align_avg_trans_error)
    _, _, trans_error = align(gt_traj_pts, loop_traj_pts, align_traj=True)
    loop_align_avg_trans_error = round(trans_error.mean(), 4)
    print(loop_align_avg_trans_error)

    print('Absolute results')
    _, _, trans_error = align(gt_traj_pts, est_traj_pts, align_traj=False)
    est_abs_avg_trans_error = round(trans_error.mean(), 4)
    print(est_abs_avg_trans_error)
    _, _, trans_error = align(gt_traj_pts, loop_traj_pts, align_traj=False)
    loop_abs_avg_trans_error = round(trans_error.mean(), 4)
    print(loop_abs_avg_trans_error)

    fig = plt.figure(10)
    plt.clf()
    plt.plot(gt_traj_pts[0, :], gt_traj_pts[2, :], color='blue') # kitti camera coord for clarity
    plt.plot(est_traj_pts[0, :], est_traj_pts[2, :], color='red') # kitti camera coord for clarity
    plt.plot(loop_traj_pts[0, :], loop_traj_pts[2, :], color='green') # kitti camera coord for clarity
    plt.axis('equal')
    plt.xlabel('x', labelpad=10)
    plt.ylabel('y', labelpad=10)
    plt.legend(['gt_align_absolute', 
                'odo_{}_{}'.format(est_align_avg_trans_error, est_abs_avg_trans_error), 
                'loop_{}_{}'.format(loop_align_avg_trans_error, loop_abs_avg_trans_error)])
    # plt.show()
    # plt.pause(0.01) #is necessary for the plot to update for some reason    
    fig_save_path = os.path.join(save_dir, 'traj_compare.png')
    plt.savefig(fig_save_path, dpi=150)

    traj_images = [img for img in os.listdir(fig_save_dir) if img.endswith(".png")]
    traj_images = sorted(traj_images, key=lambda x:int(x.split('.')[0]))

    frame = cv2.imread(os.path.join(fig_save_dir, traj_images[0]))
    height, width, layers = frame.shape

    video_name = os.path.join(save_dir, "odo_with_loop.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps=30, frameSize=(width, height))

    for image in traj_images:
        video.write(cv2.imread(os.path.join(fig_save_dir, image)))

    video.release()

    # pose optimization end, begin to optimize map
    for j in range(loop_trajs.shape[0]):
        if loop_trajs[j, 3, 3] < 1.0:
            loop_trajs[j, 3, 3] = 1.0
    first_frame_c2w = np.eye(4)
    first_frame_c2w = np.expand_dims(first_frame_c2w, axis=0)
    loop_trajs = np.concatenate((first_frame_c2w, loop_trajs), axis=0)

    pixel_gs_scene_radius = compute_scene_radius(loop_trajs)

    before_opt_psnr_list = []
    before_opt_ssim_list = []
    before_opt_lpips_list = []

    after_opt_psnr_list = []
    after_opt_ssim_list = []
    after_opt_lpips_list = []

    for part_index in range(len(odo_res_folders)):

        odo_res_folder = odo_res_folders[part_index]
        is_last_part = odo_res_folder == odo_res_folders[-1]
        info = odo_res_folder.split('_')
        start_idx = int(info[-3])
        if part_index == 0:
            total_start_idx = start_idx
        if not is_last_part and overlap:
            end_idx = int(info[-2]) + overlap_bound
            true_end_idx = int(info[-2])
        else:
            end_idx = int(info[-2])
        stride = int(info[-1])
        Log('structure refine part {:04d} to {:04d}'.format(start_idx, end_idx), tag='info')
        odo_scene_path = os.path.join(base_folder, odo_res_folder, 'params.npz')        
        if not is_last_part and overlap:
            next_part_path = os.path.join(base_folder, odo_res_folders[part_index+1], 'params.npz')
            before_opt_w2cs, gt_w2cs, params = load_params_with_overlap(odo_scene_path, optimize_keys, next_part_path, overlap_bound, stride)
        else:
            before_opt_w2cs, gt_w2cs, params = load_params(odo_scene_path, optimize_keys)

        if dataset_type == 'kitti':
            gt_imgs, depths = load_imgs(image_folder_path, depth_folder_path, start_idx, end_idx + 1, stride)
        elif dataset_type == 'euroc':
            euroc_dataset = get_dataset(
                config_dict=gradslam_data_cfg,
                basedir=dataset_config["basedir"],
                sequence=os.path.basename(dataset_config["sequence"]),
                start=start_idx,
                end=end_idx,
                stride=stride,
                desired_height=dataset_config["desired_image_height"],
                desired_width=dataset_config["desired_image_width"],
                device='cuda',
                relative_pose=True,
            )
            gt_imgs, depths = load_imgs_from_path_list(euroc_dataset.color_paths, euroc_dataset.depth_paths)

        intrinsics = params['intrinsics'].cpu().numpy()
        assert intrinsics.shape == (3, 3)
        variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                    'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                    'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
        variables['scene_radius'] = torch.tensor(30.0)/5.0
        variables['timestep'] = params['timestep']
        useless_params = {k: v for k, v in params.items() if k in ['intrinsics', 'org_height', 'org_width', 'w2c', 'gt_w2c_all_frames']}
        params = {k: v for k, v in params.items() if k in optimize_keys}

        if params['log_scales'].shape[1] == 1:
            if gaussians_distribution == 'anisotropic':
                params['log_scales'] = torch.nn.Parameter(torch.tile(params['log_scales'], (1, 3)).cuda().float().contiguous().requires_grad_(True))

        with torch.no_grad():
            for i in tqdm(range(len(before_opt_w2cs)), 'Eval before structure refine...'):
                if overlap:
                    if i > (true_end_idx - start_idx)//stride:
                        break

                color_gt = gt_imgs[i].to('cuda') # (C, H, W) [0, 1] torch.Tensor
                depth = depths[i]
                depth = torch.tensor(depth, dtype=torch.float32, device='cuda')
                depth_mask = (depth < depth_filter_far) & (depth > depth_filter_near)

                w2c = before_opt_w2cs[i]
                cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, w2c, depth_threshold=pixel_gs_depth_gamma*pixel_gs_scene_radius)
                rendervar = transformed_params2rendervar(params, params)
                w2c = torch.tensor(w2c, dtype=torch.float32, device='cuda')
                depth_sil_rendervar = transformed_params2depthplussilhouette(params, w2c,
                                                                         params)
                im, radius, _, _ = Renderer(raster_settings=cam)(**rendervar)
                # Render Depth & Silhouette
                depth_sil, _, _, _ = Renderer(raster_settings=cam)(**depth_sil_rendervar)
                silhouette = depth_sil[1, :, :]
                presence_sil_mask = (silhouette > sil_thres)

                weighted_im = im * presence_sil_mask * depth_mask
                weighted_gt_im = color_gt * presence_sil_mask * depth_mask
                psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
                ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                                data_range=1.0, size_average=True)
                lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                            torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

                viz_render_im = torch.clamp(im, 0, 1)
                viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
                cv2.imwrite(os.path.join(save_tmp_rgb_dir, "{:04d}_{:04d}_rgb.png".format(start_idx, i*stride)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_tmp_rgb_dir, "{:04d}_{:04d}_smask.png".format(start_idx, i*stride)), (presence_sil_mask.detach()*255).byte().cpu().numpy())
                cv2.imwrite(os.path.join(save_tmp_rgb_dir, "{:04d}_{:04d}_dmask.png".format(start_idx, i*stride)), (depth_mask*255).byte().cpu().numpy())

                before_opt_psnr_list.append(psnr.cpu().numpy())
                before_opt_ssim_list.append(ssim.cpu().numpy())
                before_opt_lpips_list.append(lpips_score)

        # deformation
        pre_est_w2cs = torch.tensor(np.array(before_opt_w2cs), dtype=torch.float32, device='cuda') # (k, 4, 4)
        every_gs_pre_w2cs = pre_est_w2cs[variables['timestep'].to(torch.long)] # (n, 4, 4)
        tmp = torch.ones(params['means3D'].shape[0], 1, dtype=torch.float32, device='cuda')
        gs_means3d = torch.cat((params['means3D'], tmp), dim=1) # (n, 4)
        gs_camera = torch.bmm(every_gs_pre_w2cs, gs_means3d.unsqueeze_(-1)).squeeze_(-1) # (n, 4)

        optim_c2ws = loop_trajs[(start_idx-total_start_idx)//stride : (end_idx-total_start_idx)//stride+1]
        after_est_c2ws = torch.tensor(optim_c2ws, dtype=torch.float32, device='cuda') # (k, 4, 4)
        if ba:
            after_est_c2ws = align_pose_2_first_frame(after_est_c2ws)
        every_gs_aft_c2ws = after_est_c2ws[variables['timestep'].to(torch.long)] # (n, 4, 4)
        gs_new_means3d = torch.bmm(every_gs_aft_c2ws, gs_camera.unsqueeze_(-1)).squeeze_(-1) # (n, 4)
        params['means3D'] = torch.nn.Parameter(gs_new_means3d[:, :3].cuda().float().contiguous().requires_grad_(True))


        # save the pc when need to debug
        now_pc = o3d.geometry.PointCloud()
        now_pc.points = o3d.utility.Vector3dVector(params['means3D'].detach().cpu().numpy())
        now_pc.colors = o3d.utility.Vector3dVector(params['rgb_colors'].detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(rendering_save_dir, 'after_deformation.pcd'), now_pc)

        if ba:
            for i in range(optim_c2ws.shape[0]):
                if i == 0:
                    first_w2c = np.linalg.inv(optim_c2ws[i])
                    with torch.no_grad():
                        rel_w2c = torch.nn.Parameter(torch.tensor(np.eye(4), dtype=torch.float32, device='cuda', requires_grad=True))
                        rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0)
                        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                        rel_w2c_tran = rel_w2c[:3, 3]
                        params['cam_unnorm_rots'][..., i] = rel_w2c_rot_quat
                        params['cam_trans'][..., i] = rel_w2c_tran
                else:
                    c2first = first_w2c @ optim_c2ws[i]
                    with torch.no_grad():
                        rel_w2c = torch.nn.Parameter(torch.tensor(np.linalg.inv(c2first), dtype=torch.float32, device='cuda', requires_grad=True))
                        rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0)
                        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                        rel_w2c_tran = rel_w2c[:3, 3]
                        params['cam_unnorm_rots'][..., i] = rel_w2c_rot_quat
                        params['cam_trans'][..., i] = rel_w2c_tran

        # structure refine 
        param_groups = [{'params': [v], 'name': k, 'lr': structure_refine_lrs[k]} for k, v in params.items() if k in optimize_keys]
        optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        for i in tqdm(range(structure_refine_total_iters), 'Color refining...'):

            index = random.randint(0, optim_c2ws.shape[0]-1)
            color_gt = gt_imgs[index].to('cuda') # (C, H, W) [0, 1] torch.Tensor
            if not ba:
                w2c = np.linalg.inv(optim_c2ws[index])
                cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, w2c, depth_threshold=pixel_gs_depth_gamma*pixel_gs_scene_radius)
                rendervar = transformed_params2rendervar(params, params)
            else:
                cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, np.eye(4), depth_threshold=pixel_gs_depth_gamma*pixel_gs_scene_radius)
                transformed_gaussians = transform_to_frame(params, index,
                                                    gaussians_grad=True,
                                                    camera_grad=True)
                rendervar = transformed_params2rendervar(params, transformed_gaussians)

            rendervar['means2D'].retain_grad()
            im, radius, _, pixels, = Renderer(raster_settings=cam)(**rendervar)
            variables['means2D'] = rendervar['means2D']  # Gradient only accum from color render for densification
            loss = 0.8 * l1_loss_v1(im, color_gt) + 0.2 * (1.0 - calc_ssim(im, color_gt))

            if use_min_scale_loss and gaussians_distribution == 'anisotropic' and i > min_scale_loss_warmup_iters:
                scale_loss = compute_min_scale_loss(params)
                loss += min_scale_loss_weight * scale_loss
            # print('Color_SSIM Loss: ' + str(loss), 'Scale Loss: ' + str(scale_loss))

            seen = radius > 0
            variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
            variables['seen'] = seen

            loss.backward()

            with torch.no_grad():

                if use_densify:
                    # params, variables = densify_with_bound(params, variables, optimizer, i, densify_dict, depth_filter_far, optim_c2ws)
                    params, variables = densify_use_pixel_gs(params, variables, optimizer, i, densify_dict, pixels, split_explore_weight)
                    if i <= densify_dict['stop_after'] and i % (structure_refine_total_iters//10) == 0:
                        Log("Number of Gaussians at iter {:05d}: ".format(i) + str(params['means3D'].shape[0]), tag='gaussian')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                
            # TODO  lr update

        # eval structure refine
        with torch.no_grad():
            for i in tqdm(range(optim_c2ws.shape[0]), 'Eval mapping result...'):
                if overlap:
                    if i > (true_end_idx - start_idx)//stride:
                        break
                color_gt = gt_imgs[i].to('cuda') # (C, H, W) [0, 1] torch.Tensor
                depth = depths[i]
                depth = torch.tensor(depth, dtype=torch.float32, device='cuda')
                depth_mask = (depth < depth_filter_far) & (depth > depth_filter_near)

                if not ba:
                    w2c = np.linalg.inv(optim_c2ws[i])
                    cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, w2c, depth_threshold=pixel_gs_depth_gamma*pixel_gs_scene_radius)
                    rendervar = transformed_params2rendervar(params, params)
                    w2c = torch.tensor(w2c, dtype=torch.float32, device='cuda')
                    depth_sil_rendervar = transformed_params2depthplussilhouette(params, w2c,
                                                                            params)
                else:
                    cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, np.eye(4), depth_threshold=pixel_gs_depth_gamma*pixel_gs_scene_radius)
                    transformed_gaussians = transform_to_frame(params, i,
                                                    gaussians_grad=False,
                                                    camera_grad=False)
                    rendervar = transformed_params2rendervar(params, transformed_gaussians)
                    depth_sil_rendervar = transformed_params2depthplussilhouette(params, torch.eye(4, dtype=torch.float32, device='cuda'),
                                                                    transformed_gaussians)

                im, radius, _, _ = Renderer(raster_settings=cam)(**rendervar)
                # Render Depth & Silhouette
                depth_sil, _, _, _ = Renderer(raster_settings=cam)(**depth_sil_rendervar)
                silhouette = depth_sil[1, :, :]
                presence_sil_mask = (silhouette > sil_thres)

                weighted_im = im * presence_sil_mask * depth_mask
                weighted_gt_im = color_gt * presence_sil_mask * depth_mask
                psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
                ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                                data_range=1.0, size_average=True)
                lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                            torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

                after_opt_psnr_list.append(psnr.cpu().numpy())
                after_opt_ssim_list.append(ssim.cpu().numpy())
                after_opt_lpips_list.append(lpips_score)

                if i % save_rendering_every == 0:
                    # Save Rendered RGB
                    viz_render_im = torch.clamp(im, 0, 1)
                    viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
                    cv2.imwrite(os.path.join(save_render_rgb_dir, "{:04d}_{:04d}_rgb.png".format(start_idx, i*stride)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_render_rgb_dir, "{:04d}_{:04d}_smask.png".format(start_idx, i*stride)), (presence_sil_mask.detach()*255).byte().cpu().numpy())
                    cv2.imwrite(os.path.join(save_render_rgb_dir, "{:04d}_{:04d}_dmask.png".format(start_idx, i*stride)), (depth_mask*255).byte().cpu().numpy())

                    # Save GT RGB 
                    viz_gt_im = torch.clamp(color_gt, 0, 1)
                    viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
                    cv2.imwrite(os.path.join(save_gt_rgb_dir, "{:04d}_{:04d}.png".format(start_idx, i*stride)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))

        for k, v in useless_params.items():
            params[k] = v
        save_params(params, rendering_save_dir)
        print(pixel_gs_depth_gamma*pixel_gs_scene_radius)

    before_opt_psnr_list = np.array(before_opt_psnr_list)
    before_opt_ssim_list = np.array(before_opt_ssim_list)
    before_opt_lpips_list = np.array(before_opt_lpips_list)
    before_avg_psnr = before_opt_psnr_list.mean()
    before_avg_ssim = before_opt_ssim_list.mean()
    before_avg_lpips = before_opt_lpips_list.mean()
    Log("Before structure refine average PSNR: {:.3f}".format(before_avg_psnr), tag='eval')
    Log("Before structure refine average MS-SSIM: {:.3f}".format(before_avg_ssim), tag='eval')
    Log("Before structure refine average LPIPS: {:.3f}".format(before_avg_lpips), tag='eval')
    np.savetxt(os.path.join(rendering_save_dir, 'before_opt_psnr.txt'), before_opt_psnr_list)
    np.savetxt(os.path.join(rendering_save_dir, 'before_opt_ms_ssim.txt'), before_opt_ssim_list)
    np.savetxt(os.path.join(rendering_save_dir, 'before_opt_lpips.txt'), before_opt_lpips_list)

    after_opt_psnr_list = np.array(after_opt_psnr_list)
    after_opt_ssim_list = np.array(after_opt_ssim_list)
    after_opt_lpips_list = np.array(after_opt_lpips_list)
    after_avg_psnr = after_opt_psnr_list.mean()
    after_avg_ssim = after_opt_ssim_list.mean()
    after_avg_lpips = after_opt_lpips_list.mean()
    Log("After structure refine average PSNR: {:.3f}".format(after_avg_psnr), tag='eval')
    Log("After structure refine average MS-SSIM: {:.3f}".format(after_avg_ssim), tag='eval')
    Log("After structure refine average LPIPS: {:.3f}".format(after_avg_lpips), tag='eval')
    np.savetxt(os.path.join(rendering_save_dir, 'after_opt_psnr.txt'), after_opt_psnr_list)
    np.savetxt(os.path.join(rendering_save_dir, 'after_opt_ms_ssim.txt'), after_opt_ssim_list)
    np.savetxt(os.path.join(rendering_save_dir, 'after_opt_lpips.txt'), after_opt_lpips_list)

    with open(os.path.join(rendering_save_dir, 'avg_metrics.txt'), 'w', encoding='utf-8') as file:
        file.write("Before structure refine average PSNR: {:.3f}".format(before_avg_psnr))
        file.write("Before structure refine average MS-SSIM: {:.3f}".format(before_avg_ssim))
        file.write("Before structure refine average LPIPS: {:.3f}".format(before_avg_lpips))
        file.write("After structure refine average PSNR: {:.3f}".format(after_avg_psnr))
        file.write("After structure refine average MS-SSIM: {:.3f}".format(after_avg_ssim))
        file.write("After structure refine average LPIPS: {:.3f}".format(after_avg_lpips))