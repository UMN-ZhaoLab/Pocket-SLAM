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
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import sys
import glob
import imageio
import torchvision.transforms as transforms
from PIL import Image
current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(current_dir)
from natsort import natsorted
from matplotlib.animation import FFMpegWriter
from utils.slam_external import build_rotation, calc_ssim, calc_psnr
from utils.eval_helpers import align
from importlib.machinery import SourceFileLoader
from pyquaternion import Quaternion

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
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
# from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
# from evo.tools import plot
# from evo.tools.plot import PlotMode
# from evo.tools.settings import SETTINGS

from kitti_odometry import KittiEvalOdom


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

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

from utils.recon_helpers import setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.slam_helpers import transformed_params2rendervar, l1_loss_v1, transformed_params2depthplussilhouette
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
        # self.const_cov_odo = np.array([2e-1, 2e-1, 2e-1, 1e1, 1e1, 1e1])
        self.const_cov_odo = np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1])
        self.const_cov_loop = np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1]) 
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

        # status = self.opt.optimize(self.graph_factors, self.graph_initials, self.graph_optimized)
        # if status != minisam.NonlinearOptimizationStatus.SUCCESS:
            # print("optimization error: ", status)

        # correct current pose 
        pose_trans, pose_rot = getGraphNodePose(self.graph_optimized, self.curr_node_idx)
        self.curr_se3[:3, :3] = pose_rot
        self.curr_se3[:3, 3] = pose_trans


def load_poses(scene_path):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices', 'timestep']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(all_params[k]).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(all_params[k].cuda().float().contiguous().requires_grad_(True))

    all_w2cs = []
    all_gt_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    with torch.no_grad():
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

def load_imgs(folder_path):
    color_paths = natsorted(glob.glob(f"{folder_path}/*.png"))
    color_paths = color_paths[ :  : 2]
    imgs = []
    for path in color_paths:
        color = np.asarray(imageio.imread(path), dtype=float)
        color = torch.tensor(color, dtype=torch.float32) / 255.0
        color = torch.clamp(color, 0., 1.)
        color = color.permute(2, 0, 1)
        imgs.append(color)
    return imgs

def evaluate_evo(poses_gt, poses_est, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)

    # print(len(poses_gt))
    # print(len(poses_est))
    # print(poses_est[0].shape)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    print("Align RMSE ATE \[m]", ape_stat)

    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    print("Absolute RMSE ATE \[m]", ape_stat)

    return ape_stat


if __name__ == "__main__":

    base_folder = ''
    scene_name = 'V2_03_difficult'
    # base_folder = ''
    # scene_name = '03'
    # base_folder = ''
    # scene_name = 'freiburg3_long_office_household'

    dataset = None
    if 'kitti' in base_folder:
        dataset = 'kitti'
    elif 'euroc' in base_folder:
        dataset = 'euroc'
    elif 'TUM' in base_folder:
        dataset = 'tum'
    else:
        raise NotImplementedError("Unknown dataset")

    not_concat_key = ['intrinsics', 'org_height', 'org_width', 'w2c']
    optimize_keys = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']

    color_refine_total_iters = 11000
    color_refine_lrs=dict(
        means3D=0.0001,
        rgb_colors=0.0025,
        unnorm_rotations=0.001,
        logit_opacities=0.05,
        log_scales=0.001,
        cam_unnorm_rots=0.0000,
        cam_trans=0.0000,
    )
    sil_thres = 0.5
    save_rendering_every = 1
    save_render_rgb_dir = ''
    save_gt_rgb_dir = ''
    save_tmp_rgb_dir = ''


    image_folder_path = ''
    gt_imgs = load_imgs(image_folder_path)

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
    
    if len(loop_res_folders) == 0:
        loop_res_folder = None
    else:    
        assert len(loop_res_folders) == 1, "should only has one loop_res_folder"
        loop_res_folder = loop_res_folders[0]

    odo_res_folders = sorted(odo_res_folders, key=lambda x:int(x.split('_')[-3]))
    odo_res_folders = odo_res_folders[:]
    print(odo_res_folders)

    all_odo_est_w2cs = []
    all_odo_gt_w2cs = []
    all_params = {}
    for odo_res_folder in odo_res_folders:
        print(odo_res_folder)
        info = odo_res_folder.split('_')
        start_idx = int(info[-3])
        end_idx = int(info[-2])
        odo_scene_path = os.path.join(base_folder, odo_res_folder, 'params.npz')
        odo_est_w2cs, odo_gt_w2cs, params = load_poses(odo_scene_path)
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
        if all_params == {}:
            all_params = params
            last_final_timestep = params['timestep'][-1].item()
            last_final_c2w = np.linalg.inv(odo_est_w2cs[-1])
        else:
            for k, v in params.items():
                if k in optimize_keys:
                    if k == 'means3D':
                        v = torch.cat((v, torch.ones(v.shape[0], 1, dtype=torch.float32, device='cuda')), dim=1)
                        v = torch.tensor(last_final_c2w, dtype=torch.float32, device='cuda') @ v.T
                        v = (v.T)[:, :3]
                        last_final_c2w = np.linalg.inv(all_odo_est_w2cs[-1])
                    all_params[k] = torch.nn.Parameter(torch.cat((all_params[k], v), dim=0).requires_grad_(True))
                elif k not in not_concat_key:
                    if k == 'timestep':
                        v.add_(last_final_timestep)
                        last_final_timestep = v[-1].item()
                    all_params[k] = torch.cat((all_params[k], v), dim=0)

    # all_odo_est_w2cs = all_odo_gt_w2cs
    print(len(all_odo_est_w2cs))
    print(len(all_odo_gt_w2cs))
    num_frames = len(all_odo_est_w2cs)

    loop_infos = {}
    if loop_res_folder is not None:
        found_loops_path = os.path.join(base_folder, loop_res_folder, 'found_loops.npy')
        found_loops = np.load(found_loops_path)
        
        for loop in found_loops:
            query_kf_idx = loop[0]
            # if query_kf_idx not in [417, 525]:
            #     continue
            # if query_kf_idx in [481]:
            #     continue
            ref_kf_idx = loop[1]

            # if abs(query_kf_idx - ref_kf_idx) < 50:
            #     continue

            loop_inlier_path = os.path.join(base_folder, loop_res_folder, 'eval_{}_{}'.format(query_kf_idx, ref_kf_idx), 'inliers.txt')
            if not os.path.exists(loop_inlier_path):
                loop_inlier_path = os.path.join(base_folder, loop_res_folder, 'eval_{}_{}'.format(query_kf_idx, ref_kf_idx), 'match_res/1_inliers.txt')
                if not os.path.exists(loop_inlier_path):
                    continue
            num_inliers = float(open(loop_inlier_path, 'r').readlines()[0])
            if num_inliers < 100:
                continue
            loop_scene_path = os.path.join(base_folder, loop_res_folder, 'eval_{}_{}'.format(query_kf_idx, ref_kf_idx), 'params.npz')
            loop_est_r2qs, loop_gt_r2qs, _ = load_poses(loop_scene_path)
            # print(len(loop_est_q2rs))
            # print(len(loop_gt_q2rs))
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
        # print(odo_idx)
        # save current node
        PGM.curr_node_idx = odo_idx # make start with 0
        if(PGM.curr_node_idx == 0):
            PGM.prev_node_idx = PGM.curr_node_idx
            continue
        
        # odo_transform = np.linalg.inv(PGM.curr_se3) @ np.linalg.inv(odo_est_w2c)
        odo_transform = all_odo_est_w2cs[odo_idx - 1] @ np.linalg.inv(odo_est_w2c)

        # update the current (moved) pose 
        # PGM.curr_se3 = np.linalg.inv(odo_est_w2c)
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
            # print(loop_transform)
            # print(loop_idx)

            odo_gt_w2c = all_odo_gt_w2cs[odo_idx]
            loop_gt_w2c = all_odo_gt_w2cs[loop_idx]
            gt_loop_transform = loop_gt_w2c @ np.linalg.inv(odo_gt_w2c)
            
            T_error = np.linalg.inv(loop_transform) @ gt_loop_transform
            r_err = cv2.Rodrigues(T_error[:3, :3])[0]
            r_err = np.linalg.norm(r_err) * 180 / math.pi
            t_err = np.linalg.norm(T_error[:3, 3], ord=2)
            print("icp T_error = ", r_err, t_err)
            # print(gt_loop_transform)
            # exit()

            PGM.addLoopFactor(loop_transform, loop_idx)

            # 2-2/ graph optimization 
            PGM.optimizePoseGraph()

            # 2-2/ save optimized poses
            ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)
            # ResultSaver.vizCurrentTrajectory(fig_idx=2)
            # plt.show()


        # save the ICP odometry pose result (no loop closure)
        ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx) 
        if(odo_idx % num_frames_to_skip_to_show == 0): 
            fig_save_path = os.path.join(fig_save_dir, '{}.png'.format(odo_idx))
            ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx, fig_save_path=fig_save_path)
            # writer.grab_frame()
        
        # if odo_idx == 418:
        #     plt.show()

    loop_trajs = ResultSaver.pose_list.reshape([-1, 4, 4])
    # print(loop_trajs.shape)
    # print(num_frames)
    gt_traj_pts = []
    est_traj_pts = []
    for i in range(num_frames):
        gt_c2w_trans = np.linalg.inv(all_odo_gt_w2cs[i])[:3, 3].flatten()
        est_c2w_trans = np.linalg.inv(all_odo_est_w2cs[i])[:3, 3].flatten()
        gt_traj_pts.append(gt_c2w_trans)
        est_traj_pts.append(est_c2w_trans)
    
    loop_traj_pts = []
    if loop_trajs.shape[0] != num_frames:
        loop_traj_pts.append([0.0, 0.0, 0.0])
    for i in range(loop_trajs.shape[0]):
        loop_c2w_trans = loop_trajs[i, :3, 3].flatten()
        loop_traj_pts.append(loop_c2w_trans)


    
    gt_traj_pts = np.array(gt_traj_pts).T
    est_traj_pts = np.array(est_traj_pts).T
    loop_traj_pts = np.array(loop_traj_pts).T
    print(gt_traj_pts.shape)
    print(est_traj_pts.shape)
    print(loop_traj_pts.shape)
    # print(gt_traj_pts.T[50:55, :])
    # print('-------------')
    # print(loop_traj_pts.T[50:55, :])
    # exit()

    intrinsics = all_params['intrinsics'].cpu().numpy()

    psnr_list = []
    ssim_list = []
    lpips_list = []
    with torch.no_grad():
        for i in tqdm(range(len(all_odo_est_w2cs)), 'Eval before color refine...'):
            color_gt = gt_imgs[i].to('cuda') # (C, H, W) [0, 1] torch.Tensor
            w2c = all_odo_est_w2cs[i]
            cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, w2c)
            rendervar = transformed_params2rendervar(all_params, all_params)
            w2c = torch.tensor(w2c, dtype=torch.float32, device='cuda')
            depth_sil_rendervar = transformed_params2depthplussilhouette(all_params, w2c,
                                                                     all_params)
            im, radius, _, _ = Renderer(raster_settings=cam)(**rendervar)
            # Render Depth & Silhouette
            depth_sil, _, _, _ = Renderer(raster_settings=cam)(**depth_sil_rendervar)
            silhouette = depth_sil[1, :, :]
            presence_sil_mask = (silhouette > sil_thres)

            weighted_im = im * presence_sil_mask
            weighted_gt_im = color_gt * presence_sil_mask
            psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
            ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                            data_range=1.0, size_average=True)
            lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                        torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()
            
            img_tensor = im

            # 定义转换：从张量到 PIL 图像
            transform = transforms.ToPILImage()

            # 将张量转换为 PIL 图像
            img = transform(img_tensor)

            # 保存图像到本地
            img.save(os.path.join(save_tmp_rgb_dir, "{:04d}.png".format(i)))

            psnr_list.append(psnr.cpu().numpy())
            ssim_list.append(ssim.cpu().numpy())
            lpips_list.append(lpips_score)


    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))


    # deformation
    for j in range(loop_trajs.shape[0]):
        if loop_trajs[j, 3, 3] < 1.0:
            loop_trajs[j, 3, 3] = 1.0

    first_frame_c2w = np.eye(4)
    first_frame_c2w = np.expand_dims(first_frame_c2w, axis=0)
    loop_trajs = np.concatenate((first_frame_c2w, loop_trajs), axis=0)

    pre_est_w2cs = torch.tensor(np.array(all_odo_est_w2cs), dtype=torch.float32, device='cuda') # (k, 4, 4)
    every_gs_pre_w2cs = pre_est_w2cs[all_params['timestep'].to(torch.long)] # (n, 4, 4)


    tmp = torch.ones(all_params['means3D'].shape[0], 1, dtype=torch.float32, device='cuda')
    gs_means3d = torch.cat((all_params['means3D'], tmp), dim=1) # (n, 4)
    gs_camera = torch.bmm(every_gs_pre_w2cs, gs_means3d.unsqueeze_(-1)).squeeze_(-1) # (n, 4)

    after_est_c2ws = torch.tensor(loop_trajs, dtype=torch.float32, device='cuda') # (k, 4, 4)
    every_gs_aft_c2ws = after_est_c2ws[all_params['timestep'].to(torch.long)] # (n, 4, 4)
    gs_new_means3d = torch.bmm(every_gs_aft_c2ws, gs_camera.unsqueeze_(-1)).squeeze_(-1) # (n, 4)

    all_params['means3D'] = torch.nn.Parameter(gs_new_means3d[:, :3].cuda().float().contiguous().requires_grad_(True))
    
    # color refine

    param_groups = [{'params': [v], 'name': k, 'lr': color_refine_lrs[k]} for k, v in all_params.items() if k in optimize_keys]
    optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    for i in tqdm(range(color_refine_total_iters), 'Color refining...'):
        index = random.randint(0, loop_trajs.shape[0]-1) 
        color_gt = gt_imgs[index].to('cuda') # (C, H, W) [0, 1] torch.Tensor
        w2c = np.linalg.inv(loop_trajs[index])
        cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, w2c)
        rendervar = transformed_params2rendervar(all_params, all_params)
        im, radius, _, _ = Renderer(raster_settings=cam)(**rendervar)
        loss = 0.8 * l1_loss_v1(im, color_gt) + 0.2 * (1.0 - calc_ssim(im, color_gt))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # TODO maybe: lr update

    print('Align results')
    _, _, trans_error = align(gt_traj_pts, est_traj_pts, align_traj=True)
    est_align_avg_trans_error = round(trans_error.mean(), 4)
    print(est_align_avg_trans_error)
    _, _, trans_error = align(gt_traj_pts[:, :], loop_traj_pts[:, :], align_traj=True)
    loop_align_avg_trans_error = round(trans_error.mean(), 4)
    print(loop_align_avg_trans_error)

    print('Absolute results')
    _, _, trans_error = align(gt_traj_pts, est_traj_pts, align_traj=False)
    est_abs_avg_trans_error = round(trans_error.mean(), 4)
    print(est_abs_avg_trans_error)
    _, _, trans_error = align(gt_traj_pts, loop_traj_pts, align_traj=False)
    loop_abs_avg_trans_error = round(trans_error.mean(), 4)
    print(loop_abs_avg_trans_error)

    gt_traj_poses = []
    est_traj_poses = []
    for i in range(num_frames):
        gt_c2w = np.linalg.inv(all_odo_gt_w2cs[i])
        # gt_c2w[:3, :3] = np.eye(3)
        est_c2w = np.linalg.inv(all_odo_est_w2cs[i])
        # est_c2w[:3, :3] = np.eye(3)
        gt_traj_poses.append(gt_c2w)
        est_traj_poses.append(est_c2w)

    loop_traj_poses = []
    if loop_trajs.shape[0] != num_frames:
        loop_traj_poses = [np.eye(4)]
    for i in range(loop_trajs.shape[0]):
        loop_c2w = np.eye(4)
        loop_c2w[:3, :3] = loop_trajs[i, :3, :3]
        loop_c2w[:3, 3] = loop_trajs[i, :3, 3]
        loop_traj_poses.append(loop_c2w)

    evo_res = evaluate_evo(gt_traj_poses, est_traj_poses)
    evo_res = evaluate_evo(gt_traj_poses, loop_traj_poses)

    if dataset == 'kitti':
        print('\nuse kitti evo tool')
        gt_traj_poses_dict = {}
        est_traj_poses_dict = {}
        loop_traj_poses_dict = {}
        for i in range(num_frames):
            gt_traj_poses_dict[i] = gt_traj_poses[i]
            est_traj_poses_dict[i] = est_traj_poses[i]
            loop_traj_poses_dict[i] = loop_traj_poses[i]

        eval_kitti_dir = os.path.join(save_dir, 'eval_kitti_loop')
        os.makedirs(eval_kitti_dir, exist_ok=True)
        eval_tool = KittiEvalOdom() 
        eval_tool.eval_kitti(gt_traj_poses_dict,
                            loop_traj_poses_dict,
                            result_dir=eval_kitti_dir,
                            alignment="6dof",
                            seq=scene_name)
    
        eval_kitti_dir = os.path.join(save_dir, 'eval_kitti_est')
        os.makedirs(eval_kitti_dir, exist_ok=True)
        eval_tool = KittiEvalOdom() 
        eval_tool.eval_kitti(gt_traj_poses_dict,
                            est_traj_poses_dict,
                            result_dir=eval_kitti_dir,
                            alignment="6dof",
                            seq=scene_name)


    fig = plt.figure(10)
    plt.clf()
    # plt.plot(-y, x, color='blue') # kitti camera coord for clarity
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
        # print(image)
        video.write(cv2.imread(os.path.join(fig_save_dir, image)))

    # eval color refine
    psnr_list = []
    ssim_list = []
    lpips_list = []
    with torch.no_grad():
        for i in tqdm(range(loop_trajs.shape[0]), 'Eval mapping result...'):
            color_gt = gt_imgs[i].to('cuda') # (C, H, W) [0, 1] torch.Tensor
            w2c = np.linalg.inv(loop_trajs[i])
            cam = setup_camera(color_gt.shape[2], color_gt.shape[1], intrinsics, w2c)
            rendervar = transformed_params2rendervar(all_params, all_params)
            w2c = torch.tensor(w2c, dtype=torch.float32, device='cuda')
            depth_sil_rendervar = transformed_params2depthplussilhouette(all_params, w2c,
                                                                     all_params)
            im, radius, _, _  = Renderer(raster_settings=cam)(**rendervar)
            # Render Depth & Silhouette
            depth_sil, _, _, _ = Renderer(raster_settings=cam)(**depth_sil_rendervar)
            silhouette = depth_sil[1, :, :]
            presence_sil_mask = (silhouette > sil_thres)

            weighted_im = im * presence_sil_mask
            weighted_gt_im = color_gt * presence_sil_mask
            psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
            ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                            data_range=1.0, size_average=True)
            lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                        torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

            psnr_list.append(psnr.cpu().numpy())
            ssim_list.append(ssim.cpu().numpy())
            lpips_list.append(lpips_score)

            if i % save_rendering_every == 0:
                # Save Rendered RGB
                viz_render_im = torch.clamp(im, 0, 1)
                viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
                cv2.imwrite(os.path.join(save_render_rgb_dir, "{:04d}.png".format(i)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))

                # Save GT RGB and Depth
                viz_gt_im = torch.clamp(color_gt, 0, 1)
                viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
                cv2.imwrite(os.path.join(save_gt_rgb_dir, "{:04d}.png".format(i)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    video.release()
