import argparse
import os
import shutil
import sys
import time
import math
import open3d as o3d 
import copy
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

from sp_lg.lightglue import LightGlue
from sp_lg.superpoint import SuperPoint
from sp_lg.disk import DISK
from sp_lg.utils import load_image, match_pair
from sp_lg import viz2d
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify


def get_local_features(keyframe_info, sp_extractor, device, depth_filter_far):
    depth_original = torch.from_numpy(keyframe_info['depth_original']).to(device)
    color = torch.from_numpy(keyframe_info['color']).to(device)
    if depth_original is not None:
        mask = (depth_original < 0.1) | (depth_original > np.min([depth_filter_far, 15.0]))
    else:
        print('depth_original is None')
        exit()
    # print(depth_original.shape)
    # print(mask.shape)
    # print(color.shape)
    # color_feature = torch.clone(color)
    # color_feature[:, mask[0]] = 0
    color[:, mask[0]] = 0

    feats, desc_all = extract_feature(color, depth_original, sp_extractor, device)

    return feats, desc_all

def extract_feature(color, depth, sp_extractor, device):
    img = torch.tensor(color[None, ...], dtype=torch.float).to(device)
    # print(device)
    # print(color.shape)
    # print(depth.shape)
    # exit()
    # print(img.shape)
    feats, desc_all = sp_extractor({"image": img, "depth": depth})

    # print(feats['keypoints'].shape)
    # print(feats['keypoint_scores'].shape)
    # print(feats['descriptors'].shape)

    return feats, desc_all


def match_feature(
    config,
    curr_im,
    curr_feats,
    intrinsics,
    last_im,
    last_feats,
    lg_matcher,
    device,
    topk=512,
    save_path=None,
):
    img = torch.tensor(curr_im[None, ...], dtype=torch.float).to(device)
    last_img = torch.tensor(last_im[None, ...], dtype=torch.float).to(device)

    # match last_image and current_image
    data = {"image0": img, "image1": last_img}
    pred = {
        **{k + "0": v for k, v in curr_feats.items()},
        **{k + "1": v for k, v in last_feats.items()},
        **data,
    }
    pred = {**pred, **lg_matcher(pred)}
    pred = {
        k: v.to(device).detach()[0] if isinstance(v, torch.Tensor) else v
        for k, v in pred.items()
    }

    if "cuda" in str(device):
        torch.cuda.empty_cache()

    # create match indices
    matches0, mscores0 = pred["matches0"], pred["matching_scores0"]

    valid = matches0 > -1
    matches = torch.stack([torch.where(valid)[0], matches0[valid]], -1)
    pred = {**pred, "matches": matches, "matching_scores": mscores0[valid]}

    mscores, indices = mscores0[valid].sort(dim=0, descending=True)
    mscores = mscores[:topk]
    indices = indices[:topk]
    matches = matches[indices]

    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    m_kpts0 = m_kpts0.detach().cpu().numpy()
    m_kpts1 = m_kpts1.detach().cpu().numpy()
    # print(m_kpts0.shape)
    # print(m_kpts1.shape)
    # print(mscores)

    if m_kpts0.shape[0] < 20:
        return None, None, None

    mat_K = intrinsics.detach().cpu().numpy()
    # print(mat_K)
    # # use essential matrix for two view geometry verification
    # K = np.array(
    #     [
    #         intrinsics[0][0],
    #         0,
    #         intrinsics[0][2],
    #         0,
    #         intrinsics[1][1],
    #         intrinsics[1][2],
    #         0,
    #         0,
    #         1,
    #     ]
    # ).reshape([3, 3])
    essential_matrix, mask = cv2.findEssentialMat(
        m_kpts0, m_kpts1, mat_K, cv2.RANSAC, 0.999, 1.0
    )
        
    # pts, R, T, mask = cv2.recoverPose(essential_matrix, m_kpts0, m_kpts1)
    mask = mask.flatten().astype(bool)
    m_kpts0 = m_kpts0[mask]
    m_kpts1 = m_kpts1[mask]
    mscores = mscores[mask]

    print(m_kpts0.shape)
    print(m_kpts1.shape)
    
    # # # get gt matches for debug, use gt pose for two view geometry verification
    # mscores, m_kpts0, m_kpts1 = self.get_gt_matches(m_kpts0, m_kpts1, batch)

    if not config["tracking"]["use_gt_poses"] and save_path is not None:
        # for visualization
        axes = viz2d.plot_images(
            [img[0, ...].permute(1, 2, 0), last_img[0, ...].permute(1, 2, 0)]
        )
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        # viz2d.add_text(0, f'Stop after {pred["stop"]} layers', fs=20)

        # kpc0, kpc1 = viz2d.cm_prune(pred['prune0'].cpu()), viz2d.cm_prune(pred['prune1'].cpu())
        # viz2d.plot_images([img[0, ...].permute(1, 2, 0).cpu(), last_img[0, ...].permute(1, 2, 0).cpu()])
        # viz2d.plot_keypoints([kpts0.cpu(), kpts1.cpu()], colors=[kpc0, kpc1], ps=10)

        print("match prev_prev_image and current_image")
        # plt.show()
        plt.savefig(save_path)

    kpts0 = torch.tensor(np.array(m_kpts0)).to(device)
    kpts1 = torch.tensor(np.array(m_kpts1)).to(device)
    return kpts0, kpts1, mscores

def estimate_pnp_for_loop(mkpts_cur, mkpts_knn, intrinsic, curr_keyframe_info, knn_keyframe_info, dataset):
    pnp_prior_pose_w2c = None
    kps_curr = torch.clone(mkpts_cur).detach().cpu().numpy()
    kps_knn = torch.clone(mkpts_knn).detach().cpu().numpy()
    if knn_keyframe_info["depth_original"] is not None:
        knn_depth = np.copy(knn_keyframe_info["depth_original"])[0, :, :]
    else:
        print('depth original is None')
        exit()
    mat_K = torch.clone(intrinsic).detach().cpu().numpy()

    curr_gt_pose_w2c = np.copy(curr_keyframe_info['gt_w2c'])
    knn_gt_pose_w2c = np.copy(knn_keyframe_info['gt_w2c'])
    # print(kps_curr.shape)
    # print(kps_knn.shape)
    print(knn_depth.shape)
    print(np.max(knn_depth))
    print(np.min(knn_depth))
    # print(mat_K.shape)
    # print(curr_gt_pose_w2c.shape)
    # print(knn_gt_pose_w2c.shape)
    
    points_in_knn_cam = []
    uv_in_curr_image = []
    for kpidx in range(kps_knn.shape[0]):
        point_depth = knn_depth[
            int(kps_knn[kpidx, 1]), int(kps_knn[kpidx, 0])
        ]
        if point_depth < 0.1 or point_depth > np.max([50, dataset.depth_filter_far]):
            continue
        point_in_knn_cam = (
            point_depth
            * np.linalg.inv(mat_K)
            @ np.array([kps_knn[kpidx, 0], kps_knn[kpidx, 1], 1]).reshape([3, 1])
        )
        points_in_knn_cam.append(point_in_knn_cam)
        uv_in_curr_image.append([int(kps_curr[kpidx, 0]), int(kps_curr[kpidx, 1])])
    points_in_knn_cam = np.array(points_in_knn_cam).reshape([-1, 3])
    uv_in_curr_image = np.ascontiguousarray(uv_in_curr_image).reshape([-1, 1, 2])
    # print(points_in_knn_cam.shape)
    # print(uv_in_curr_image.shape)
    # exit()
    
    try:
        (success, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(
            points_in_knn_cam.astype(np.float64),
            uv_in_curr_image.astype(np.float64),
            mat_K,
            np.zeros([5, 1]),
            flags=cv2.SOLVEPNP_SQPNP,
            confidence=0.99,
            reprojectionError=10.0,
            iterationsCount=100,
        )
    except:
        return np.eye(4), 0

    if not success:
        return np.eye(4), 0

    print(success)
    # print(rotation_vector)
    # print(translation_vector)
    print(inliers.shape)
    # exit()
    num_inliers = inliers.shape[0]
    
    rot_curr_last = cv2.Rodrigues(rotation_vector)[0]
    trans_curr_last = translation_vector
    
    est_T_curr_last = np.eye(4)
    est_T_curr_last[:3, :3] = rot_curr_last
    est_T_curr_last[:3, 3:4] = trans_curr_last

    gt_T_curr_last = curr_gt_pose_w2c @ np.linalg.inv(knn_gt_pose_w2c)

    # print(np.linalg.inv(est_T_curr_w))
    # print("==================")
    # print(np.linalg.inv(curr_gt_pose_w2c))
    # print("==================")
    # print(np.linalg.inv(last_gt_pose_w2c))
    # print("==================")
    # print(np.linalg.inv(est_T_train_test) @ gt_T_train_test)
    T_error = np.linalg.inv(est_T_curr_last) @ gt_T_curr_last
    r_err = cv2.Rodrigues(T_error[:3, :3])[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    t_err = np.linalg.norm(T_error[:3, 3], ord=2)
    print("T_error = ", r_err, t_err)
    # exit()

    return est_T_curr_last, num_inliers


def estimate_pnp(mkpts_cur, mkpts_last, curr_data, last_data, dataset):
    pnp_prior_pose_w2c = None
    kps_curr = torch.clone(mkpts_cur).detach().cpu().numpy()
    kps_last = torch.clone(mkpts_last).detach().cpu().numpy()
    if last_data["depth_original"] is not None:
        last_depth = torch.clone(last_data["depth_original"]).detach().cpu().numpy()[0, :, :]
    else:
        last_depth = torch.clone(last_data["depth"]).detach().cpu().numpy()[0, :, :]
    last_K = torch.clone(last_data['intrinsics']).detach().cpu().numpy()
    last_est_pose_w2c = torch.clone(last_data['est_w2c']).detach().cpu().numpy()

    curr_gt_pose_w2c = torch.clone(curr_data['iter_gt_w2c_list'][-1]).detach().cpu().numpy()
    last_gt_pose_w2c = torch.clone(last_data['iter_gt_w2c_list'][-2]).detach().cpu().numpy()
    # print(kps_curr.shape)
    # print(kps_last.shape)
    print(last_depth.shape)
    print(np.max(last_depth))
    print(np.min(last_depth))
    # print(last_K.shape)
    # print(last_est_pose_w2c.shape)
    # print(curr_gt_pose_w2c.shape)
    # print(last_gt_pose_w2c.shape)
    # print(curr_data["id"], last_data["id"])
    print('------------------------------------------ ', len(last_data['iter_gt_w2c_list']))

    points_in_last_cam = []
    uv_in_curr_image = []
    for kpidx in range(kps_last.shape[0]):
        point_depth = last_depth[
            int(kps_last[kpidx, 1]), int(kps_last[kpidx, 0])
        ]
        if point_depth < 0.1 or point_depth > np.max([50, dataset.depth_filter_far]):
            continue
        point_in_last_cam = (
            point_depth
            * np.linalg.inv(last_K)
            @ np.array([kps_last[kpidx, 0], kps_last[kpidx, 1], 1]).reshape([3, 1])
        )
        points_in_last_cam.append(point_in_last_cam)
        uv_in_curr_image.append([int(kps_curr[kpidx, 0]), int(kps_curr[kpidx, 1])])
    points_in_last_cam = np.array(points_in_last_cam).reshape([-1, 3])
    uv_in_curr_image = np.ascontiguousarray(uv_in_curr_image).reshape([-1, 1, 2])
    # print(points_in_last_cam.shape)
    # print(uv_in_curr_image.shape)
    # exit()
    
    try:
        (success, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(
            points_in_last_cam.astype(np.float64),
            uv_in_curr_image.astype(np.float64),
            last_K,
            np.zeros([5, 1]),
            flags=cv2.SOLVEPNP_SQPNP,
            confidence=0.90,
            reprojectionError=10.0,
            iterationsCount=200,
        )
    except:
        return None

    print(success)
    # print(rotation_vector)
    # print(translation_vector)
    print(inliers.shape)
    # exit()
    num_inliers = inliers.shape[0]
    
    rot_curr_last = cv2.Rodrigues(rotation_vector)[0]
    trans_curr_last = translation_vector
    
    est_T_curr_last = np.eye(4)
    est_T_curr_last[:3, :3] = rot_curr_last
    est_T_curr_last[:3, 3:4] = trans_curr_last

    # est_T_curr_w = est_T_curr_last @ last_gt_pose_w2c
    est_T_curr_w = est_T_curr_last @ last_est_pose_w2c

    gt_T_curr_last = curr_gt_pose_w2c @ np.linalg.inv(last_gt_pose_w2c)

    # print(np.linalg.inv(est_T_curr_w))
    # print("==================")
    # print(np.linalg.inv(curr_gt_pose_w2c))
    # print("==================")
    # print(np.linalg.inv(last_gt_pose_w2c))
    # print("==================")
    # print(np.linalg.inv(est_T_train_test) @ gt_T_train_test)
    T_error = np.linalg.inv(est_T_curr_last) @ gt_T_curr_last
    r_err = cv2.Rodrigues(T_error[:3, :3])[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    t_err = np.linalg.norm(T_error[:3, 3], ord=2)
    print("pnp T_error = ", r_err, t_err)
    # exit()

    pnp_prior_pose_w2c = est_T_curr_w
    return pnp_prior_pose_w2c, est_T_curr_last, num_inliers

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # if not source_temp.has_normals():
    #     estimate_normal(source_temp)
    #     estimate_normal(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def get_patch_rgb(
    last_image, pts_last_warp_normalize, patch_size=11, channel_dim=3, device="cuda:0"
):

    image_width = last_image.shape[2]
    image_height = last_image.shape[1]

    batch_patch_uv = (
        pts_last_warp_normalize.clone()
        .view(pts_last_warp_normalize.shape[0], 1, pts_last_warp_normalize.shape[1])
        .repeat(1, patch_size * patch_size, 1)
    )
    offset_kernel = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, patch_size) - patch_size // 2,
                torch.arange(0, patch_size) - patch_size // 2,
            ),
            dim=-1,
        )
        .view(1, patch_size * patch_size, 2)
        .repeat(batch_patch_uv.shape[0], 1, 1)
        .to(device)
    )
    batch_patch_uv = batch_patch_uv + offset_kernel
    batch_patch_uv = batch_patch_uv.view(-1, 2)

    batch_patch_uv = batch_patch_uv.reshape([1, 1, -1, 2])
    batch_patch_uv[..., 0] = batch_patch_uv[..., 0] / image_width * 2.0 - 1.0
    batch_patch_uv[..., 1] = batch_patch_uv[..., 1] / image_height * 2.0 - 1.0

    pts_last_warp_rgb = (
        torch.nn.functional.grid_sample(
            last_image[None, ...].float().to(device),
            batch_patch_uv,
            padding_mode="border",
            mode="bicubic",
        )
        .permute(0, 2, 3, 1)
        .reshape([-1, channel_dim])
    )

    pts_last_warp_patch_rgb = pts_last_warp_rgb.reshape(
        [-1, patch_size * patch_size, channel_dim]
    ).to(device)

    return pts_last_warp_patch_rgb


def get_patch_feature(
    last_image, pts_last_warp_normalize, patch_size=11, channel_dim=3, device="cuda:0"
):

    # # feature [1, h, w, d]

    image_width = last_image.shape[2]
    image_height = last_image.shape[1]

    batch_patch_uv = (
        pts_last_warp_normalize.clone()
        .view(pts_last_warp_normalize.shape[0], 1, pts_last_warp_normalize.shape[1])
        .repeat(1, patch_size * patch_size, 1)
    )
    offset_kernel = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, patch_size) - patch_size // 2,
                torch.arange(0, patch_size) - patch_size // 2,
            ),
            dim=-1,
        )
        .view(1, patch_size * patch_size, 2)
        .repeat(batch_patch_uv.shape[0], 1, 1)
        .to(device)
    )
    batch_patch_uv = batch_patch_uv + offset_kernel
    batch_patch_uv = batch_patch_uv.view(-1, 2)

    batch_patch_uv = batch_patch_uv.reshape([1, 1, -1, 2])
    batch_patch_uv[..., 0] = batch_patch_uv[..., 0] / image_width * 2.0 - 1.0
    batch_patch_uv[..., 1] = batch_patch_uv[..., 1] / image_height * 2.0 - 1.0

    pts_last_warp_feature = (
        torch.nn.functional.grid_sample(
            last_image.permute(0, 3, 1, 2).float().to(device),
            batch_patch_uv,
            padding_mode="border",
            mode="bicubic",
        )
        .permute(0, 2, 3, 1)
        .reshape([-1, channel_dim])
    )

    pts_last_warp_patch_feature = pts_last_warp_feature.reshape(
        [-1, patch_size * patch_size, channel_dim]
    ).to(device)

    return pts_last_warp_patch_feature


def get_patch_depth(
    last_image, pts_last_warp_normalize, patch_size=11, channel_dim=1, device="cuda:0"
):

    image_width = last_image.shape[2]
    image_height = last_image.shape[1]

    batch_patch_uv = (
        pts_last_warp_normalize.clone()
        .view(pts_last_warp_normalize.shape[0], 1, pts_last_warp_normalize.shape[1])
        .repeat(1, patch_size * patch_size, 1)
    )
    offset_kernel = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, patch_size) - patch_size // 2,
                torch.arange(0, patch_size) - patch_size // 2,
            ),
            dim=-1,
        )
        .view(1, patch_size * patch_size, 2)
        .repeat(batch_patch_uv.shape[0], 1, 1)
        .to(device)
    )
    batch_patch_uv = batch_patch_uv + offset_kernel
    batch_patch_uv = batch_patch_uv.view(-1, 2)

    batch_patch_uv = batch_patch_uv.reshape([1, 1, -1, 2])
    batch_patch_uv[..., 0] = batch_patch_uv[..., 0] / image_width * 2.0 - 1.0
    batch_patch_uv[..., 1] = batch_patch_uv[..., 1] / image_height * 2.0 - 1.0

    pts_last_warp_depth = (
        torch.nn.functional.grid_sample(
            last_image[None, ...].float().to(device),
            batch_patch_uv,
            padding_mode="border",
            mode="bicubic",
        )
        .permute(0, 2, 3, 1)
        .reshape([-1, channel_dim])
    )

    pts_last_warp_patch_depth = pts_last_warp_depth.reshape(
        [-1, patch_size * patch_size, channel_dim]
    ).to(device)

    return pts_last_warp_patch_depth


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel, std=1.5):
    _1D_window = gaussian(window_size, std).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(pred, gt, window, channel):
    ntotpx, nviews, nc, h, w = pred.shape
    flat_pred = pred.view(-1, nc, h, w)
    mu1 = F.conv2d(flat_pred, window, padding=0, groups=channel).view(
        ntotpx, nviews, nc
    )
    mu2 = F.conv2d(gt, window, padding=0, groups=channel).view(ntotpx, nc)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2).unsqueeze(1)
    mu1_mu2 = mu1 * mu2.unsqueeze(1)

    sigma1_sq = (
        F.conv2d(flat_pred * flat_pred, window, padding=0, groups=channel).view(
            ntotpx, nviews, nc
        )
        - mu1_sq
    )

    # print(gt.shape)
    # print(window.shape)
    # print(mu2_sq.shape)
    # print(channel)
    # print(ntotpx)

    sigma2_sq = (
        F.conv2d(gt * gt, window, padding=0, groups=channel).view(ntotpx, 1, channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            (pred * gt.unsqueeze(1)).view(-1, nc, h, w),
            window,
            padding=0,
            groups=channel,
        ).view(ntotpx, nviews, nc)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    values = 1 - ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return torch.sum(values, dim=2) / 2


class SSIM(torch.nn.Module):
    def __init__(self, h_patch_size, channel=3):
        super(SSIM, self).__init__()
        self.window_size = 2 * h_patch_size + 1
        self.channel = channel
        self.register_buffer("window", create_window(self.window_size, self.channel))

    def forward(self, img_pred, img_gt):
        ntotpx, nviews, npatch, channels = img_pred.shape

        patch_size = int(sqrt(npatch))
        patch_img_pred = (
            img_pred.reshape(ntotpx, nviews, patch_size, patch_size, channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        patch_img_gt = img_gt.reshape(ntotpx, patch_size, patch_size, channels).permute(
            0, 3, 1, 2
        )

        return _ssim(patch_img_pred, patch_img_gt, self.window, self.channel)


def get_loss_from_match(
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
):

    # Get Frame Camera Pose
    cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx])
    cam_tran = params["cam_trans"][..., time_idx]
    rel_w2curr = torch.eye(4).cuda().float()
    rel_w2curr[:3, :3] = build_rotation(cam_rot)
    rel_w2curr[:3, 3] = cam_tran

    rel_w2last = last_data["est_w2c"].to(device)

    # rel_w2curr = curr_data['iter_gt_w2c_list'][curr_data['id']].to(device)
    # rel_w2last = last_data['iter_gt_w2c_list'][last_data['id']].to(device)
    # print(len(curr_data['iter_gt_w2c_list']), curr_data['id'])
    # print(len(last_data['iter_gt_w2c_list']), last_data['id'])
    # print((rel_w2curr @ curr_data['w2c']).inverse())
    # print((rel_w2last @ last_data['w2c']).inverse())
    # exit()

    image_width = curr_data["im"].shape[2]
    image_height = curr_data["im"].shape[1]

    # print(curr_data['intrinsics'])
    # print('depth_trunc = ', config['data']['depth_trunc'])

    mat_K = curr_data["intrinsics"]

    cxcy = torch.tensor([mat_K[0, 2], mat_K[1, 2]]).to(device)[None, ...]
    fxfy = torch.tensor([mat_K[0, 0], mat_K[1, 1]]).to(device)[None, ...]

    normalized_kps = (mkpts_cur - cxcy) / fxfy
    normalized_kps = torch.cat(
        [normalized_kps, torch.ones(normalized_kps.shape[0], 1).to(device)], dim=-1
    )

    mkpts_cur_depth = curr_data["depth"][
        0, mkpts_cur[:, 1].long(), mkpts_cur[:, 0].long()
    ][..., None].to(device)

    # mask = (mkpts_cur_depth > 0.1) & (mkpts_cur_depth < config["data"]["depth_trunc"])
    mask = (mkpts_cur_depth > 0.1)
    normalized_kps = normalized_kps[mask[:, 0]]
    mkpts_cur_depth = mkpts_cur_depth[mask[:, 0]]
    mkpts_last = mkpts_last[mask[:, 0]]
    mscores = mscores[mask[:, 0]]

    # print(mask.shape)
    # print(normalized_kps.shape)
    # print(mkpts_cur_depth.shape)
    # print(mkpts_last.shape)
    # print(mscores.shape)

    pts_cur = (normalized_kps * mkpts_cur_depth).to(device)

    T_last_cur = rel_w2last.to(device) @ rel_w2curr.inverse()
    # print(T_last_cur)

    Rp = torch.sum(pts_cur[:, None, :] * T_last_cur[:3, :3], -1)
    t = T_last_cur[:3, -1].repeat(pts_cur.shape[0], 1)
    pts_last_warp = Rp + t

    pts_last_warp_normalize = pts_last_warp[:, :2] / pts_last_warp[:, -1:]
    pts_last_warp_normalize = pts_last_warp_normalize * fxfy + cxcy

    # patch_size = 11
    # boundary = (patch_size - 1) / 2
    # bound = torch.tensor(
    #     [
    #         [boundary + 20, image_width - boundary - 20],
    #         [boundary + 20, image_height - boundary - 20],
    #     ]
    # ).to(device)
    # mask = (
    #     (pts_last_warp_normalize[:, 0] > bound[0, 0])
    #     & (pts_last_warp_normalize[:, 0] < bound[0, 1])
    #     & (pts_last_warp_normalize[:, 1] > bound[1, 0])
    #     & (pts_last_warp_normalize[:, 1] < bound[1, 1])
    # )
    # pts_last_warp_normalize = pts_last_warp_normalize[mask]
    # mkpts_last = mkpts_last[mask]
    # mscores = mscores[mask]
    # mscores = torch.nn.functional.normalize(mscores.float(), p=2, dim=0)

    # patch_size = 11
    # channel_dim = 3
    # pts_last_warp_patch_5_rgb = get_patch_rgb(
    #     last_data["im"],
    #     pts_last_warp_normalize,
    #     patch_size=patch_size,
    #     channel_dim=channel_dim,
    #     device=device,
    # ).to(device)
    # pts_last_patch_5_rgb = get_patch_rgb(
    #     last_data["im"], mkpts_last, patch_size=patch_size, channel_dim=channel_dim
    # ).to(device)
    # pts_last_warp_patch_5_rgb = pts_last_warp_patch_5_rgb * mscores[..., None, None]
    # pts_last_patch_5_rgb = pts_last_patch_5_rgb * mscores[..., None, None]

    # patch_size = 11
    # channel_dim = 1
    # pts_last_warp_patch_5_depth = get_patch_depth(
    #     last_data["depth"],
    #     pts_last_warp_normalize,
    #     patch_size=patch_size,
    #     channel_dim=channel_dim,
    #     device=device,
    # ).to(device)
    # pts_last_patch_5_depth = get_patch_depth(
    #     last_data["depth"],
    #     mkpts_last,
    #     patch_size=patch_size,
    #     channel_dim=channel_dim,
    #     device=device,
    # ).to(device)
    # pts_last_warp_patch_5_depth = pts_last_warp_patch_5_depth * mscores[..., None, None]
    # pts_last_patch_5_depth = pts_last_patch_5_depth * mscores[..., None, None]

    # patch_size = 7
    # channel_dim = 3
    # pts_last_warp_patch_3_rgb = get_patch_rgb(
    #     last_data["im"],
    #     pts_last_warp_normalize,
    #     patch_size=patch_size,
    #     channel_dim=channel_dim,
    #     device=device,
    # ).to(device)
    # pts_last_patch_3_rgb = get_patch_rgb(
    #     last_data["im"],
    #     mkpts_last,
    #     patch_size=patch_size,
    #     channel_dim=channel_dim,
    #     device=device,
    # ).to(device)
    # pts_last_warp_patch_3_rgb = pts_last_warp_patch_3_rgb * mscores[..., None, None]
    # pts_last_patch_3_rgb = pts_last_patch_3_rgb * mscores[..., None, None]

    '''
    # patch_size = 1
    # channel_dim = 3
    # pts_last_warp_patch_1_rgb = get_patch_rgb(last_data['im'], pts_last_warp_normalize, patch_size=patch_size,
    #                                                channel_dim=channel_dim, device=device).to(device)
    # pts_last_patch_1_rgb = get_patch_rgb(last_data['im'], mkpts_last, patch_size=patch_size, channel_dim=channel_dim, device=device).to(device)
    # pts_last_warp_patch_1_rgb = pts_last_warp_patch_1_rgb * mscores[..., None, None]
    # pts_last_patch_1_rgb = pts_last_patch_1_rgb * mscores[..., None, None]

    # patch_size = 1
    # channel_dim = 1
    # pts_last_warp_patch_1_depth = get_patch_depth(last_data['depth'], pts_last_warp_normalize, patch_size=patch_size,
    #                                                    channel_dim=channel_dim, device=device).to(device)
    # pts_last_patch_1_depth = get_patch_depth(last_data['depth'], mkpts_last, patch_size=patch_size, channel_dim=channel_dim, device=device).to(device)
    # pts_last_warp_patch_1_depth = pts_last_warp_patch_1_depth * mscores[..., None, None]
    # pts_last_patch_1_depth = pts_last_patch_1_depth * mscores[..., None, None]
    '''

    patch_size = 1
    channel_dim = last_data["descs"].shape[-1]
    pts_last_warp_patch_1_feature = get_patch_feature(
        last_data["descs"],
        pts_last_warp_normalize,
        patch_size=patch_size,
        channel_dim=channel_dim,
        device=device,
    ).to(device)
    pts_last_patch_1_feature = get_patch_feature(
        last_data["descs"],
        mkpts_last,
        patch_size=patch_size,
        channel_dim=channel_dim,
        device=device,
    ).to(device)
    pts_last_warp_patch_1_feature = (
        pts_last_warp_patch_1_feature * mscores[..., None, None]
    )
    pts_last_patch_1_feature = pts_last_patch_1_feature * mscores[..., None, None]

    # # for visualization
    # if iter == 0 or iter == config['tracking']['num_iters'] - 1:
    #     patch_size = 101
    #     channel_dim = 3
    #     pts_last_warp_patch_50_rgb = get_patch_rgb(last_data['im'], pts_last_warp_normalize, patch_size=patch_size,
    #                                                    channel_dim=channel_dim, device=device).to(device)
    #     pts_last_patch_50_rgb = get_patch_rgb(last_data['im'], mkpts_last, patch_size=patch_size, channel_dim=channel_dim, device=device).to(device)
    #     for ii in range(0, pts_last_warp_patch_50_rgb.shape[0], 50):
    #         print(pts_last_warp_patch_50_rgb.shape)
    #         patch_warp = pts_last_warp_patch_50_rgb[ii, :, :].reshape([patch_size, patch_size, channel_dim]).detach().cpu().numpy()
    #         patch = pts_last_patch_50_rgb[ii, :, :].reshape([patch_size, patch_size, channel_dim]).detach().cpu().numpy()
    #         # print(patch_warp - patch)
    #         print(rel_w2last)
    #         print(rel_w2curr)
    #         print(patch.shape)
    #         print(T_last_cur)
    #         print(T_last_cur.detach().inverse())
    #         gt_w2curr = curr_data['iter_gt_w2c_list'][curr_data['id']].to(device)
    #         gt_w2last = last_data['iter_gt_w2c_list'][last_data['id']].to(device)
    #         gt_T_last_cur = gt_w2last.to(device) @ gt_w2curr.inverse()
    #         print(gt_T_last_cur)

    #         # patch = np.zeros(patch.shape).astype(np.uint8) + 125
    #         cv2.imshow('patch_warp', (patch_warp.transpose(1, 0, 2) * 255).astype(np.uint8))
    #         cv2.imshow('patch', (patch.transpose(1, 0, 2) * 255).astype(np.uint8))
    #         img = last_data['im'].permute(1, 2, 0).detach().cpu().numpy()
    #         pts_last_warp_normalize_show = pts_last_warp_normalize.detach().cpu().numpy()
    #         mkpts_last_show = mkpts_last.detach().cpu().numpy()
    #         cv2.circle(img, tuple([int(mkpts_last_show[ii, 0]), int(mkpts_last_show[ii, 1])]), radius=2, color=(0, 255, 0),
    #                        thickness=-1)
    #         cv2.circle(img, tuple([int(pts_last_warp_normalize_show[ii, 0]), int(pts_last_warp_normalize_show[ii, 1])]), radius=2,
    #                    color=(255, 255, 0),
    #                    thickness=-1)
    #         cv2.imshow('img', img)
    #         cv2.waitKey()
    #         cv2.destroyAllWindows()

    # ssim_loss_5_rgb = SSIM(5, 3).to(device)
    # loss_5_rgb = ssim_loss_5_rgb.forward(
    #     pts_last_warp_patch_5_rgb[:, None, ...], pts_last_patch_5_rgb
    # ).sum()

    # ssim_loss_5_depth = SSIM(5, 1).to(device)
    # loss_5_depth = ssim_loss_5_depth.forward(
    #     pts_last_warp_patch_5_depth[:, None, ...], pts_last_patch_5_depth
    # ).sum()

    # ssim_loss_3 = SSIM(3, 3).to(device)
    # loss_3 = ssim_loss_3.forward(
    #     pts_last_warp_patch_3_rgb[:, None, ...], pts_last_patch_3_rgb
    # ).sum()

    loss_1_feature = (
        torch.nn.functional.smooth_l1_loss(
            pts_last_warp_patch_1_feature,
            pts_last_patch_1_feature,
            beta=0.1,
            reduction="mean",
        )
        * 1.0
    )

    '''
    # loss_1_rgb = torch.nn.functional.smooth_l1_loss(
    #             pts_last_warp_patch_1_rgb,
    #             pts_last_patch_1_rgb,
    #             beta=0.1,
    #             reduction="sum",
    #         ) * 1.0
    #
    # loss_1_depth = torch.nn.functional.smooth_l1_loss(
    #             pts_last_warp_patch_1_depth,
    #             pts_last_patch_1_depth,
    #             beta=0.1,
    #             reduction="sum",
    #         ) * 1.0
    '''
    
    # # normalize kp
    # pts_last_warp_normalize[:, 0] = pts_last_warp_normalize[:, 0] / self.dataset.W
    # pts_last_warp_normalize[:, 1] = pts_last_warp_normalize[:, 1] / self.dataset.H
    # mkpts_last[:, 0] = mkpts_last[:, 0] / self.dataset.W
    # mkpts_last[:, 1] = mkpts_last[:, 1] / self.dataset.H

    loss_kp_mean = (
        torch.nn.functional.smooth_l1_loss(
            pts_last_warp_normalize * mscores[..., None],
            mkpts_last * mscores[..., None],
            beta=0.1,
            reduction="mean",
        )
        * 1.0
    )
    # print(loss_kp_mean, pts_last_warp_normalize.shape[0])

    # loss_pose = (loss_3 + loss_5_rgb + loss_5_depth * 0.01) * 0.01
    # loss_pose = (loss_3 + loss_5_rgb) * 0.01
    # loss_pose += loss_5_depth * 0.01
    # loss_pose = (loss_5_rgb + loss_5_depth) * 0.01
    # loss_pose = (loss_3 + loss_5_rgb + loss_5_depth) * 0.01
    loss_pose = 0.0
    # if iter < config["tracking"]["num_iters"] / 2:
    #     print("add kp loss", iter)
    #     loss_pose += loss_kp_mean
    loss_pose += loss_kp_mean
    loss_pose += loss_1_feature

    return loss_pose
