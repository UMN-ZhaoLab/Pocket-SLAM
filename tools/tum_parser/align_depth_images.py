import cv2
import numpy as np
import os
import torch
import sys
from skimage import io
import open3d as o3d
import numpy as np
import copy
import shutil
from tqdm import tqdm

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(current_dir)

from show_point_cloud import show_point_cloud


# color_images_folder = ""
# disparity_images_folder = "IGEV/IGEV-Stereo/output"
depth_images_folder = "IGEV_output/image_2_depth/numpy"
color_images_folder = "kitti/00/image_2"
disparity_images_folder = ""
pose_file_path = "kitti/00/traj.txt"
calib_file_path = "kitti/00/calib.txt"
step = 3

pose_f = open(pose_file_path, "r")
poses = []
for line in pose_f.readlines():
    tmp_line = line.split(" ")
    tmp_line = [float(d) for d in tmp_line if len(d) > 0]
    # print(tmp_line)
    poses.append(tmp_line)
poses = np.array(poses).reshape([-1, 3, 4])
print(poses.shape)

calib_f = open(calib_file_path, "r")
calib_lines = calib_f.readlines()
fx_image_2 = calib_lines[2].split(' ')[1]
baseline_plus_fx_image_2 = calib_lines[2].split(' ')[4]
K_image_2 = np.array([float(d) for d in calib_lines[2].split(' ')[1:]]).reshape([3, 4])[:3, :3]

fx_image_0 = calib_lines[0].split(' ')[1]
K_image_0 = np.array([float(d) for d in calib_lines[0].split(' ')[1:]]).reshape([3, 4])[:3, :3]

baseline_image0_image1 = 0.5371657188644179
baseline_image2_image3 = 0.5323318578407914

# print(fx_image_2)
# print(baseline_plus_fx_image_2)
# print(K_image_2)

fx = fx_image_2
baseline = baseline_image2_image3
K = K_image_2
print(fx)
print(baseline)
print(K)

image_files = os.listdir(color_images_folder)
image_files = sorted(image_files, key=lambda x:int(x[:-4]))
all_pts = []
for image_file in image_files[245:280:step]:
    color_image = cv2.imread(os.path.join(color_images_folder, image_file), cv2.IMREAD_UNCHANGED)
    if len(color_image.shape) == 2:
        color_image = np.dstack([color_image, color_image, color_image])

    # depth_image = np.load(os.path.join(depth_images_folder, image_file[:-4] + '.npy'))
    
    disparity_image = np.load(os.path.join(disparity_images_folder, image_file[:-4] + '.npy'))
    depth_image = float(fx) * baseline / disparity_image
    
    print(image_file)
    print(color_image.shape)
    print(depth_image.shape)
    print(disparity_image.shape)
    # print(np.max(depth_image))
    # print(np.min(depth_image))
    # exit()
    
    x, y = np.meshgrid(
            range(color_image.shape[1]), range(color_image.shape[0])
        )
    pts = np.vstack(
        (
            x.flatten(),
            y.flatten(),
            np.ones(color_image.shape[0] * color_image.shape[1]),
        )
    )

    depth_image = depth_image.flatten()
    color_image = color_image.reshape([-1, 3]) / 255.0
    valid_depth_indices = np.where((depth_image > 0.1) & (depth_image < 30))[0]
    depth_image = depth_image.astype(float)
    color_image = color_image.astype(float)
    pts = pts[:, valid_depth_indices]
    depth_image = depth_image[valid_depth_indices]
    color_image = color_image[valid_depth_indices, :]
    print(pts.shape)
    
    image_idx = int(image_file[:-4])
    image_pose = poses[image_idx]

    X = np.multiply(depth_image.flatten(), np.linalg.inv(K) @ pts)

    # image_pose = np.linalg.inv(np.vstack((image_pose, np.array([0, 0, 0, 1]))))[:3, :]
    # image_pose = np.eye(4)
    pts_in_world = image_pose @ np.vstack((X, np.ones(pts.shape[1]))) # (4, n)
    pts_in_world = np.hstack([pts_in_world.transpose()[:, :3], color_image])
    print('pts_in_world.shape: ' + str(pts_in_world.shape))
    if len(all_pts) == 0:
        all_pts = pts_in_world
    else:
        print(all_pts.shape)
        print(pts_in_world.shape)
        all_pts = np.vstack([all_pts, pts_in_world])

show_point_cloud(all_pts, step=1)
