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
import csv
import trimesh
from tqdm import tqdm
from pyquaternion import Quaternion

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(current_dir)

from show_point_cloud import show_point_cloud


cam0_opt_K = np.array([435.2046959714599, 0, 367.4517211914062, 0, 435.2046959714599, 252.2008514404297, 0.0, 0.0, 1.0]).reshape([3, 3])
cam0_opt_distort = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

T_i_c0 = np.array(
    [
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

T_i_vicon = np.array([ 0.33638, -0.01749,  0.94156,  0.06901,
         -0.02078, -0.99972, -0.01114, -0.02781,
          0.94150, -0.01582, -0.33665, -0.12395,
              0.0,      0.0,      0.0,      1.0]).reshape([4, 4])

baseline_plus_fx = 47.90639384423901

scene_name = 'V2_02_medium'
depth_images_folder = "euroc/" + scene_name + "/mav0/cam0/depth_sceneflow"
# depth_images_folder = "euroc/" + scene_name + "/mav0/cam0/depth_sgbm"
color_images_folder = "euroc/" + scene_name + "/mav0/cam0/data_rect"
disparity_images_folder = "euroc/" + scene_name + "/mav0/cam0/disparity_sceneflow"
pose_file_path = "euroc/" + scene_name + "/mav0/state_groundtruth_estimate0/data.csv"
# pose_file_path = "euroc/" + scene_name + "/mav0/vicon0/data.csv"
# calib_file_path = "kitti/00/calib.txt"
step = 3

depth_images_path = os.listdir(depth_images_folder)
depth_images_path = sorted(depth_images_path, key=lambda x:float(x[:-4]))

with open(pose_file_path) as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [list(map(float, row)) for row in reader]
data = np.array(data)
print(data.shape)

pose_ts = data[:, 0]
pose_indices = []
for i in range(len(depth_images_path)):
    color_ts = float(depth_images_path[i].split(".")[0])
    # print(color_ts)
    k = np.argmin(np.abs(pose_ts - color_ts))
    diff = np.min(np.abs(pose_ts - color_ts))
    if diff / 1e9 > 0.01:
        print('wrong', diff / 1e9, color_ts / 1e9, (color_ts + diff) / 1e9)
        # exit()
        pose_indices.append(-1)
    else:
        pose_indices.append(k)
print(len(pose_indices))
# exit()

all_pts = []
for i in range(len(depth_images_path)):
# for i in range(300, 605, 10):
    if pose_indices[i] == -1:
        continue
    
    image_name = depth_images_path[i][:-4]
    print(image_name)

    print(i, pose_ts[pose_indices[i]])
    trans = data[pose_indices[i], 1:4]
    quat = data[pose_indices[i], 4:8] # w, x, y, z
    quat = quat[[1, 2, 3, 0]]
    T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
    T_w_i[:3, 3] = trans
    T_w_c = np.dot(T_w_i, T_i_c0)

    # trans = data[pose_indices[i], 1:4]
    # quat = data[pose_indices[i], 4:8] # w, x, y, z
    # pyquat = Quaternion(quat[0], quat[1], quat[2], quat[3])
    # T_w_i = np.eye(4)
    # T_w_i[:3, :3] = pyquat.rotation_matrix
    # T_w_i[:3, 3] = trans
    # T_w_i = T_w_i @ np.linalg.inv(T_i_vicon)
    # T_w_c = np.dot(T_w_i, T_i_c0)
    # print(T_w_i[:3, 3].flatten())

    depth_image_path = os.path.join(depth_images_folder, image_name + '.npy')
    color_image_path = os.path.join(color_images_folder, image_name + '.png')
    disparity_image_path = os.path.join(disparity_images_folder, image_name + '.npy')

    color_image = cv2.imread(color_image_path, cv2.IMREAD_UNCHANGED)
    if len(color_image.shape) == 2:
        color_image = np.dstack([color_image, color_image, color_image])

    depth_image = np.load(depth_image_path)

    # disparity_image = np.load(disparity_image_path)
    # depth_image = baseline_plus_fx / disparity_image
    
    print(color_image.shape)
    print(depth_image.shape)
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
    valid_depth_indices = np.where((depth_image > 1) & (depth_image < 10))[0]
    depth_image = depth_image.astype(float)
    color_image = color_image.astype(float)
    pts = pts[:, valid_depth_indices]
    depth_image = depth_image[valid_depth_indices]
    color_image = color_image[valid_depth_indices, :]
    print(pts.shape)
    
    image_pose = T_w_c

    X = np.multiply(depth_image.flatten(), np.linalg.inv(cam0_opt_K) @ pts)

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
