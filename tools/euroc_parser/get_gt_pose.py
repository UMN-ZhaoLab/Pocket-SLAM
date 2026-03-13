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

baseline_times_fx = 47.90639384423901

depth_images_folder = "euroc/mh02/mav0/cam0/depth_sceneflow"
pose_file_path = "euroc/mh02/mav0/state_groundtruth_estimate0/data.csv"

output_pose_file_path = 'euroc/mh02/mav0/cam0/traj.txt'

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
    depth_ts = float(depth_images_path[i].split(".")[0])
    # print(depth_ts)
    k = np.argmin(np.abs(pose_ts - depth_ts))
    diff = np.min(np.abs(pose_ts - depth_ts))
    if diff > 1e6:
        print('wrong, need skip', diff, depth_images_path[i])
        # exit()
        pose_indices.append([-1, diff, depth_ts])
    else:
        pose_indices.append([k, diff, depth_ts])
print(len(pose_indices))
# exit()

output_f = open(output_pose_file_path, 'w')
for i in range(len(pose_indices)):
    idx = pose_indices[i][0]
    diff = pose_indices[i][1]
    timestamp = pose_indices[i][2]
    if idx == -1:
        continue
    # print(diff)
    trans = data[idx, 1:4]
    quat = data[idx, 4:8]
    quat = quat[[1, 2, 3, 0]]

    T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
    T_w_i[:3, 3] = trans
    T_w_c = np.dot(T_w_i, T_i_c0)

    output_pose = T_w_c[:3, :].flatten()
    # print(T_w_c.shape)
    # print(output_pose)
    # exit()

    output_str = "{:.0f} ".format(timestamp)
    for j in range(output_pose.shape[0]):
        output_str += "{} ".format(output_pose[j])
    output_str = output_str[:-1] + "\n"
    output_f.write(output_str)

output_f.close()
