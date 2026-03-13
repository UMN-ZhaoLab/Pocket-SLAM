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

pts_folder = ""
pose_file_path = "kitti/00/traj.txt"
step = 1

pose_f = open(pose_file_path, "r")
poses = []
for line in pose_f.readlines():
    tmp_line = line.split(" ")
    tmp_line = [float(d) for d in tmp_line if len(d) > 0]
    # print(tmp_line)
    poses.append(tmp_line)
poses = np.array(poses).reshape([-1, 3, 4])
print(poses.shape)

pts_files = os.listdir(pts_folder)
pts_files = sorted(pts_files, key=lambda x:int(x[:-4]))
all_pts = []
for pts_file in pts_files[245:250:step]:
    pts = np.load(os.path.join(pts_folder, pts_file))
    print(pts.shape)
    image_idx = int(pts_file[:-4])
    image_pose = poses[image_idx]

    # image_pose = np.linalg.inv(np.vstack((image_pose, np.array([0, 0, 0, 1]))))[:3, :]
    pts_in_world = image_pose @ np.vstack((pts[:, :3].transpose(), np.ones(pts.shape[0])))
    pts_in_world = np.hstack([pts_in_world.transpose(), pts[:, 3:]])
    print(pts_in_world.shape)
    
    if len(all_pts) == 0:
        all_pts = pts_in_world
    else:
        print(all_pts.shape)
        print(pts_in_world.shape)
        all_pts = np.vstack([all_pts, pts_in_world])

show_point_cloud(all_pts, step=1)
