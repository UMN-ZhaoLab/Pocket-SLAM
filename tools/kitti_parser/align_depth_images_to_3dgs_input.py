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
import imageio.v2 as imageio

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(current_dir)

from show_point_cloud import show_point_cloud

parent_folder = ''
# sequences = ["{:02d}".format(i) for i in range(11)]
sequences = ['10'] #00(4541), 02(4661), 03(801), 04(271), 05(2761), 08(4071), 09(1591), 10(1201)
subsection = [i for i in range(0, 1201, 300)]
subsection[-1] = 1201
# process_bar = tqdm(range(11))


# color_images_folder = ""
# disparity_images_folder = "IGEV/IGEV-Stereo/output"

for sequence in sequences:
    depth_images_folder = os.path.join(parent_folder, sequence, "depth_sceneflow")
    color_images_folder = os.path.join(parent_folder, sequence, "image_2")
    disparity_images_folder = os.path.join(parent_folder, sequence, "disparity_sceneflow")
    pose_file_path = os.path.join(parent_folder, sequence, "traj.txt")
    calib_file_path = os.path.join(parent_folder, sequence, "calib.txt")

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

    for section_index in range(len(subsection)):
        if section_index == len(subsection) - 1:
            break
        start = subsection[section_index]
        end = subsection[section_index+1]
        stride = 2
        save_pcd_folder = os.path.join(parent_folder, sequence, "3dgs_input", "{:04d}_{:04d}_{:02d}".format(start,end,stride))
        if not os.path.exists(save_pcd_folder):
            os.makedirs(save_pcd_folder)
        save_pcd_path=os.path.join(save_pcd_folder, "initialize_pcd.pcd")
        all_pts = []
        img_bar = tqdm(range(start, end, stride), f'sequence: {sequence}'+' section: {:04d} to {:04d}'.format(start, end))
        for image_file in image_files[start:end:stride]:
            # color_image = cv2.imread(os.path.join(color_images_folder, image_file), cv2.IMREAD_UNCHANGED)
            color_image = np.asarray(imageio.imread(os.path.join(color_images_folder, image_file)))
            if len(color_image.shape) == 2:
                color_image = np.dstack([color_image, color_image, color_image])

            depth_image = np.load(os.path.join(depth_images_folder, image_file[:-4] + '.npy'))
            
            disparity_image = np.load(os.path.join(disparity_images_folder, image_file[:-4] + '.npy'))
            # depth_image = float(fx) * baseline / disparity_image
            
            # print('------------------------', image_file, '------------------------')
            # print(color_image.shape)
            # print(depth_image.shape)
            # print(disparity_image.shape)
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
            # print(pts.shape)
            
            image_idx = int(image_file[:-4])
            image_pose = poses[image_idx]

            X = np.multiply(depth_image.flatten(), np.linalg.inv(K) @ pts)

            # image_pose = np.linalg.inv(np.vstack((image_pose, np.array([0, 0, 0, 1]))))[:3, :]
            # image_pose = np.eye(4)
            pts_in_world = image_pose @ np.vstack((X, np.ones(pts.shape[1]))) # (4, n)
            pts_in_world = np.hstack([pts_in_world.transpose()[:, :3], color_image])
            print('pts_in_world.shape: ' + str(pts_in_world.shape))

            tmp_pcd = o3d.geometry.PointCloud()
            tmp_pcd.points = o3d.utility.Vector3dVector(pts_in_world[:, :3])
            tmp_pcd.colors = o3d.utility.Vector3dVector(pts_in_world[:, 3:])
            tmp_pcd = tmp_pcd.random_down_sample(1.0 / 16.0)
            new_xyz = np.asarray(tmp_pcd.points)
            new_rgb = np.asarray(tmp_pcd.colors)
            pts_in_world = np.hstack([new_xyz, new_rgb])
            print('After downsample pts_in_world.shape: ' + str(pts_in_world.shape))

            if len(all_pts) == 0:
                all_pts = pts_in_world
            else:
                # print(all_pts.shape)
                # print(pts_in_world.shape)
                all_pts = np.vstack([all_pts, pts_in_world])

            img_bar.update()

        total_pcd = o3d.geometry.PointCloud()
        total_pcd.points = o3d.utility.Vector3dVector(all_pts[:, :3])
        total_pcd.colors = o3d.utility.Vector3dVector(all_pts[:, 3:])
        o3d.io.write_point_cloud(save_pcd_path, total_pcd)
    
    # process_bar.update()

# show_point_cloud(all_pts, step=1)
