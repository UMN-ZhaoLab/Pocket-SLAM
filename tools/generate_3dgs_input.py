import numpy as np
from natsort import natsorted
import torch
import json
import os
from tqdm import tqdm
import math
import imageio.v2 as imageio

def read_poses_file(filename, calibration):
    """ 
        read pose file (with the kitti format)
    """
    pose_file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(
            torch.from_numpy(np.matmul(Tr_inv, np.matmul(pose, Tr))).float()
        )  # lidar pose in world frame

    pose_file.close()
    return poses

# 生成运行3d高斯需要的json文件
if __name__ == "__main__":
    parent_folder = ''
    sequences = ["{:02d}".format(i) for i in range(11)]
    
    for sequence in sequences:
        image_path = os.path.join(parent_folder, sequence, 'image_2')
        pose_file_path = os.path.join(parent_folder, sequence, 'traj.txt')
        calib_file_path = os.path.join(parent_folder, sequence, "calib.txt")
        
        calib_f = open(calib_file_path, "r")
        calib_lines = calib_f.readlines()
        K = np.array([float(d) for d in calib_lines[2].split(' ')[1:]]).reshape([3, 4])[:3, :3]

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        test_img = np.asarray(imageio.imread(os.path.join(image_path, '000000.png')))
        H, W = test_img.shape[0], test_img.shape[1]

        calib = {}
        calib['Tr'] = np.eye(4)
        poses = read_poses_file(pose_file_path, calib)

        image_filenames = natsorted(os.listdir(image_path))

        camera_angle_x = 2 * np.arctan(0.5 * W / fx)

        subsection_folders = natsorted(os.listdir(os.path.join(parent_folder, sequence, '3dgs_input')))
        subsection_folders = [name for name in subsection_folders if '.' not in name]
        
        for subsection in subsection_folders:
            info = subsection.split('_')
            start = int(info[0])
            # end = int(info[1])
            end = start
            stride = int(info[2])

            output_path = os.path.join(parent_folder, sequence, '3dgs_input', subsection, 'transforms_test.json')

            result = {}
            result['camera_angle_x'] = camera_angle_x
            result['frames'] = []

            process_bar = tqdm(range(start, end, stride))
            
            for i in range(start, end, stride):
                filename = image_filenames[i]
                # if index < 100:
                #     index += 1
                #     continue
                # if index > 150:
                #     break
                frame_setting = {}
                frame_setting['file_path'] = os.path.join(image_path, filename[:-4])
                frame_setting['transform_matrix'] = poses[i].tolist()
                result['frames'].append(frame_setting)
                process_bar.update()
                # if index == 600:
                #     break
            
            with open(output_path, 'w') as file:
                json.dump(result, file, indent=4)