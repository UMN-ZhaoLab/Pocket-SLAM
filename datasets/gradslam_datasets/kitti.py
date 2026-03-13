import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class KittiDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/image_2/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth_sceneflow/*.npy"))
        feature_paths = natsorted(glob.glob(f"{self.input_folder}/global_features/*.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths, feature_paths


    # def load_poses(self):
    #     poses = []
    #     with open(self.pose_path, "r") as f:
    #         lines = f.readlines()
    #     for i in range(self.num_imgs):
    #         line = lines[i]
    #         c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
    #         # c2w[:3, 1] *= -1
    #         # c2w[:3, 2] *= -1
    #         c2w = torch.from_numpy(c2w).float()
    #         poses.append(c2w)
    #     return poses
    
    def read_poses_file(self, filename, calibration):
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

    def load_poses(self):
        calib = {}
        calib['Tr'] = np.eye(4)
        poses = self.read_poses_file(self.pose_path, calib)

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)