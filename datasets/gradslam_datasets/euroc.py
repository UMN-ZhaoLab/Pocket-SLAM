import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class EurocDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 752,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir)
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
            need_sync_pose=True,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/data_rect/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth_sceneflow/*.npy"))
        # depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth_sgbm/*.npy"))
        feature_paths = natsorted(glob.glob(f"{self.input_folder}/global_features/*.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths, feature_paths

    def load_poses(self):
        poses = []
        timestamps = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            tmp_line = line.split(' ')
            timestamp = tmp_line[0]
            c2w = np.zeros((4, 4))
            c2w[:3, :] = np.array([float(d) for d in tmp_line[1:]]).reshape(3, 4)
            c2w[3, 3] = 1.0
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
            timestamps.append(timestamp)
        
        self.timestamps = timestamps
        return poses

    def sync_pose(self):    
        print(len(self.timestamps))
        timestamps_dict = {timestamp: 1 for timestamp in self.timestamps}
        # print(self.timestamps)
        # exit()
        num_no_gt_poses = 0
        keep_color_paths = []
        keep_depth_paths = []
        keep_feature_paths = []
        for iidx, image_path in enumerate(self.color_paths):
            timestamp = os.path.basename(image_path).split('.')[0]
            if timestamp not in timestamps_dict:
                # print(image_path)
                num_no_gt_poses += 1
                continue
        
            keep_color_paths.append(self.color_paths[iidx])
            keep_depth_paths.append(self.depth_paths[iidx])
            
            if (len(self.feature_paths) > 0):
                keep_feature_paths.append(self.feature_paths[iidx])
        print(num_no_gt_poses, len(self.color_paths))
        
        self.color_paths = keep_color_paths
        self.depth_paths = keep_depth_paths
        self.feature_paths = keep_feature_paths
        print(len(self.color_paths))
        print(len(self.depth_paths))
        print(len(self.feature_paths))
        print(len(self.poses))

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)