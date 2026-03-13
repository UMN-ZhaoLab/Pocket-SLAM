import sys
import os

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(base_dir)
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../third_party/IGEV-Stereo")
sys.path.append(base_dir)

import torchvision.transforms  as transforms

from third_party.TransVPR.feature_extractor import Extractor_base
from third_party.TransVPR.blocks import POOL

# sys.path.append('core')
DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import csv
import trimesh


# # params
base_path = ''
igev_sceneflow_model_path = 'third_party/IGEV-Stereo/pretrained_models/sceneflow.pth'
vpr_model_path = 'third_party/TransVPR/TransVPR_MSLS.pth'

scene_names = [
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg2_xyz",
]

for scene_name in scene_names:
    print(scene_name)
    dataset_path = os.path.join(base_path, scene_name)
    run_global_feature = True

    rgb_folder = os.path.join(dataset_path, 'rgb')
    rgb_images_path = os.listdir(rgb_folder)

    number_of_images = len(rgb_images_path)
    print(len(rgb_images_path))

    rgb_images_path = sorted(rgb_images_path, key=lambda x: float(x[:-4]))

    if run_global_feature:
        def transform(img_size):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([img_size[0], img_size[1]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        global_feature_folder = os.path.join(dataset_path, 'global_features')
        os.makedirs(global_feature_folder, exist_ok=True)

        image_names = rgb_images_path

        checkpoint = torch.load(vpr_model_path)
        model = Extractor_base()
        pool = POOL(model.embedding_dim)
        model.add_module('pool', pool)
        model.load_state_dict(checkpoint)
        model = model.to(device=DEVICE)

        img_size = np.array([480,640])
        N_patch = img_size//(2**4)
        input_transform = transform(img_size)

        for image_name in tqdm(image_names, total=len(image_names)):
            image_path = os.path.join(rgb_folder, image_name)

            img = Image.open(image_path)
            img = img.convert("RGB")
            img = input_transform(img)
            img = img[None, ...].to(device=DEVICE)

            # start_time = time.time()
            patch_feat = model(img)
            global_feat, attention_mask = model.pool(patch_feat)
            # end_time = time.time()  
            # print('run time = {}'.format(end_time - start_time))

            global_feat = global_feat.detach().cpu().numpy()[0, :]
            # print(global_feat.shape)
            
            np.save(os.path.join(global_feature_folder, image_name[:-4]), global_feat)