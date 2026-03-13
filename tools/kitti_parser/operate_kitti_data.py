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
import shutil


sequences = [str(i).zfill(2) for i in range(11, 22)]
# sequences = ['01', '02', '04', '07', '08', '09', '10']
print(sequences)

image_folder = "" # path to kitti dataset
pose_folder = '' # path to kitti pose files
igev_kitti_model_path = '../../third_party/IGEV-Stereo/pretrained_models/kitti15.pth'
vpr_model_path = '../../third_party/TransVPR/TransVPR_MSLS.pth'

for sequence in sequences:
    print(sequence)
    
    run_depth_sgbm = False
    run_depth_igev = True
    run_get_gt_pose = True
    run_global_feature = True

    calib_file_path = os.path.join(image_folder, sequence, "calib.txt")

    left_images_folder = os.path.join(image_folder, sequence, "image_2")
    right_images_folder = os.path.join(image_folder, sequence, "image_3")

    left_images_path = os.listdir(left_images_folder)
    right_images_path = os.listdir(right_images_folder)
    left_images_path = sorted(left_images_path, key=lambda x: float(x[:-4]))
    right_images_path = sorted(right_images_path, key=lambda x: float(x[:-4]))
    number_of_images = len(left_images_path)

    if run_get_gt_pose:
        if int(sequence) <= 10:
            pose_file_path = os.path.join(pose_folder, "{}.txt".format(sequence))
            gt_pose_file_path = os.path.join(image_folder, sequence, "traj.txt")
            shutil.copy(pose_file_path, gt_pose_file_path)
        else:
            gt_pose_file_path = os.path.join(image_folder, sequence, "traj.txt")
            f = open(gt_pose_file_path, 'w')
            output_str = "1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0\n"
            for iidx in range(number_of_images):
                f.write(output_str)
            f.close()


    calib_f = open(calib_file_path, "r")
    calib_lines = calib_f.readlines()
    fx_image_2 = float(calib_lines[2].split(' ')[1])
    K_image_2 = np.array([float(d) for d in calib_lines[2].split(' ')[1:]]).reshape([3, 4])[:3, :3]
    baseline_plus_fx_image_2 = float(calib_lines[2].split(' ')[4])
    baseline_plus_fx_image_3 = float(calib_lines[3].split(' ')[4])
    baseline_image2_image3 = (abs(baseline_plus_fx_image_2) + abs(baseline_plus_fx_image_3)) / fx_image_2  # 0.5323318578407914

    # fx_image_0 = float(calib_lines[0].split(' ')[1])
    # K_image_0 = np.array([float(d) for d in calib_lines[0].split(' ')[1:]]).reshape([3, 4])[:3, :3]
    # baseline_plus_fx_image_1 = float(calib_lines[1].split(' ')[4])
    # baseline_image0_image1 = abs(baseline_plus_fx_image_1) / fx_image_0  # 0.5371657188644179

    fx = float(fx_image_2)
    baseline = float(baseline_image2_image3)
    K = K_image_2
    print(fx)
    print(baseline)
    print(K)

    width = 1241
    height = 376


    if run_depth_sgbm:
        sgbm_depth_folder = os.path.join(image_folder, sequence, 'depth_sgbm')
        os.makedirs(sgbm_depth_folder, exist_ok=True)
        for i in tqdm(range(number_of_images)):
            cam0_image_path = left_images_path[i]
            cam1_image_path = right_images_path[i]

            cam0_image_rect = cv2.imread(os.path.join(left_images_folder, cam0_image_path), 0)
            cam1_image_rect = cv2.imread(os.path.join(right_images_folder, cam1_image_path), 0)
            
            stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
            stereo.setUniquenessRatio(40)
            disparity = stereo.compute(cam0_image_rect, cam1_image_rect) / 16.0
            disparity[disparity == 0] = 1e10
            depth = baseline * fx / disparity
            depth[depth < 0] = 0
            sgbm_depth_path = os.path.join(sgbm_depth_folder, cam0_image_path[:-4] + '.npy')
            np.save(sgbm_depth_path, depth)

    if run_depth_igev:
        igev_disparity_folder = os.path.join(image_folder, sequence, 'disparity_sceneflow')
        os.makedirs(igev_disparity_folder, exist_ok=True)
        igev_depth_folder = os.path.join(image_folder, sequence, 'depth_sceneflow')
        os.makedirs(igev_depth_folder, exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

        # Architecture choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
        args = parser.parse_args()
        
        model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        model.load_state_dict(torch.load(igev_kitti_model_path))

        model = model.module
        model.to(DEVICE)
        model.eval()

        def load_image(imfile):
            img = np.array(Image.open(imfile)).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.dstack([img, img, img])
            print(img.shape)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].to(DEVICE)

        with torch.no_grad():
            for i in tqdm(range(number_of_images)):
                cam0_image_path = left_images_path[i]
                cam1_image_path = right_images_path[i]

                cam0_image_rect = load_image(os.path.join(left_images_folder, cam0_image_path))
                cam1_image_rect = load_image(os.path.join(right_images_folder, cam1_image_path))

                padder = InputPadder(cam0_image_rect.shape, divis_by=32)
                cam0_image_rect, cam1_image_rect = padder.pad(cam0_image_rect, cam1_image_rect)

                disp = model(cam0_image_rect, cam1_image_rect, iters=args.valid_iters, test_mode=True)
                disp = disp.cpu().numpy()
                disp = padder.unpad(disp)
                filename = os.path.join(igev_disparity_folder, cam0_image_path)
                plt.imsave(filename, disp.squeeze(), cmap='jet')
                np.save(os.path.join(igev_disparity_folder, cam0_image_path[:-4]), disp.squeeze())
                
                depth = baseline * fx / disp.squeeze()
                depth[depth < 0.1] = 0
                np.save(os.path.join(igev_depth_folder, cam0_image_path[:-4]), depth)

    if run_global_feature:
        def transform(img_size):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([img_size[0], img_size[1]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        global_feature_folder = os.path.join(image_folder, sequence, "global_features")
        os.makedirs(global_feature_folder, exist_ok=True)

        image_names = os.listdir(left_images_folder)

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
            image_path = os.path.join(left_images_folder, image_name)

            img = Image.open(image_path)
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