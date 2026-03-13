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

# # for M
cam0_raw_K = np.array([458.654, 0, 367.215, 0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape([3, 3])
cam0_raw_distort = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])

cam0_opt_K = np.array([435.2046959714599, 0, 367.4517211914062, 0, 435.2046959714599, 252.2008514404297, 0.0, 0.0, 1.0]).reshape([3, 3])
cam0_opt_distort = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
cam0_R = np.array([0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
                    0.001365741834644127, 0.9999741760894847, 0.007055629199258132,
                    -0.008089410156878961, -0.007044357138835809, 0.9999424675829176]).reshape([3, 3])


cam1_raw_K = np.array([457.587, 0, 379.999, 0, 456.134, 255.238, 0.0, 0.0, 1.0]).reshape([3, 3])
cam1_raw_distort = np.array([-0.28368365, 0.07451284, -0.00010473, 0.00025262, 0.0])

cam1_opt_K = np.array([435.2046959714599, 0, 367.4517211914062, 0, 435.2046959714599, 252.2008514404297, 0.0, 0.0, 1.0]).reshape([3, 3])
cam1_opt_distort = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
cam1_R = np.array([0.9999633527347896, -0.003625811871560086, 0.007755443660172947,
         0.003680398547259526, 0.9999684752771629, -0.007035845251224894, 
         -0.007729688520722713, 0.007064130529506649, 0.999945173484644]).reshape([3, 3])

T_i_c0 = np.array(
    [
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

baseline_times_fx = 47.90639384423901  # Following ORB-SLAM2 config, baseline*fx

width = 752
height = 480

cam0_mapx, cam0_mapy = cv2.initUndistortRectifyMap(
    cam0_raw_K,
    cam0_raw_distort,
    cam0_R,
    cam0_opt_K,
    (width, height),
    cv2.CV_32FC1,
)

cam1_mapx, cam1_mapy = cv2.initUndistortRectifyMap(
    cam1_raw_K,
    cam1_raw_distort,
    cam1_R,
    cam1_opt_K,
    (width, height),
    cv2.CV_32FC1,
)

# # params
base_path = '' # path to euroc dataset
igev_sceneflow_model_path = '../../third_party/IGEV-Stereo/pretrained_models/sceneflow.pth'
vpr_model_path = '../../third_party/TransVPR/TransVPR_MSLS.pth'

scene_names = [
    "V2_01_easy",
]

for scene_name in scene_names:
    print(scene_name)
    dataset_path = os.path.join(base_path, scene_name)
    run_stereo_rectify = True
    run_depth_sgbm = False
    run_depth_igev = True
    run_get_gt_pose = True
    run_global_feature = True

    cam0_images_folder = os.path.join(dataset_path, 'mav0', 'cam0', 'data')
    cam1_images_folder = os.path.join(dataset_path, 'mav0', 'cam1', 'data')
    cam0_rect_folder = os.path.join(dataset_path, 'mav0', 'cam0', 'data_rect')
    cam1_rect_folder = os.path.join(dataset_path, 'mav0', 'cam1', 'data_rect')
    os.makedirs(cam0_rect_folder, exist_ok=True)
    os.makedirs(cam1_rect_folder, exist_ok=True)

    cam0_images_path = os.listdir(cam0_images_folder)
    cam1_images_path = os.listdir(cam1_images_folder)

    number_of_images = len(cam0_images_path)
    print(len(cam0_images_path))
    print(len(cam1_images_path))

    if len(cam0_images_path) != len(cam1_images_path):

        for image_path in cam0_images_path:
            if image_path not in cam1_images_path:
                print(image_path)
                os.remove(os.path.join(cam0_images_folder, image_path))
        print('--------------------------')
        for image_path in cam1_images_path:
            if image_path not in cam0_images_path:
                print(image_path)
                os.remove(os.path.join(cam1_images_folder, image_path))
        
        cam0_images_path = os.listdir(cam0_images_folder)
        cam1_images_path = os.listdir(cam1_images_folder)
        if len(cam0_images_path) != len(cam1_images_path):
            print('cam0 and cam1 are not equal')
            exit()

    cam0_images_path = sorted(cam0_images_path, key=lambda x: float(x[:-4]))
    cam1_images_path = sorted(cam1_images_path, key=lambda x: float(x[:-4]))

    if run_stereo_rectify:
        for i in tqdm(range(number_of_images)):
            cam0_image_path = cam0_images_path[i]
            cam1_image_path = cam1_images_path[i]

            cam0_image = cv2.imread(os.path.join(cam0_images_folder, cam0_image_path), 0)
            cam1_image = cv2.imread(os.path.join(cam1_images_folder, cam1_image_path), 0)
            cam0_image_rect = cv2.remap(cam0_image, cam0_mapx, cam0_mapy, cv2.INTER_LINEAR)
            cam1_image_rect = cv2.remap(cam1_image, cam1_mapx, cam1_mapy, cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(cam0_rect_folder, cam0_image_path), cam0_image_rect)
            cv2.imwrite(os.path.join(cam1_rect_folder, cam1_image_path), cam1_image_rect)

    if run_depth_sgbm:
        sgbm_depth_folder = os.path.join(dataset_path, 'mav0/cam0/depth_sgbm')
        os.makedirs(sgbm_depth_folder, exist_ok=True)
        for i in tqdm(range(number_of_images)):
            cam0_image_path = cam0_images_path[i]
            cam1_image_path = cam1_images_path[i]

            cam0_image_rect = cv2.imread(os.path.join(cam0_rect_folder, cam0_image_path), 0)
            cam1_image_rect = cv2.imread(os.path.join(cam1_rect_folder, cam1_image_path), 0)
            
            stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
            stereo.setUniquenessRatio(40)
            disparity = stereo.compute(cam0_image_rect, cam1_image_rect) / 16.0
            disparity[disparity == 0] = 1e10
            depth = baseline_times_fx / disparity
            depth[depth < 0] = 0
            sgbm_depth_path = os.path.join(sgbm_depth_folder, cam0_image_path[:-4] + '.npy')
            np.save(sgbm_depth_path, depth)

    if run_depth_igev:
        igev_disparity_folder = os.path.join(dataset_path, 'mav0/cam0/disparity_sceneflow')
        os.makedirs(igev_disparity_folder, exist_ok=True)
        igev_depth_folder = os.path.join(dataset_path, 'mav0/cam0/depth_sceneflow')
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
        model.load_state_dict(torch.load(igev_sceneflow_model_path))

        model = model.module
        model.to(DEVICE)
        model.eval()

        def load_image(imfile):
            img = np.array(Image.open(imfile)).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.dstack([img, img, img])
            # print(img.shape)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].to(DEVICE)

        with torch.no_grad():
            for i in tqdm(range(number_of_images)):
                cam0_image_path = cam0_images_path[i]
                cam1_image_path = cam1_images_path[i]

                cam0_image_rect = load_image(os.path.join(cam0_rect_folder, cam0_image_path))
                cam1_image_rect = load_image(os.path.join(cam1_rect_folder, cam1_image_path))

                padder = InputPadder(cam0_image_rect.shape, divis_by=32)
                cam0_image_rect, cam1_image_rect = padder.pad(cam0_image_rect, cam1_image_rect)

                disp = model(cam0_image_rect, cam1_image_rect, iters=args.valid_iters, test_mode=True)
                disp = disp.cpu().numpy()
                disp = padder.unpad(disp)
                filename = os.path.join(igev_disparity_folder, cam0_image_path)
                plt.imsave(filename, disp.squeeze(), cmap='jet')
                np.save(os.path.join(igev_disparity_folder, cam0_image_path[:-4]), disp.squeeze())
                
                depth = baseline_times_fx / disp.squeeze()
                depth[depth < 0.1] = 0
                np.save(os.path.join(igev_depth_folder, cam0_image_path[:-4]), depth)

    if run_get_gt_pose:
        pose_file_path = os.path.join(dataset_path, 'mav0/state_groundtruth_estimate0/data.csv')

        output_pose_file_path = os.path.join(dataset_path, 'mav0/cam0/traj.txt')

        images_path = os.listdir(cam0_rect_folder)
        images_path = sorted(images_path, key=lambda x:float(x[:-4]))

        with open(pose_file_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        print(data.shape)

        pose_ts = data[:, 0]
        pose_indices = []
        for i in range(len(images_path)):
            depth_ts = float(images_path[i].split(".")[0])
            # print(depth_ts)
            k = np.argmin(np.abs(pose_ts - depth_ts))
            diff = np.min(np.abs(pose_ts - depth_ts))
            if diff > 1e6:
                print('wrong, need skip', diff, images_path[i])
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

    if run_global_feature:
        def transform(img_size):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([img_size[0], img_size[1]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        global_feature_folder = os.path.join(dataset_path, 'mav0/cam0/global_features')
        os.makedirs(global_feature_folder, exist_ok=True)

        image_names = os.listdir(cam0_rect_folder)

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
            image_path = os.path.join(cam0_rect_folder, image_name)

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