# -*- coding: utf-8 -*
import numpy as np
import time
import os
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms  as transforms
import sys
import cv2

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
sys.path.append(current_dir)
from third_party.TransVPR.feature_extractor import Extractor_base
from third_party.TransVPR.blocks import POOL

def transform(img_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size[0], img_size[1]]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def inference(clip_folder):
    image_folder = os.path.join(clip_folder, 'images')
    
    output_path = os.path.join(clip_folder, 'global_features.txt')
    output_f = open(output_path, 'w')
    image_names = os.listdir(image_folder)

    ckpt = os.path.join(current_dir, './TransVPR_MSLS.pth')
    checkpoint = torch.load(ckpt)
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module('pool', pool)
    model.load_state_dict(checkpoint)

    img_size = np.array([480,640])
    N_patch = img_size//(2**4)
    input_transform = transform(img_size)

    for image_name in tqdm(image_names, total=len(image_names)):
        image_path = os.path.join(image_folder, image_name)
        print(image_path)
        img = Image.open(image_path)
        # print(np.asarray(img)[:10, :10, 0])
        # print(np.asarray(img).shape)
        # img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # print(img[:10, :10, 0])
        # print(img.shape)
        # exit()
        # print(img.shape)
        img = input_transform(img)

        img = img[None, ...]
        # img = np.repeat(img, 8, 0)
        # print(img.shape)

        # print(input) 
        start_time = time.time()
        patch_feat = model(img)
        # print(patch_feat.shape)
        # print(patch_feat[0, 2, 100, :20])
        global_feat, attention_mask = model.pool(patch_feat)
        end_time = time.time()  
        # print('run time = {}'.format(end_time - start_time))
        # print(global_feat.shape)
        # print(attention_mask.shape)
        # print(global_feat[0, :])

        global_feat = global_feat.detach().numpy()[0, :]

        output_str = '{} '.format(image_name)
        for d in range(global_feat.shape[0]):
            output_str = output_str + '{} '.format(global_feat[d])
        output_str = output_str[:-1] + '\n'
        output_f.write(output_str)

    output_f.close()


if __name__ == "__main__":

    base_folder = ''
    vpr_model_path = 'TransVPR_MSLS.pth'
    
    global_feature_folder = os.path.join(base_folder, 'global_features')
    os.makedirs(global_feature_folder, exist_ok=True)

    image_folder = os.path.join(base_folder, 'image_2')
    image_names = os.listdir(image_folder)

    checkpoint = torch.load(vpr_model_path)
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module('pool', pool)
    model.load_state_dict(checkpoint)

    img_size = np.array([480,640])
    N_patch = img_size//(2**4)
    input_transform = transform(img_size)

    for image_name in tqdm(image_names, total=len(image_names)):
        image_path = os.path.join(image_folder, image_name)

        img = Image.open(image_path)
        img = input_transform(img)
        img = img[None, ...]
        # img = np.repeat(img, 8, 0)
        # print(img.shape)

        # print(input) 
        start_time = time.time()
        patch_feat = model(img)
        # print(patch_feat.shape)
        # print(patch_feat[0, 2, 100, :20])
        global_feat, attention_mask = model.pool(patch_feat)
        end_time = time.time()  
        # print('run time = {}'.format(end_time - start_time))
        # print(global_feat.shape)
        # print(attention_mask.shape)
        # print(global_feat[0, :])

        global_feat = global_feat.detach().numpy()[0, :]
        print(global_feat.shape)

        np.save(os.path.join(global_feature_folder, image_name[:-4]), global_feat)
