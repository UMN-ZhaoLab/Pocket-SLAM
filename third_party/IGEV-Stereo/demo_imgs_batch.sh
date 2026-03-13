#!/bin/bash

# 批量执行IGEV生成视差图

folderA="euroc/mh02/mav0/cam0/data_rect"
folderB="euroc/mh02/mav0/cam1/data_rect"

filesA=($(ls "$folderA"))

for file in "${filesA[@]}"; do
    fileA="$folderA/$file"
    fileB="$folderB/$file"

    if [ -f "$fileB" ]; then
        echo "processing $fileA ..."
        python demo_imgs.py --restore_ckpt pretrained_models/middlebury.pth -l=$fileA -r=$fileB --save_numpy
        mv demo-output/image_2.png image_2_disparity/image/$file
        filename=$(basename $file .png)
        mv demo-output/image_2.npy image_2_disparity/numpy/$filename.npy
    else
        echo "文件 $fileB 不存在"
    fi
done