import cv2
import numpy as np
import time
import random
import math
import os
import open3d as o3d
from tqdm import tqdm
from show_point_cloud import show_point_cloud

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([[7.188560000000e+02,0,6.071928000000e+02],[0,7.188560000000e+02,1.852157000000e+02],[0.,0.,1.]])
right_camera_matrix = np.array([[7.188560000000e+02,0,6.071928000000e+02],[0,7.188560000000e+02,1.852157000000e+02],[0.,0.,1.]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
right_distortion = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

# 旋转矩阵
R = np.array([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]).reshape([3, 3])
# 平移矩阵
T = np.array([-0.5371657188644179, 0, 0])

size = (1241, 376)

output_path = ''
os.makedirs(output_path, exist_ok=True)

data_path = ''

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

for i in tqdm(range(4541)):
    imfile1 = os.path.join(data_path, 'image_0', '{}.png'.format(str(i).zfill(6)))
    imfile2 = os.path.join(data_path, 'image_1', '{}.png'.format(str(i).zfill(6)))
    print(imfile1)
    print(imfile2)
    # exit()    
    
    imgL = cv2.imread(imfile1, cv2.IMREAD_UNCHANGED)
    imgR = cv2.imread(imfile2, cv2.IMREAD_UNCHANGED)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 3
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16
    print(disparity.shape)
    print(threeD.shape)
    
    # cv2.imshow("depth", dis_color)
    # cv2.imshow("left", imgL)
    # cv2.imshow("disp", disp)  # 显示深度图的双目画面
    
    # cv2.waitKey(0)
    
    disparity = disparity.flatten()
    imgL = imgL.flatten()
    threeD = threeD.reshape([-1, 3])
    valid_depth_indices = np.where(disparity > 1)[0]
    threeD = threeD[valid_depth_indices, :]
    imgL = imgL[valid_depth_indices]
    imgL = imgL.reshape([-1, 1]) / 255.0
    # print(threeD.shape)
    # print(imgL.shape)
    # print(np.max(threeD[:, 2]))
    # print(np.min(threeD[:, 2]))
    # exit()
    # threeD[:, 2] *= -1
    valid_depth_indices = np.where((threeD[:, 2] > 0.1) & (threeD[:, 2] < 30))[0]
    threeD = threeD[valid_depth_indices, :]
    imgL = imgL[valid_depth_indices]
    print(threeD.shape)
    pts = np.hstack([threeD, imgL, imgL, imgL])
    # show_point_cloud(pts, step=1)
    
    output_file_path = os.path.join(output_path, '{}'.format(str(i).zfill(6)))
    np.save(output_file_path, pts)
    
    # exit()