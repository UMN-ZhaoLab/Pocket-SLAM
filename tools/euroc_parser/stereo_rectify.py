import numpy as np
import cv2
import os
from tqdm import tqdm


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

dataset_path = 'euroc/mh02'
sgbm_depth_folder = 'euroc/mh02/mav0/cam0/depth_sgbm'
os.makedirs(sgbm_depth_folder, exist_ok=True)

cam0_images_folder = os.path.join(dataset_path, 'mav0', 'cam0', 'data')
cam1_images_folder = os.path.join(dataset_path, 'mav0', 'cam1', 'data')

cam0_rect_folder = os.path.join(dataset_path, 'mav0', 'cam0', 'data_rect')
cam1_rect_folder = os.path.join(dataset_path, 'mav0', 'cam1', 'data_rect')
os.makedirs(cam0_rect_folder, exist_ok=True)
os.makedirs(cam1_rect_folder, exist_ok=True)

cam0_images_path = os.listdir(cam0_images_folder)
cam1_images_path = os.listdir(cam1_images_folder)

cam0_images_path = sorted(cam0_images_path, key=lambda x: float(x[:-4]))
cam1_images_path = sorted(cam1_images_path, key=lambda x: float(x[:-4]))

number_of_images = len(cam0_images_path)
print(len(cam0_images_path))
print(len(cam1_images_path))

for i in tqdm(range(number_of_images)):
    cam0_image_path = cam0_images_path[i]
    cam1_image_path = cam1_images_path[i]

    cam0_image = cv2.imread(os.path.join(cam0_images_folder, cam0_image_path), 0)
    cam1_image = cv2.imread(os.path.join(cam1_images_folder, cam1_image_path), 0)
    cam0_image_rect = cv2.remap(cam0_image, cam0_mapx, cam0_mapy, cv2.INTER_LINEAR)
    cam1_image_rect = cv2.remap(cam1_image, cam1_mapx, cam1_mapy, cv2.INTER_LINEAR)

    # cv2.imwrite(os.path.join(cam0_rect_folder, cam0_image_path), cam0_image_rect)
    # cv2.imwrite(os.path.join(cam1_rect_folder, cam1_image_path), cam1_image_rect)

    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
    stereo.setUniquenessRatio(40)
    disparity = stereo.compute(cam0_image_rect, cam1_image_rect) / 16.0
    disparity[disparity == 0] = 1e10
    depth = 47.90639384423901 / (
        disparity
    )  ## Following ORB-SLAM2 config, baseline*fx
    depth[depth < 0] = 0
    sgbm_depth_path = os.path.join(sgbm_depth_folder, cam0_image_path[:-4] + '.npy')
    np.save(sgbm_depth_path, depth)
