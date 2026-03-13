# 视差转深度
import os
import numpy as np
from natsort import natsorted

# distance from 2 to 3: 0.5327173276050372
# f of 2: 958.35805

disparity_folder = 'euroc/mh02/mav0/cam0/disparity_sceneflow'
disparity_filenames = natsorted(os.listdir(disparity_folder))

depth_folder = 'euroc/mh02/mav0/cam0/depth_sceneflow'
os.makedirs(depth_folder, exist_ok=True)

# tmp = 0.5327173276050372 * 958.35805
# tmp = 0.5327173276050372 * 7.188560000000e+02
tmp = 47.90639384423901  ## Euroc, Following ORB-SLAM2 config, baseline*fx
for filename in disparity_filenames:
    if 'npy' not in filename:
        continue
    print('processing: ' + filename)
    disparity_path = os.path.join(disparity_folder, filename)
    disparity = np.load(disparity_path)
    depth = tmp / disparity
    depth[depth < 0.1] = 0
    np.save(os.path.join(depth_folder, filename), depth)