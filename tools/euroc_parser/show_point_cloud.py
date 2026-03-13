import cv2
import numpy as np
import os
import torch
import sys
from skimage import io
import open3d as o3d
import numpy as np
import copy
import shutil
from tqdm import tqdm

current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(current_dir)


def show_point_cloud(pc_data, step=1, gamma=1.0, name="window", save=False):
    """

    :param pc_data: N * 6 (x y z, i i i), intensity within [0, 1]
    :return:
    """

    # # centered
    pc_data = pc_data.astype(np.float64)
    pc_center = np.sum(pc_data[:, :3], axis=0) / pc_data.shape[0]
    pc_center = pc_center.reshape([1, 3])
    print("pc_center is {}".format(pc_center))
    pc_data[:, :2] = pc_data[:, :2] - pc_center[:, :2]

    show_list_intensity = []

    print(pc_data.shape)
    print(np.max(pc_data, axis=0))
    print(np.min(pc_data, axis=0))

    tmp_points = pc_data[::step, :3]
    tmp_colors = pc_data[::step, 3:6]

    tmp_colors = np.power(tmp_colors, gamma)

    tmp_pcd_i = o3d.geometry.PointCloud()
    tmp_pcd_i.points = o3d.utility.Vector3dVector(tmp_points)
    tmp_pcd_i.colors = o3d.utility.Vector3dVector(tmp_colors)

    # # print("3-1. Downsample with a voxel size %.2f" % 0.2)
    # tmp_pcd_i = tmp_pcd_i.voxel_down_sample(0.1)

    # num_points = 20  # 邻域球内的最少点数，低于该值的点为噪声点
    # radius = 0.4   # 邻域半径大小
    # # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    # tmp_pcd_i, ind = tmp_pcd_i.remove_radius_outlier(num_points, radius)

    show_list_intensity.append(tmp_pcd_i)

    # tmp_kp = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    # tmp_kp.compute_vertex_normals()
    # tmp_kp.paint_uniform_color([1.0, 0.0, 0.0])
    # show_list_intensity.append(tmp_kp)

    # coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=5.0, origin=[0, 0, 0])
    # show_list_intensity.append(coords)

    print("Visualize the point cloud.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=960, height=540, left=0, top=0)
    for i in range(len(show_list_intensity)):
        vis.add_geometry(show_list_intensity[i])

    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
    vis.destroy_window()
    print("DONE!")

    if save:
        o3d.io.write_point_cloud("./point_cloud.pcd", tmp_pcd_i)