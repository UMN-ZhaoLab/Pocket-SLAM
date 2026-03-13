import os
from os.path import join as p_join
from datetime import datetime

scenes = ["06"]

primary_device="cuda:0"
seed = 0
scene_name = '03'

map_every = 1
keyframe_every = 1
mapping_window_size = 2 #default: 24

tracking_iters = 100
mapping_iters = 100

kitti_yaml = './configs/kitti/kitti03.yaml'
image_width = 1242
image_height = 375

start_idx = 750
end_idx = 800
stride = 2

group_name = "kitti7"
# ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# run_name = f"{scene_name}_{seed}"
# run_name = f"{scene_name}"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"
# run_name = run_name + '_' + ts

config = dict(
    workdir=f"{group_name}",
    run_name=run_name,
    scene_path=f'',
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=1, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_warp_loss=True,
    weight_warp=10,
    # weight_warp=100,
    use_grad_mask=False,
    opt_local_map=False,
    use_wandb=False,
    pixel_gs_depth_gamma=0.37,
    wandb=dict(
        entity="",
        project="",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="kitti/sequences",
        # basedir="",
        gradslam_data_cfg=kitti_yaml,
        sequence=scene_name,
        desired_image_height=image_height,
        desired_image_width=image_width,
        start=start_idx,
        end=end_idx,
        stride=stride,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        icp_corr_threshold=0.5,
        loss_weights=dict(
            im=1.0,
            depth=0.2,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    pocket_slam=dict(
        enable=True,
        # Target Gaussian count after pruning (N_tar in the paper).
        # Set to -1 to auto-derive as 60% of current count each round.
        N_tar=80000,
        # Per-tile min/max survival budget (B_min, B_max).
        B_min=1,
        B_max=2000,
        # Tile size in pixels (matches CUDA rasterizer default).
        tile_size=16,
    ),
    viz=dict(
        render_mode='centers', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=2560, viz_h=1600, # 2560*1600 default: viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)