import os
from os.path import join as p_join
from datetime import datetime

scenes = ["MH_01_easy"]

primary_device="cuda:0"
seed = 0
scene_name = 'MH_01_easy'

map_every = 1
keyframe_every = 1
mapping_window_size = 24 #default: 24

# Paper (Sec. IV Implementation Details): 50 tracking / 100 mapping iters
tracking_iters = 50
mapping_iters = 100
run_loop_closure = True

image_width = 752
image_height = 480

# MH01 paper range (see bash_scripts/run_euroc_sequence.bash)
# MH_01_easy: 900–3630; MH_02: 780–2990; MH_03: 350–2600;
# MH_04: 368–1970; MH_05: 400–2220. stride=5.
start_idx = 900
end_idx = 3630
stride = 5

group_name = "euroc_paper"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"

config = dict(
    workdir=f"{group_name}",
    run_name=run_name,
    scene_path=f"experiments/euroc/{run_name}/params.npz",
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
    weight_warp=100,
    use_grad_mask=False,
    opt_local_map=False,
    run_loop_closure=run_loop_closure,
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
        basedir=f"euroc/{scene_name}/mav0/cam0",
        gradslam_data_cfg="./configs/euroc/euroc.yaml",
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
            depth=1.0,  # paper λ_d = 1.0 (Eq. 3)
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
            depth=1.5,  # paper λ_d* = 1.5 (Eq. 7)
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
        # Paper default N_tar_ratio=0.4; 0.7 leans quality vs compression.
        N_tar_ratio=0.7,
        N_tar=None,
        B_min=5,
        B_max=200,
        tile_size=16,
        global_cap=False,
        protect_age=0,
        warmup_frames=1,
        prune_interval=1,
        prune_margin=1.0,
        score_opacity_weight=0.0,
        post_prune_iters=30,
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
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