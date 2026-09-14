"""Full EuRoC V2_01_easy run (official range 0-2200, stride 5).

Paper tables use MH01–MH05; this is the complete available Vicon sequence
with IGEV depth + paper hyperparams + quality-leaning Pocket prune.
"""
primary_device = "cuda:0"
seed = 0
scene_name = "V2_01_easy"

map_every = 1
keyframe_every = 1
mapping_window_size = 24

# Paper Implementation Details
tracking_iters = 50
mapping_iters = 100

image_width = 752
image_height = 480

# Official V2_01_easy range from run_euroc_sequence.bash
start_idx = 0
end_idx = 2200
stride = 5

group_name = "euroc_paper_full"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"

config = dict(
    workdir=f"{group_name}",
    run_name=run_name,
    scene_path=f"experiments/euroc/{run_name}/params.npz",
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=500,
    eval_every=1,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    use_warp_loss=True,
    weight_warp=100,
    use_grad_mask=False,
    opt_local_map=False,
    run_loop_closure=True,
    use_wandb=False,
    pixel_gs_depth_gamma=0.37,
    wandb=dict(
        entity="",
        project="",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=False,
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
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        icp_corr_threshold=0.5,
        loss_weights=dict(im=1.0, depth=1.0),
        lrs=dict(
            means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0,
            logit_opacities=0.0, log_scales=0.0,
            cam_unnorm_rots=0.0004, cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.5),
        lrs=dict(
            means3D=0.0001, rgb_colors=0.0025, unnorm_rotations=0.001,
            logit_opacities=0.05, log_scales=0.001,
            cam_unnorm_rots=0.0000, cam_trans=0.0000,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0, remove_big_after=0, stop_after=20,
            prune_every=20, removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False, reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500, remove_big_after=3000, stop_after=5000,
            densify_every=100, grad_thresh=0.0002, num_to_split_into=2,
            removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    pocket_slam=dict(
        enable=True,
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
        render_mode="color", offset_first_viz_cam=True, show_sil=False,
        visualize_cams=False, viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0, view_scale=2,
        viz_fps=5, enter_interactive_post_online=False,
    ),
)
