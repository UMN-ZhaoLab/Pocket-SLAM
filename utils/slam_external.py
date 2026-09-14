"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def accumulate_mean2d_gradient_pixGS(variables, pixels):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1) * (pixels[variables['seen']].squeeze(-1))
    variables['denom'][variables['seen']] += (pixels[variables['seen']].squeeze(-1))
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    if 'pocket_grad_accum' in variables:
        n_keep = int(to_keep.sum().item())
        # Keep pocket buffers aligned with Gaussian count after densify/prune races
        if variables['pocket_grad_accum'].shape[0] == to_keep.shape[0]:
            variables['pocket_grad_accum'] = variables['pocket_grad_accum'][to_keep]
            variables['pocket_denom'] = variables['pocket_denom'][to_keep]
        else:
            variables['pocket_grad_accum'] = torch.zeros(n_keep, device="cuda").float()
            variables['pocket_denom'] = torch.zeros(n_keep, device="cuda").float()
    return params, variables


def resolve_n_tar(prune_dict, n_gaussians):
    """Paper: N_tar = 0.4 * N_init. Absolute N_tar overrides if provided."""
    if prune_dict.get('N_tar') is not None:
        return int(prune_dict['N_tar'])
    ratio = float(prune_dict.get('N_tar_ratio', 0.4))
    return max(1, int(n_gaussians * ratio))


def project_means_to_image(means_cam, intrinsics):
    """
    Project camera-frame 3D Gaussian centers to pixel coordinates.

    NOTE: rasterizer `means2D` is only an autograd placeholder (stays zeros);
    Pocket-SLAM must project explicitly for tile assignment.
    """
    z = means_cam[:, 2].clamp(min=1e-4)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u = fx * (means_cam[:, 0] / z) + cx
    v = fy * (means_cam[:, 1] / z) + cy
    # (N, 3): u, v, z  — z kept so callers can mask behind-camera points
    return torch.stack([u, v, means_cam[:, 2]], dim=-1)


def compute_tile_budgets(pocket_grad_accum, pocket_denom, means2D,
                          image_height, image_width, N_tar,
                          tile_size=16, B_min=1, B_max=None):
    """
    Tile-Level Budget Mechanism (Pocket-SLAM Sec. III-C).

    Computes per-tile Gaussian survival budgets based on the average gradient
    magnitude of Gaussians projected into each tile during tracking.

    G_k = (1/N_k) * sum_{i in T_k} g_i
    B_k = clip(floor(N_tar * G_k / sum_j G_j), B_min, B_max)

    Empty tiles stay at 0 (B_min only applied to occupied tiles).
    """
    num_tile_rows = (image_height + tile_size - 1) // tile_size
    num_tile_cols = (image_width + tile_size - 1) // tile_size
    num_tiles = num_tile_rows * num_tile_cols

    if B_max is None:
        B_max = N_tar

    uniform_budget = max(B_min, N_tar // max(num_tiles, 1))

    # Per-Gaussian average gradient magnitude
    grads = pocket_grad_accum / (pocket_denom + 1e-8)
    grads = grads.detach()
    grads[grads.isnan()] = 0.0

    px = means2D[:, 0].detach()
    py = means2D[:, 1].detach()
    # Optional depth channel for behind-camera masking
    if means2D.shape[1] >= 3:
        pz = means2D[:, 2].detach()
        in_front = pz > 1e-3
    else:
        in_front = torch.ones_like(px, dtype=torch.bool)

    valid = in_front & (px >= 0) & (px < image_width) & (py >= 0) & (py < image_height)
    if valid.sum() == 0:
        return torch.full((num_tile_rows, num_tile_cols), uniform_budget,
                          device='cuda', dtype=torch.long)

    px_v = px[valid]
    py_v = py[valid]
    g_v  = grads[valid]

    tile_col = (px_v / tile_size).long().clamp(0, num_tile_cols - 1)
    tile_row = (py_v / tile_size).long().clamp(0, num_tile_rows - 1)
    tile_idx = tile_row * num_tile_cols + tile_col

    tile_grad_sum  = torch.zeros(num_tiles, device='cuda')
    tile_count     = torch.zeros(num_tiles, device='cuda')
    tile_grad_sum.scatter_add_(0, tile_idx, g_v)
    tile_count.scatter_add_(0, tile_idx, torch.ones_like(g_v))

    # G_k = average gradient magnitude per tile
    G_k = tile_grad_sum / (tile_count + 1e-8)
    G_total = G_k.sum()

    if G_total < 1e-8:
        # No gradient signal: uniform budget only on occupied tiles
        budgets = torch.zeros(num_tiles, device='cuda', dtype=torch.long)
        occupied = tile_count > 0
        budgets[occupied] = uniform_budget
        return budgets.reshape(num_tile_rows, num_tile_cols)

    raw_budgets = (N_tar * G_k / G_total).floor().long()
    tile_budgets = torch.zeros_like(raw_budgets)
    occupied = tile_count > 0
    tile_budgets[occupied] = raw_budgets[occupied].clamp(min=B_min, max=B_max)
    return tile_budgets.reshape(num_tile_rows, num_tile_cols)


def pocket_slam_prune(params, variables, optimizer, means2D, radius,
                      time_idx, image_height, image_width, prune_dict):
    """
    Pocket-SLAM: Rendering-Area-Aware Pruning + Tile-Level Budget (Sec. III-B/C).

    After mapping converges, prunes Gaussians by:
      1. Computing each Gaussian's rendering-area score:
            C_i = opacity_i * pi * radius_i^2
            S_i = C_i / sum_j C_j
      2. Within each tile, retaining only the top B_k Gaussians (by S_i),
         where B_k comes from the tracking-stage gradient budgets.

    Newly added Gaussians (variables['timestep'] == time_idx) are exempt.
    means2D must be explicitly projected pixel coords (NOT rasterizer placeholder zeros).
    """
    tile_size = prune_dict.get('tile_size', 16)
    N = params['means3D'].shape[0]
    N_tar     = resolve_n_tar(prune_dict, N)
    B_min     = prune_dict.get('B_min', 5)
    B_max     = prune_dict.get('B_max', 200)
    use_global_cap = prune_dict.get('global_cap', False)
    # 0 = protect only Gaussians added at this frame (paper default)
    protect_age = int(prune_dict.get('protect_age', 0))
    warmup_frames = int(prune_dict.get('warmup_frames', 1))
    prune_interval = int(prune_dict.get('prune_interval', 1))
    score_opacity_weight = float(prune_dict.get('score_opacity_weight', 0.0))
    prune_margin = float(prune_dict.get('prune_margin', 1.0))

    # Warmup / interval / already under target
    if time_idx < warmup_frames:
        print(f"[Pocket-SLAM] Skip prune (warmup time_idx={time_idx})")
        return params, variables
    if prune_interval > 1 and (time_idx % prune_interval) != 0:
        print(f"[Pocket-SLAM] Skip prune (interval time_idx={time_idx})")
        return params, variables
    if N <= int(N_tar * prune_margin):
        print(f"[Pocket-SLAM] Skip prune (N={N} <= N_tar*{prune_margin}={int(N_tar * prune_margin)})")
        return params, variables

    num_tile_rows = (image_height + tile_size - 1) // tile_size
    num_tile_cols = (image_width  + tile_size - 1) // tile_size

    # --- Tile budgets (from tracking-stage gradients) ---
    # Do NOT shrink budgets by protected count — that starves mature Gaussians
    # whenever densification adds a large batch and tanks PSNR.
    if 'tile_budgets' in variables:
        tile_budgets = variables['tile_budgets']
    else:
        uniform = max(B_min, N_tar // max(num_tile_rows * num_tile_cols, 1))
        tile_budgets = torch.full((num_tile_rows, num_tile_cols), uniform,
                                  device='cuda', dtype=torch.long)
    tile_budgets_flat = tile_budgets.reshape(-1)

    # --- Score: rendering-area (paper) + optional opacity mix ---
    opacities = torch.sigmoid(params['logit_opacities']).squeeze(-1).detach()
    radii     = radius.detach().float()
    C = opacities * np.pi * (radii ** 2)
    C_total = C.sum()
    if C_total < 1e-8:
        return params, variables
    area_score = C / C_total
    w = min(max(score_opacity_weight, 0.0), 1.0)
    if w > 0:
        op_score = opacities / (opacities.sum() + 1e-8)
        S = (1.0 - w) * area_score + w * op_score
    else:
        S = area_score

    # --- 2D positions (explicit projection; do not use rasterizer placeholder) ---
    px = means2D[:, 0].detach()
    py = means2D[:, 1].detach()
    if means2D.shape[1] >= 3:
        pz = means2D[:, 2].detach()
        in_front = pz > 1e-3
    else:
        in_front = torch.ones(N, dtype=torch.bool, device=px.device)

    # Protect newly added Gaussians (paper: current frame; optional age window)
    new_mask = variables['timestep'] >= (time_idx - protect_age)

    # Old, in-frame, visible Gaussians are candidates for pruning
    in_frame = (in_front &
                (px >= 0) & (px < image_width) &
                (py >= 0) & (py < image_height) &
                (~new_mask) & (radii > 0))

    # Keep new Gaussians and out-of-frame / behind-camera old Gaussians
    to_keep = new_mask.clone()
    to_keep[~in_frame & ~new_mask] = True

    old_in_frame_idx = torch.where(in_frame)[0]
    M = old_in_frame_idx.shape[0]
    budget_sum = int(tile_budgets_flat.sum().item())

    if M > 0:
        px_old = px[old_in_frame_idx]
        py_old = py[old_in_frame_idx]
        S_old  = S[old_in_frame_idx]

        tile_col_idx = (px_old / tile_size).long().clamp(0, num_tile_cols - 1)
        tile_row_idx = (py_old / tile_size).long().clamp(0, num_tile_rows - 1)
        tile_idx = tile_row_idx * num_tile_cols + tile_col_idx

        # Sort by (tile_idx ascending, S descending) → within-tile rank 0 = highest S
        sort_key = tile_idx.float() * 1e8 - S_old
        sort_order = torch.argsort(sort_key)

        sorted_tile_idx = tile_idx[sort_order]

        tile_change = torch.cat([
            torch.ones(1, dtype=torch.bool, device='cuda'),
            sorted_tile_idx[1:] != sorted_tile_idx[:-1]
        ])
        tile_group          = tile_change.cumsum(0) - 1
        tile_start_pos      = torch.where(tile_change)[0]
        within_tile_rank    = (torch.arange(M, device='cuda')
                               - tile_start_pos[tile_group])

        # Per-tile budget; empty tiles stay 0 from compute_tile_budgets
        budgets_per_elem = tile_budgets_flat[sorted_tile_idx].clamp(min=0, max=B_max)
        keep_in_sorted = within_tile_rank < budgets_per_elem

        orig_sorted = old_in_frame_idx[sort_order]
        to_keep[orig_sorted[keep_in_sorted]] = True

    to_remove = ~to_keep
    pruned = int(to_remove.sum().item())
    n_new = int(new_mask.sum().item())
    n_tiles_used = int((tile_budgets_flat > 0).sum().item())

    if pruned > 0:
        params, variables = remove_points(to_remove, params, variables, optimizer)
        torch.cuda.empty_cache()
        print(
            f"[Pocket-SLAM] Pruned {pruned} → remaining: {params['means3D'].shape[0]} "
            f"(N_tar={N_tar}, in_frame={M}, new={n_new}, budget_sum={budget_sum}, "
            f"active_tiles={n_tiles_used})"
        )
    else:
        print(f"[Pocket-SLAM] No prune (in_frame={M}, N={N}, N_tar={N_tar})")

    # Optional global cap (off by default; paper uses per-tile budgets only).
    N_after = params['means3D'].shape[0]
    if use_global_cap and N_after > N_tar:
        grad_score = variables.get('pocket_grad_accum', torch.zeros(N_after, device='cuda'))
        grad_score = grad_score / (variables.get('pocket_denom', torch.ones(N_after, device='cuda')) + 1e-8)
        grad_score = grad_score.detach()
        grad_score[grad_score.isnan()] = 0.0
        if grad_score.max() > 0:
            grad_score = grad_score / (grad_score.max() + 1e-8)

        opacities = torch.sigmoid(params['logit_opacities']).squeeze(-1).detach()
        opacity_score = opacities / (opacities.max() + 1e-8)
        combined = 0.5 * opacity_score + 0.5 * grad_score

        protected = variables['timestep'] >= (time_idx - protect_age)
        keep_mask = torch.zeros(N_after, dtype=torch.bool, device='cuda')
        n_protected = int(protected.sum().item())
        n_slots = max(N_tar - n_protected, 0)

        if n_slots > 0:
            scores = combined.clone()
            scores[protected] = -1.0
            _, top_idx = torch.topk(scores, k=min(n_slots, N_after - n_protected))
            keep_mask[top_idx] = True
        keep_mask |= protected

        to_remove = ~keep_mask
        removed = int(to_remove.sum().item())
        if removed > 0:
            params, variables = remove_points(to_remove, params, variables, optimizer)
            torch.cuda.empty_cache()
            print(f"[Pocket-SLAM] Global cap pruned {removed} → remaining: {params['means3D'].shape[0]}")

    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def prune_gaussians(params, variables, optimizer, iter, prune_dict):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
            torch.cuda.empty_cache()
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
    
    return params, variables


def densify(params, variables, optimizer, iter, densify_dict):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            if 'timestep' in variables:
                selected_timestep_values = variables['timestep'][to_clone]
                duplicated_values = selected_timestep_values.clone()
                variables['timestep'] = torch.cat((variables['timestep'], duplicated_values))

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = densify_dict['num_to_split_into']  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            if params['log_scales'].shape[1] == 3:
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)  # anisotropic
            else:
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)  # isotropic
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            if 'timestep' in variables:
                selected_timestep_values = variables['timestep'][to_split]
                duplicated_values = selected_timestep_values.clone().repeat(n)
                variables['timestep'] = torch.cat((variables['timestep'], duplicated_values))

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def densify_use_pixel_gs(params, variables, optimizer, iter, densify_dict, pixels, split_explore_weight):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient_pixGS(variables, pixels)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            if 'timestep' in variables:
                selected_timestep_values = variables['timestep'][to_clone]
                duplicated_values = selected_timestep_values.clone()
                variables['timestep'] = torch.cat((variables['timestep'], duplicated_values))

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = densify_dict['num_to_split_into']  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            if params['log_scales'].shape[1] == 3:
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)  # anisotropic
            else:
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)  # isotropic
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=split_explore_weight*stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            if 'timestep' in variables:
                selected_timestep_values = variables['timestep'][to_split]
                duplicated_values = selected_timestep_values.clone().repeat(n)
                variables['timestep'] = torch.cat((variables['timestep'], duplicated_values))

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def densify_with_bound(params, variables, optimizer, iter, densify_dict, far_bound, c2ws):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            
            # only clone gs within far_bound
            if 'timestep' in variables:
                c2ws_tmp = torch.tensor(c2ws, dtype=torch.float32, device='cuda')
                traj_pts = c2ws_tmp[variables['timestep'].to(torch.long), :3, 3] # (n, 3)
                distances = torch.norm(params['means3D'] - traj_pts, dim=1) # (n)
                distance_mask = distances < far_bound
                to_clone = to_clone & distance_mask
                clone_num = torch.nonzero(to_clone).shape[0]

            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            if 'timestep' in variables:
                selected_timestep_values = variables['timestep'][to_clone]
                duplicated_values = selected_timestep_values.clone()
                variables['timestep'] = torch.cat((variables['timestep'], duplicated_values))

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])

            n = densify_dict['num_to_split_into']  # number to split into
            # only split gs within far_bound
            if 'timestep' in variables:
                traj_pts = c2ws_tmp[variables['timestep'].to(torch.long), :3, 3] # (n, 3)
                distances = torch.norm(params['means3D'] - traj_pts, dim=1) # (n)
                distance_mask = distances < far_bound
                to_split = to_split & distance_mask
                split_num = torch.nonzero(to_split).shape[0] * n

            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_unnorm_rots', 'cam_trans']}
            if params['log_scales'].shape[1] == 3:
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)  # anisotropic
            else:
                stds = torch.exp(params['log_scales'])[to_split].repeat(n, 3)  # isotropic
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]
            if 'timestep' in variables:
                selected_timestep_values = variables['timestep'][to_split]
                duplicated_values = selected_timestep_values.clone().repeat(n)
                variables['timestep'] = torch.cat((variables['timestep'], duplicated_values))

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def update_learning_rate(optimizer, means3D_scheduler, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = means3D_scheduler(iteration)
                param_group['lr'] = lr
                return lr


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper