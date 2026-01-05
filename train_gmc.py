import os
import torch
import wandb
import json
import lpips
import random

import numpy as np

from gmc_utils.pose_utils import *
from gmc_utils.vis_utils import *
from gmc_utils.model_utils import *
from gmc_utils.eval_utils import *
from gmc_utils.knn_utils import *
from gmc_utils.database_utils import *
from gmc_utils.loss_utils import *
from gmc_utils.config import YamlParser

from gmc_models.model import *


def train_step(gaussians1, pipe1, opt1, motion_model1, quat_model1, viewpoint_stack1,
               gaussians2, pipe2, opt2, motion_model2, quat_model2, viewpoint_stack2,
               optimizer_motion, motion_query_func, lpips_model,
               index_gpu, index_gpu_noise, index_gpu_for_gt_motion,
               scale_factors, dim_noise, background, i, args, device, fg_ids,
               log_term3, g_id):
    """
    fg_ids is binary tensor of shape (N, 1) where N is the number of gaussians
    fg_ids[i] = 1 if the i-th gaussian is a foreground gaussian, 0 otherwise
    """
    scale_xyz = scale_factors[3]

    batch_size = min(gaussians1.get_xyz.shape[0], args.train.batch_size)
    feat_dim = args.gs.feat_dim
    
    loss = 0
        
    ### Select random indices for query
    if fg_ids is not None:
        # Select random indices from fg_ids for gaussians1
        # make sure there are batch_size indices selected
        indices = fg_ids[torch.randperm(fg_ids.shape[0])][:batch_size]
    else:
        indices = torch.randperm(gaussians1.get_xyz.shape[0])[:batch_size]
    rot_matrix1, canonical_xyz1, energy_xyz1, query_vecs = construct_database(gaussians1, motion_model1, quat_model1, motion_query_func, scale_factors, i, args, indices)
    query_vecs = torch.cat([query_vecs, torch.zeros((batch_size, 1), device=device)], -1)
    
    # Construct noised database
    rot_matrix2, canonical_xyz2, energy_xyz2, database = construct_database(gaussians2, motion_model2, quat_model2, motion_query_func, scale_factors, i, args, fg_ids)
    eps2_noise = get_noise_for_database(database.shape[0], args)

    if i >= args.train.joint_start_iteration:
        eps2_noise = eps2_noise * 0
    noised_database = torch.cat([database, eps2_noise.to(device)], -1)
    
    update_index_with_new_data(index_gpu_noise, noised_database)
    _, nn_idx_noised = index_gpu_noise.search(query_vecs.detach().cpu().numpy(), 1)  # 1 stands for nearest neighbour
    nn_idx_noised = torch.tensor(nn_idx_noised).to(device)

    selected_database_entries = torch.gather(noised_database, 0, nn_idx_noised.repeat(1, dim_noise))
    
    loss += torch.mean(torch.sum((query_vecs[:, :-1] - selected_database_entries[:, :-1])**2, dim=-1))

    if args.log.log_energy_term_loss:
        log_energy_term_loss(query_vecs, selected_database_entries, feat_dim, i, log_term3=log_term3)

    update_index_with_new_data(index_gpu_for_gt_motion, gaussians1.get_xyz[indices])
    _, nn_idx_xyz = index_gpu_for_gt_motion.search(gaussians1.get_xyz[indices].detach().cpu().numpy(), args.loss.local_agree_loss_neighbour_num + 1)
    nn_idx_xyz = nn_idx_xyz[:, :-1]  # exclude self
    if args.loss.local_agree_loss_neighbour_num == 1:
        nn_idx_xyz = nn_idx_xyz[:, None]
    nn_idx_xyz = torch.tensor(nn_idx_xyz).to(device).squeeze(1)
    
    # Local distance preservation loss
    self_local_distance_loss_weight = args.loss.self_local_distance_weight_start + (args.loss.self_local_distance_weight_end - args.loss.self_local_distance_weight_start) * min(i, args.loss.self_local_distance_weight_steps) / args.loss.self_local_distance_weight_steps
    self_local_distance_loss = local_distance_preservation_loss_canonical_space(canonical_xyz1, energy_xyz1, gaussians1, indices, nn_idx_xyz, scale_xyz)
    loss += self_local_distance_loss_weight * self_local_distance_loss

    cross_local_distance_loss_weight = args.loss.cross_local_distance_weight_start + (args.loss.cross_local_distance_weight_end - args.loss.cross_local_distance_weight_start) * min(i, args.loss.cross_local_distance_weight_steps) / args.loss.cross_local_distance_weight_steps
    cross_local_distance_loss = local_distance_preservation_loss_true_space(gaussians1, canonical_xyz1, canonical_xyz2, rot_matrix1, rot_matrix2, indices, nn_idx_noised, nn_idx_xyz, scale_xyz)
    loss += cross_local_distance_loss_weight * cross_local_distance_loss

    wandb.log({
            "train/self_local_distance_loss": self_local_distance_loss,
            "train/cross_local_distance_loss": cross_local_distance_loss}, step=i)
    
    ### backpropagate to gaussians with 3D and 2D loss
    if i >= args.train.joint_start_iteration:
        # Gaussian 1
        g1_loss_info, g1_grad_info = render_loss(viewpoint_stack1, gaussians1, pipe1, opt1, background, lpips_model, args)

        loss += args.loss.self_render_loss_weight * g1_loss_info["render_loss"]
        
        wandb.run.log({
            f"train/g{g_id}_render_lpips_loss": g1_loss_info["lpips_loss"],
            f"train/g{g_id}_render_rgb_loss": g1_loss_info["render_rgb_loss"],
            f"train/g{g_id}_render_loss": g1_loss_info["render_loss"],
            f"evaluation_render/g{g_id}_psnr": g1_loss_info["psnr_val"]}, step=i)
    
        # Gaussian 1 -> Gaussian 1'
        with torch.no_grad():
            rot_matrix1, canonical_xyz1, energy_xyz1, query_vecs = construct_database(gaussians1, motion_model1, quat_model1, motion_query_func, scale_factors, i, args, fg_ids)
            rot_matrix2, canonical_xyz2, energy_xyz2, database = construct_database(gaussians2, motion_model2, quat_model2, motion_query_func, scale_factors, i, args, fg_ids) 

            pred_scene_flow = get_scene_flow(gaussians1, rot_matrix1, canonical_xyz1, query_vecs, 
                                             gaussians2, rot_matrix2, canonical_xyz2, database, 
                                             scale_xyz, index_gpu, args, i, device, fg_ids)

        gaussians1.add_xyz(pred_scene_flow, fg_ids)
        
        ### Render from Flowed Gaussians and add loss
        g1f_loss_info, g1f_grad_info = render_loss(viewpoint_stack2, gaussians1, pipe1, opt1, background, lpips_model, args)
        loss += args.loss.cross_render_loss_weight * g1f_loss_info["render_loss"]

        # change back to the original positions
        gaussians1.add_xyz(-pred_scene_flow, fg_ids)
        
        wandb.run.log({
            f"train/g{g_id}f_render_lpips_loss": g1f_loss_info["lpips_loss"],
            f"train/g{g_id}f_render_rgb_loss": g1f_loss_info["render_rgb_loss"],
            f"train/g{g_id}f_render_loss": g1f_loss_info["render_loss"],
            f"evaluation_render/g{g_id}f_psnr": g1f_loss_info["psnr_val"]}, step=i)

    wandb.run.log({
            "train/loss": loss,
            "train/iter": i}, step=i)
    
    loss.backward()

    if args.joint.turn_on_desify_prune:
        if i >= args.train.joint_start_iteration and (i - args.train.joint_start_iteration + 1 + 1 < args.joint.desify_prune_iters):
            # Keep track of max radii in image-space for pruning
            gaussians1.max_radii2D[g1_grad_info["visibility_filter"]] = torch.max(gaussians1.max_radii2D[g1_grad_info["visibility_filter"]], g1_grad_info["radii"][g1_grad_info["visibility_filter"]])
            gaussians1.add_densification_stats(g1_grad_info["viewspace_point_tensor"], g1_grad_info["visibility_filter"], g1_grad_info["image_width"], g1_grad_info["image_height"])

            gaussians1.max_radii2D[g1f_grad_info["visibility_filter"]] = torch.max(gaussians1.max_radii2D[g1f_grad_info["visibility_filter"]], g1f_grad_info["radii"][g1f_grad_info["visibility_filter"]])
            gaussians1.add_densification_stats(g1f_grad_info["viewspace_point_tensor"], g1f_grad_info["visibility_filter"], g1f_grad_info["image_width"], g1f_grad_info["image_height"])

    optimizer_motion.step()
    
    if i >= args.train.joint_start_iteration:
        gaussians1.optimizer.step()

    
def train(args):
    feat_dim = args.gs.feat_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = args.log.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_dir = f'{out_dir}/{args.log.expname}'
    os.makedirs(out_dir, exist_ok=True)
    out_dir_vis = f'{out_dir}/vis'
    os.makedirs(out_dir_vis, exist_ok=True)

    # save configs
    args_file = os.path.join(out_dir, 'args.json')
    with open(args_file, 'w') as af:
        json.dump(vars(args), af, indent=4)

    fg_ids = None
    if args.database.fg_info_path is not None:
        fg_ids = np.load(args.database.fg_info_path)[:, 6:7]
        fg_ids = fg_ids.squeeze(1).nonzero()[0]
    
    motion_model1, quat_model1, motion_model2, quat_model2, optimizer_motion, motion_query_func, start, scale_factors = setup_gmc(args.train.load_gmc_ckpt_path, device, args)

    gaussians1, scene1, dataset1, opt1, pipe1 = setup_gaussians(args.gs.pretrained_frame1_dir, args.gs.checkpoint1, args.gs.load_iteration, args.gs.lr)
    gaussians2, scene2, dataset2, opt2, pipe2 = setup_gaussians(args.gs.pretrained_frame2_dir, args.gs.checkpoint2, args.gs.load_iteration, args.gs.lr)
    
    bg_color = [1,1,1] if dataset1.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if scale_factors == []:
        scale_rgb, scale_features, scale_canonical_xyz, scale_xyz, scale_feature_pca = calc_scale_factors(gaussians1, motion_query_func, motion_model1, quat_model1, args)

        scale_factors = [scale_rgb, scale_features, scale_canonical_xyz, scale_xyz, scale_feature_pca]
    else:
        scale_rgb, scale_features, scale_canonical_xyz, scale_xyz, scale_feature_pca = scale_factors

    # KNN tool
    dim_base = 3 + feat_dim + 3  # RGB + DINO + xyz'
    dim_noise = dim_base + 1
    index_gpu, index_gpu_noise, index_gpu_for_gt_motion = setup_faiss(dim_base, dim_noise)

    wandb.init(project="GMC",
               name=args.log.expname,
               tags=args.log.wandb_tags,
               config=args)

    viewpoint_stack1, viewpoint_stack2 = None, None

    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    for i in range(start, args.train.total_iteration + 1):
        if not viewpoint_stack1:
            viewpoint_stack1 = scene1.getTrainCameras().copy()
        if not viewpoint_stack2:
            viewpoint_stack2 = scene2.getTrainCameras().copy()

        optimizer_motion.zero_grad()
        gaussians1.optimizer.zero_grad(set_to_none = True)
        gaussians2.optimizer.zero_grad(set_to_none = True)

        wandb.log({"gaussians/GS1_num": gaussians1.get_xyz.shape[0]}, step=i)
        wandb.log({"gaussians/GS2_num": gaussians2.get_xyz.shape[0]}, step=i)

        if i % 2 == 0:
            opt_g1 = True
        else:
            opt_g1 = False
                
        if opt_g1:
            train_step(gaussians1, pipe1, opt1, motion_model1, quat_model1, viewpoint_stack1,
                       gaussians2, pipe2, opt2, motion_model2, quat_model2, viewpoint_stack2,
                       optimizer_motion, motion_query_func, lpips_model,
                       index_gpu, index_gpu_noise, index_gpu_for_gt_motion,
                       scale_factors, dim_noise, background, i, args, device, fg_ids,
                       True, g_id=1)
        
        if not opt_g1:
            train_step(gaussians2, pipe2, opt2, motion_model2, quat_model2, viewpoint_stack2,
                       gaussians1, pipe1, opt1, motion_model1, quat_model1, viewpoint_stack1,
                       optimizer_motion, motion_query_func, lpips_model,
                       index_gpu, index_gpu_noise, index_gpu_for_gt_motion,
                       scale_factors, dim_noise, background, i, args, device, fg_ids,
                       False, g_id=2)

        ### densify and prune  
        if args.joint.turn_on_desify_prune:
            if i >= args.train.joint_start_iteration and (i - args.train.joint_start_iteration + 1 < args.joint.desify_prune_iters):
                if (i - args.train.joint_start_iteration + 1) > opt1.densify_from_iter and (i - args.train.joint_start_iteration + 1) % opt1.densification_interval == 0:
                    gaussians1.densify_and_prune(opt1.densify_grad_threshold * 2, 0.005, scene1.cameras_extent, 20, densify=args.joint.turn_on_desify)

                if (i - args.train.joint_start_iteration + 1) % opt1.opacity_reset_interval == 0:
                    gaussians1.reset_opacity()

                if (i - args.train.joint_start_iteration + 1) > opt2.densify_from_iter and (i - args.train.joint_start_iteration + 1) % opt2.densification_interval == 0:
                    gaussians2.densify_and_prune(opt2.densify_grad_threshold * 2, 0.005, scene2.cameras_extent, 20, densify=args.joint.turn_on_desify)

                if (i - args.train.joint_start_iteration + 1) % opt2.opacity_reset_interval == 0:
                    gaussians2.reset_opacity()
        
        ### save checkpoints
        if (i+1) % args.log.gmc_ckpt_freq == 0:
            save_gmc_ckpt(motion_model1, quat_model1, motion_model2, quat_model2, optimizer_motion, scale_factors, i, out_dir)

            if i >= args.train.joint_start_iteration:
                save_gaussians(gaussians1, gaussians2, i, out_dir)

            
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    args = YamlParser().args

    train(args)

    print("Training done.")
