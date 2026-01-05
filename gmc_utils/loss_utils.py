import torch
import wandb

from random import randint

from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


def log_energy_term_loss(query_vecs, selected_database_entries, feat_dim, i, log_term3=False):
    with torch.no_grad():
        rgb_loss = torch.mean(torch.sum((query_vecs[:, :3] - selected_database_entries[:, :3])**2, dim=-1))
        feat_loss = torch.mean(torch.sum((query_vecs[:, 3:(3+feat_dim)] - selected_database_entries[:, 3:(3+feat_dim)])**2, dim=-1))
        potential_loss = torch.mean(torch.sum((query_vecs[:, (3+feat_dim):(3+feat_dim+3)] - selected_database_entries[:, (3+feat_dim):(3+feat_dim+3)])**2, dim=-1))
    
        wandb.run.log({
            "train/rgb_loss": rgb_loss,
            "train/feat_loss": feat_loss,
            "train/potential_loss": potential_loss}, step=i)

        if log_term3:
            term3_g1 = torch.mean(torch.sum((query_vecs[:, (3+feat_dim):(3+feat_dim+3)])**2, dim=-1))
            term3_g2 = torch.mean(torch.sum((selected_database_entries[:, (3+feat_dim):(3+feat_dim+3)])**2, dim=-1))

            wandb.run.log({
                "train/term3_g1": term3_g1,
                "train/term3_g2": term3_g2}, step=i)


def render_loss(viewpoint_stack, gaussians, pipe, opt, background, lpips_model, args):
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                
    render_pkg = render(viewpoint_cam, gaussians, background, render_feature=False)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    
    lpips_loss = lpips_model(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
    render_rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + args.loss.lpips_weight * lpips_loss
    render_loss = render_rgb_loss

    with torch.no_grad():
        psnr_val = psnr(image, gt_image).mean().double()

    loss_info = {"lpips_loss": lpips_loss,
                 "render_rgb_loss": render_rgb_loss,
                 "render_loss": render_loss,
                 "psnr_val": psnr_val}

    grad_info = {"viewspace_point_tensor": viewspace_point_tensor,
                 "visibility_filter": visibility_filter,
                 "radii": radii,
                 "image_width": image.shape[2],
                 "image_height": image.shape[1]}

    return loss_info, grad_info


def local_distance_preservation_loss_canonical_space(potentials, energy_xyz, gaussians, indices, nn_idx_xyz, scale_xyz):
    xyz = gaussians.get_xyz[indices].detach() * scale_xyz
    xyz_in_potential_space = potentials + energy_xyz
    
    nn_distances = torch.sum((xyz[:, None, :] - xyz[nn_idx_xyz]) ** 2, dim=-1)
    nn_distances_potential_space = torch.sum((xyz_in_potential_space[:, None, :] - xyz_in_potential_space[nn_idx_xyz]) ** 2, dim=-1)

    # calculate mean of L1 distance
    local_distance_preservation_loss = torch.mean(torch.abs(nn_distances - nn_distances_potential_space))

    return local_distance_preservation_loss


def local_distance_preservation_loss_true_space(gaussians_q, potentials_q, potentials_d, rot_matrix_q, rot_matrix_d, indices, nn_idx_noised, nn_idx_xyz, scale_xyz):
    xyz = gaussians_q.get_xyz[indices].detach() * scale_xyz

    m_d_inv = rot_matrix_d[nn_idx_noised.squeeze(-1)].transpose(1, 2)
    relative_rot_q = m_d_inv @ rot_matrix_q
    rot_applied_q = torch.bmm(relative_rot_q, xyz.unsqueeze(-1)).squeeze(-1)
    relative_trans_q = torch.bmm(m_d_inv, (potentials_q - potentials_d[nn_idx_noised.squeeze(-1)]).unsqueeze(-1)).squeeze(-1)

    pred_scene_flow = rot_applied_q + relative_trans_q - xyz

    xyz_flowed = xyz + pred_scene_flow

    nn_distances = torch.sum((xyz[:, None, :] - xyz[nn_idx_xyz]) ** 2, dim=-1)
    nn_distances_flowed = torch.sum((xyz_flowed[:, None, :] - xyz_flowed[nn_idx_xyz]) ** 2, dim=-1)

    # calculate mean of L1 distance
    local_distance_preservation_loss = torch.mean(torch.abs(nn_distances - nn_distances_flowed))

    return local_distance_preservation_loss
