import torch
from gmc_utils.pose_utils import normalize_quaternion, quaternions_to_rotation_matrices


def construct_database(gaussians, motion_model, quat_model, motion_query_func, 
                       scale_factors, i, args, indices=None):

    scale_rgb, scale_features, scale_potentials, scale_xyz, scale_feature_pca = scale_factors

    if indices is None:
        pnm_points_input = gaussians.get_xyz
    else:
        pnm_points_input = gaussians.get_xyz[indices]

    mask = torch.bernoulli(torch.full(pnm_points_input.size(), 1 - args.gmc.dropout_ratio))
    pnm_points_input = pnm_points_input * mask.to(pnm_points_input.device)
    pnm_feature_term_input = gaussians.get_semantic_feature[:, args.gs.feat_dim:]

    if indices is not None:
        pnm_feature_term_input = pnm_feature_term_input[indices]

    pnm_points_input = pnm_points_input.detach()
    pnm_feature_term_input = pnm_feature_term_input.detach()
        
    potentials = motion_query_func(pnm_points_input * scale_xyz * args.gmc.xyz_input_scale, 
                                   pnm_feature_term_input * scale_feature_pca * args.gmc.dino_input_scale, 
                                   motion_model)

    if indices is None:
        energy_xyz = gaussians.get_xyz * scale_xyz
        energy_rgb = gaussians.get_rgb
        energy_features = gaussians.get_semantic_feature[:, :args.gs.feat_dim]
    else:
        energy_xyz = gaussians.get_xyz[indices] * scale_xyz
        energy_rgb = gaussians.get_rgb[indices]
        energy_features = gaussians.get_semantic_feature[:, :args.gs.feat_dim][indices]

    energy_xyz = energy_xyz.detach()

    quaternion = motion_query_func(pnm_points_input * scale_xyz * args.gmc.xyz_input_scale, 
                                    pnm_feature_term_input * scale_feature_pca * args.gmc.dino_input_scale, 
                                    quat_model)
    quaternion = normalize_quaternion(quaternion)
    rot_matrix = quaternions_to_rotation_matrices(quaternion)
    energy_xyz = torch.matmul(rot_matrix, energy_xyz.unsqueeze(-1)).squeeze(-1)
        
    energy_rgb = energy_rgb.detach()
    energy_features = energy_features.detach()
    
    database = torch.cat([energy_rgb * scale_rgb * args.database.color_dist_weight, 
                          energy_features * scale_features * args.database.feat_dist_weight, 
                          (potentials + energy_xyz) * scale_potentials * args.database.xyz_dist_weight], -1)

    return rot_matrix, potentials, energy_xyz, database


def get_noise_for_database(sample_size, args):
    if args.database.noise_type == 'gumbel':
        mu = args.database.noise_gumbel_mu
        beta = args.database.noise_gumbel_beta
        eps_noise = torch.distributions.gumbel.Gumbel(mu, beta).sample((sample_size, 1))
        eps_noise = torch.sqrt(eps_noise + torch.max(-eps_noise))
    elif args.database.noise_type == 'zero':
        eps_noise = torch.zeros((sample_size, 1))

    return eps_noise
    