import os
import pickle
import torch
import torch.nn as nn

from gmc_models.model import MotionPotential, MotionQuat
from gmc_utils.pose_utils import normalize_quaternion, quaternions_to_rotation_matrices
from scene import Scene, GaussianModel


def load_args_obj(pretrained_frame_dir):
    args_dir = f'{pretrained_frame_dir}/args'
    
    with open(f'{args_dir}/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
        
    with open(f'{args_dir}/opt.pkl', 'rb') as f:
        opt = pickle.load(f)
        
    with open(f'{args_dir}/pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
        
    return dataset, opt, pipe


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_motion_potential(inputs, features, fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = torch.cat([inputs_flat, features], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def load_gt_points(pretrained_frame_dir, checkpoint_gt, checkpoint, filter_opacities=False, op_threshold=0.5):
    dataset, opt, pipe = load_args_obj(pretrained_frame_dir)
    gaussians = GaussianModel(dataset.sh_degree, dataset.isotropic)

    if checkpoint_gt:
        (model_params, first_iter) = torch.load(checkpoint_gt)
    else:
        (model_params, first_iter) = torch.load(checkpoint)

    gaussians.restore(model_params, opt)

    xyz = gaussians.get_xyz.cpu().detach().numpy()

    if filter_opacities:
        opacity = gaussians.get_opacity.squeeze(1).cpu().detach().numpy()
        xyz = xyz[opacity > op_threshold]

    return xyz


def setup_gaussians(pretrained_frame_dir, checkpoint, load_iteration, gslr_args):
    dataset, opt, pipe = load_args_obj(pretrained_frame_dir)
    gaussians = GaussianModel(dataset.sh_degree, dataset.isotropic)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    scale_fgs_lr(gaussians, gslr_args)

    return gaussians, scene, dataset, opt, pipe


def setup_gmc(gmc_ckpt_path, device, args):
    gmc_lr = args.gmc.learning_rate
    pcadim = args.gs.pca_dim

    netchunk_per_gpu = 262144
    input_ch_xyz = 3
    
    motion_query_func = lambda inputs, features, network_fn : run_motion_potential(inputs, features, network_fn,
                                                                                   netchunk=netchunk_per_gpu)
    
    motion_vars = []
    
    motion_model1 = MotionPotential(input_ch_xyz=input_ch_xyz, input_ch_feature=pcadim, output_ch=3)
    if args.gmc.init_trans_zero:
        motion_model1.initialize_weights()

    motion_model1 = nn.DataParallel(motion_model1.to(device))
    
    for name, param in motion_model1.named_parameters():
        motion_vars.append(param)

    quat_model1 = MotionQuat(input_ch_xyz=input_ch_xyz, input_ch_feature=pcadim, output_ch=4, no_bias=False)
    quat_model1 = nn.DataParallel(quat_model1.to(device))
    
    for name, param in quat_model1.named_parameters():
        motion_vars.append(param)
        
    motion_model2 = MotionPotential(input_ch_xyz=input_ch_xyz, input_ch_feature=pcadim, output_ch=3)
    if args.gmc.init_trans_zero:
        motion_model2.initialize_weights()

    motion_model2 = nn.DataParallel(motion_model2.to(device))

    for name, param in motion_model2.named_parameters():
        motion_vars.append(param)

    quat_model2 = MotionQuat(input_ch_xyz=input_ch_xyz, input_ch_feature=pcadim, output_ch=4, no_bias=False)
    quat_model2 = nn.DataParallel(quat_model2.to(device))
    
    for name, param in quat_model2.named_parameters():
        motion_vars.append(param)
        
    optimizer_motion = torch.optim.Adam(params=motion_vars, lr=gmc_lr, betas=(0.9, 0.999))
    
    start = 0
    scale_factors = []

    if gmc_ckpt_path is not None:
        ### Load checkpoints
        ckpt = torch.load(gmc_ckpt_path)
        
        if ckpt is not None:
            start = ckpt['global_step'] + 1
            scale_factors = ckpt['scale_factors']

            motion_model1.load_state_dict(ckpt['motion_model1_state_dict'])
            motion_model2.load_state_dict(ckpt['motion_model2_state_dict'])

            quat_model1.load_state_dict(ckpt['quat_model1_state_dict'])
            quat_model2.load_state_dict(ckpt['quat_model2_state_dict'])
            
            optimizer_motion.load_state_dict(ckpt['optimizer_state_dict'])
            
            for param_group in optimizer_motion.param_groups:
                param_group['lr'] = gmc_lr
        
    return motion_model1, quat_model1, motion_model2, quat_model2, optimizer_motion, motion_query_func, start, scale_factors


def calc_scale_factors(gaussians, motion_query_func, motion_model, quat_model, args):
    pnm_points = gaussians.get_xyz
    pnm_feature_term_input = gaussians.get_semantic_feature[:, args.gs.feat_dim:]

    std_xyz = torch.std(gaussians.get_xyz, dim=0)
    std_xyz = torch.sum(std_xyz)
    scale_xyz = 1 / std_xyz
    
    std_feature_pca = torch.std(pnm_feature_term_input, dim=0)
    std_feature_pca = torch.sum(std_feature_pca)
    scale_feature_pca = 1 / std_feature_pca
    
    potentials = motion_query_func(pnm_points * scale_xyz * args.gmc.xyz_input_scale, 
                                   pnm_feature_term_input * scale_feature_pca * args.gmc.dino_input_scale, 
                                   motion_model)

    energy_xyz = gaussians.get_xyz * scale_xyz

    quaternion = motion_query_func(pnm_points * scale_xyz * args.gmc.xyz_input_scale, 
                                    pnm_feature_term_input * scale_feature_pca * args.gmc.dino_input_scale, 
                                    quat_model)
    quaternion = normalize_quaternion(quaternion)
    rot_matrix = quaternions_to_rotation_matrices(quaternion)
    energy_xyz = torch.matmul(rot_matrix, energy_xyz.unsqueeze(-1)).squeeze(-1)

    curr_potentials_term = potentials + energy_xyz

    std_rgb = torch.std(gaussians.get_rgb, dim=0)
    std_feature = torch.std(gaussians.get_semantic_feature[:, :args.gs.feat_dim], dim=0)
    std_potentials = torch.std(curr_potentials_term, dim=0)

    std_rgb = torch.sum(std_rgb)
    std_feature = torch.sum(std_feature)
    std_potentials = torch.sum(std_potentials)

    scale_rgb = 1 / std_rgb 
    scale_features = 1 / std_feature
    scale_potentials = 1 / std_potentials

    return scale_rgb.item(), scale_features.item(), scale_potentials.item(), scale_xyz.item(), scale_feature_pca.item()
    

def scale_fgs_lr(fgs, lr_args):
    fgs.set_learning_rate('xyz', lr_args.fgs_xyz_lr)
    fgs.set_learning_rate('f_dc', lr_args.fgs_f_dc_lr)
    fgs.set_learning_rate('f_rest', lr_args.fgs_f_rest_lr)
    fgs.set_learning_rate('opacity', lr_args.fgs_opacity_lr)
    fgs.set_learning_rate('scaling', lr_args.fgs_scaling_lr)
    fgs.set_learning_rate('rotation', lr_args.fgs_rotation_lr)
    fgs.set_learning_rate('semantic_feature', lr_args.fgs_semantic_feature_lr)
