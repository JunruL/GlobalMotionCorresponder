import os
import torch
import json
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


def eval(args):
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
    
    load_gmc_ckpt_path = f'{args.log.out_dir}/{args.log.expname}/{args.eval.load_iteration:06d}.tar'
    motion_model1, quat_model1, motion_model2, quat_model2, optimizer_motion, motion_query_func, start, scale_factors = setup_gmc(load_gmc_ckpt_path, device, args)

    assert args.eval.load_iteration == start - 1, "Loaded GMC checkpoint iteration does not match the specified iteration to visualize."

    checkpoint1 = f'{args.log.out_dir}/{args.log.expname}/{(args.eval.load_iteration):06d}_gs1.pth'
    checkpoint2 = f'{args.log.out_dir}/{args.log.expname}/{(args.eval.load_iteration):06d}_gs2.pth'
    gaussians1, scene1, dataset1, opt1, pipe1 = setup_gaussians(args.gs.pretrained_frame1_dir, checkpoint1, args.gs.load_iteration, args.gs.lr)
    gaussians2, scene2, dataset2, opt2, pipe2 = setup_gaussians(args.gs.pretrained_frame2_dir, checkpoint2, args.gs.load_iteration, args.gs.lr)

    # for 3d evaluation
    pseudo_gt_pc2 = load_gt_points(args.gs.pretrained_frame2_dir, args.gs.checkpoint2_gt, args.gs.checkpoint2)

    max_pt = args.eval.max_pt
    
    if max_pt > 0:
        _, sampled_ids = furthest_point_sampling(pseudo_gt_pc2, max_pt)
        pseudo_gt_pc2 = pseudo_gt_pc2[sampled_ids]

    if scale_factors == []:
        scale_rgb, scale_features, scale_potentials, scale_xyz, scale_feature_pca = calc_scale_factors(gaussians1, motion_query_func, motion_model1, quat_model1, args)

        scale_factors = [scale_rgb, scale_features, scale_potentials, scale_xyz, scale_feature_pca]
    else:
        scale_rgb, scale_features, scale_potentials, scale_xyz, scale_feature_pca = scale_factors

    # KNN tool
    dim_base = 3 + feat_dim + 3  # RGB + DINO + xyz
    dim_noise = dim_base + 1
    index_gpu, index_gpu_noise, index_gpu_for_gt_motion = setup_faiss(dim_base, dim_noise)

    with torch.no_grad():
        rot_matrix1, potentials1, energy_xyz1, query_vecs = construct_database(gaussians1, motion_model1, quat_model1, motion_query_func, scale_factors, args.eval.load_iteration, args)
        rot_matrix2, potentials2, energy_xyz2, database = construct_database(gaussians2, motion_model2, quat_model2, motion_query_func, scale_factors, args.eval.load_iteration, args)
        
        update_index_with_new_data(index_gpu, database)
        _, nn_idx = index_gpu.search(query_vecs.detach().cpu().numpy(), 1)  # 1 stands for nearest neighbour

        nn_idx = torch.tensor(nn_idx).to(device).squeeze(1)

        scalar_lst = [0.05 * t for t in range(0, 21)]

        fid_dirs, points = eval_helper(gaussians1, potentials1, rot_matrix1, scene1, pipe1,
                                        gaussians2, potentials2, rot_matrix2, scene2, pipe2, 
                                        scale_xyz, scalar_lst, out_dir_vis, nn_idx, args.eval.load_iteration, args, device)
        
        start_folder = dataset1.source_path + '/test'
        end_folder = dataset2.source_path + '/test'

        # preprocess points, end, and start with max_pt
        if max_pt > 0:
            _, sample_ids = furthest_point_sampling(points[0], max_pt)
            for j in range(len(points)):
                points[j] = points[j][sample_ids]

        emd = si_emd(points, points[0], pseudo_gt_pc2)['total'] * 1000

        min_num_points = min(pseudo_gt_pc2.shape[0], points[0].shape[0])
        neighbors = int(0.01 * min_num_points)
        mped = si_mped(points, points[0], pseudo_gt_pc2, neighbors)['total'] * 1000

        if len(fid_dirs) > 0:
            fid = si_fid(fid_dirs, points, start_folder, end_folder, rand_proj=False)['total']
        else:
            fid = 0

        # remove folder fid_dirs
        for d in fid_dirs:
            os.system(f"rm -rf {d}") 

        print(f"SI-FID: {fid}")
        print(f"SI-EMD: {emd}")
        print(f"SI-MPED (NN{neighbors}): {mped}")

        print('Test done')


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    args = YamlParser().args

    eval(args)
