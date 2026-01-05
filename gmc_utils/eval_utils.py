import torch
import os
import torchvision
import numpy as np

from tqdm import tqdm
from pyemd import emd_samples

from gmc_utils.knn_utils import update_index_with_new_data
from gmc_utils.pose_utils import interpolate_rotations
from gmc_utils.mped import MPED_VALUE
from cleanfid import fid

from gaussian_renderer import render


def furthest_point_sampling(point_cloud: np.ndarray, N_sample: int):
    """
    Perform Furthest Point Sampling (FPS) on a point cloud and return sampled points and their indices.
    
    If N_sample >= N, randomly duplicate points in the point cloud to reach N_sample.
    
    Args:
        point_cloud (np.ndarray): Input point cloud of shape (N, 3).
        N_sample (int): Number of points to sample.
    
    Returns:
        tuple:
            - np.ndarray: The sampled point cloud of shape (N_sample, 3).
            - np.ndarray: Indices of the sampled points in the original point cloud.
    """
    N, dim = point_cloud.shape

    print(f"FPS: N={N}, N_sample={N_sample}")

    if N_sample >= N:
        # Calculate the number of duplicates needed
        num_duplicates = N_sample - N
        
        # Randomly select indices to duplicate
        duplicate_indices = np.random.choice(N, size=num_duplicates, replace=True)
        
        # Combine original indices with duplicated indices
        selected_indices = np.concatenate([np.arange(N), duplicate_indices])
        
        # Extract the sampled points
        sampled_points = point_cloud[selected_indices]
        
        return sampled_points, selected_indices
    
    # Initialize arrays for FPS
    sampled_points = np.zeros((N_sample, dim))
    selected_indices = np.zeros(N_sample, dtype=int)
    
    # Randomly pick the first point
    selected_indices[0] = np.random.randint(0, N)
    sampled_points[0] = point_cloud[selected_indices[0]]
    
    # Initialize a distance array to keep track of the minimum distance to any sampled point
    distances = np.full(N, np.inf)
    
    for i in range(1, N_sample):
        # Compute distances from all points to the latest sampled point
        current_point = sampled_points[i - 1]
        dist_to_current = np.linalg.norm(point_cloud - current_point, axis=1)
        
        # Update the minimum distances
        distances = np.minimum(distances, dist_to_current)
        
        # Select the point with the maximum distance as the next sampled point
        selected_indices[i] = np.argmax(distances)
        sampled_points[i] = point_cloud[selected_indices[i]]
    
    return sampled_points, selected_indices


def get_scene_flow(gaussians_q, rot_matrix_q, potentials_q, query_vecs, 
                   gaussians_d, rot_matrix_d, potentials_d, database, 
                   scale_xyz, index_gpu, args, i, device, fg_ids):
                   
    update_index_with_new_data(index_gpu, database)
    _, nn_idx = index_gpu.search(query_vecs.detach().cpu().numpy(), 1)  # 1 stands for nearest neighbour
    nn_idx = torch.tensor(nn_idx).to(device).squeeze(1)

    if fg_ids is not None:
        xyz_q = gaussians_q.get_xyz[fg_ids]
        xyz_d = gaussians_d.get_xyz[fg_ids]
    else:
        xyz_q = gaussians_q.get_xyz
        xyz_d = gaussians_d.get_xyz

    if args.database.xyz_dist_weight == 0:
        pred_scene_flow = xyz_d[nn_idx] - xyz_q
    else:
        m_inv_d = rot_matrix_d[nn_idx].transpose(1, 2)
        relative_rot_q = m_inv_d @ rot_matrix_q
        rot_applied_q = torch.bmm(relative_rot_q, xyz_q.unsqueeze(-1) * scale_xyz).squeeze(-1)
        relative_trans_q = torch.bmm(m_inv_d, (potentials_q - potentials_d[nn_idx]).unsqueeze(-1)).squeeze(-1)

        pred_scene_flow = (rot_applied_q + relative_trans_q) / scale_xyz - xyz_q
            
    return pred_scene_flow


def save_gmc_ckpt(motion_model1, quat_model1, motion_model2, quat_model2, optimizer_motion, scale_factors, i, save_dir):
    save_dict = {
        'global_step': i,
        'optimizer_state_dict': optimizer_motion.state_dict(),
        'motion_model1_state_dict': motion_model1.state_dict(),
        'motion_model2_state_dict': motion_model2.state_dict(),
        'quat_model1_state_dict': quat_model1.state_dict(),
        'quat_model2_state_dict': quat_model2.state_dict(),
        'scale_factors': scale_factors}

    path = os.path.join(save_dir, '{:06d}.tar'.format(i))
    torch.save(save_dict, path)
    print('Saved checkpoints at', path)


def save_gaussians(gaussians1, gaussians2, i, save_dir):
    # Save Gaussians
    path_1 = os.path.join(save_dir, '{:06d}_gs1.pth'.format(i))
    torch.save((gaussians1.capture(), i), path_1)
    print('Saved GS1 checkpoints at', path_1)

    path_2 = os.path.join(save_dir, '{:06d}_gs2.pth'.format(i))
    torch.save((gaussians2.capture(), i), path_2)
    print('Saved GS2 checkpoints at', path_2)


def eval_helper(gaussians1, potentials1, rot_matrix1, scene1, pipe1,
                gaussians2, potentials2, rot_matrix2, scene2, pipe2, 
                scale_xyz, scalar_lst, out_dir_vis, nn_idx, i, args, device):

    bg_color_vis = [1, 1, 1] if args.vis.white_background else [0, 0, 0]
    background_vis = torch.tensor(bg_color_vis, dtype=torch.float32, device=device)

    ### Scene Flowed
    pc1_flowed_dir = f"{out_dir_vis}/si_eval"
    os.makedirs(pc1_flowed_dir, exist_ok=True)

    m2_inv = rot_matrix2[nn_idx].transpose(1, 2)
    relative_rot_1 = m2_inv @ rot_matrix1
    relative_trans_1 = torch.bmm(m2_inv, (potentials1 - potentials2[nn_idx]).unsqueeze(-1)).squeeze(-1)

    points_lst = []
    render_folder_lst = []

    for scalar in scalar_lst:
        pc1_flowed_dir_scale = f"{pc1_flowed_dir}/scale{scalar:.2f}"
        os.makedirs(pc1_flowed_dir_scale, exist_ok=True)

        interpolated_rotations = interpolate_rotations(relative_rot_1, scalar)
        interpolated_translations = relative_trans_1 * scalar

        flowed_pc1 = torch.bmm(interpolated_rotations, gaussians1.get_xyz.unsqueeze(-1) * scale_xyz).squeeze(-1) + interpolated_translations
    
        flowed_pc1 = flowed_pc1 / scale_xyz

        original_xyz = gaussians1.get_xyz
        gaussians1.set_xyz(flowed_pc1)

        ### Render from Flowed Gaussians
        for idx, view in enumerate(tqdm(scene1.getTestCameras(), desc="Rendering for evaluation")):
            render_pkg = render(view, gaussians1, background_vis, render_feature=False)
            file_name = os.path.join(pc1_flowed_dir_scale, '{0:05d}'.format(idx) + ".png")

            torchvision.utils.save_image(render_pkg["render"], file_name)

        render_folder_lst.append(pc1_flowed_dir_scale)
            
        # change back to the original positions
        gaussians1.set_xyz(original_xyz)

        points_lst.append(flowed_pc1.detach().cpu().numpy())

    return render_folder_lst, points_lst


def si_fid(fid_paths, points, start, end, rand_proj=False):
    result = {}
    result["weighted_fids"] = []
    result["start_fids"] = []
    result["end_fids"] = []
    result["ratios"] = []
    prev_ratio = 0.0
    prev_f = fid.compute_fid(start, fid_paths[0], rand_proj=rand_proj)
    result["total"] = 0.0

    total_diff = 0.0
    for i in range(1, len(points)):
        total_diff += np.sum(np.linalg.norm(points[i-1].squeeze() - points[i].squeeze(), axis=1))

    max_diff = np.sum(np.linalg.norm(points[0] - points[-1], axis=1))

    print("Computing FID for start")
    print("len(fid_paths): ", len(fid_paths))
    print("len(points): ", len(points))
    for i in tqdm(range(1, len(fid_paths))):
        fid_path = fid_paths[i]
        fid_start = fid.compute_fid(start, fid_path, rand_proj=rand_proj)
        fid_end = fid.compute_fid(end, fid_path, rand_proj=rand_proj)
        result["start_fids"].append(fid_start)
        result["end_fids"].append(fid_end)
        cur_max_pt_diffs = np.sum(np.linalg.norm(points[0].squeeze() - points[i].squeeze(), axis=1))
        cur_ratio = min(cur_max_pt_diffs / max_diff, 1.0)
        
        result["ratios"].append(cur_ratio)
        cur_f = fid_start * (1.0 - cur_ratio) + fid_end * cur_ratio
        result["total"] += (cur_f + prev_f) / 2.0 * np.abs(cur_ratio - prev_ratio)
        result["weighted_fids"].append(
            (cur_f + prev_f) / 2.0 * np.abs(cur_ratio - prev_ratio)
        )
        prev_ratio = cur_ratio
        prev_f = cur_f
    return result


def si_emd(points, start, end):
    start = torch.from_numpy(start)
    end = torch.from_numpy(end)

    points = [torch.from_numpy(points[i]) for i in range(len(points))]
        
    cur_pts = points[0]
    prev_f = emd_samples(cur_pts, start)
    max_diff = np.sum(np.linalg.norm(np.array(points[0].squeeze(0)) - np.array(points[-1].squeeze(0)), axis=1))

    result = {}
    result["weighted_emd"] = []
    result["start_emd"] = []
    result["end_emd"] = []
    result["ratios"] = []
    prev_ratio = 0.0

    result["total"] = 0.0
    
    total_diff = 0.0
    for i in range(1, len(points)):
        total_diff += np.sum(np.linalg.norm(points[i-1].squeeze(0) - points[i].squeeze(0), axis=1))
    
    for i in tqdm(range(1, len(points))):
        cur_pts = points[i]

        emd_start = emd_samples(cur_pts, start)
        emd_end = emd_samples(cur_pts, end)
        result["start_emd"].append(emd_start)
        result["end_emd"].append(emd_end)

        cur_max_pt_diffs = np.sum(np.linalg.norm(points[0].squeeze() - points[i].squeeze(), axis=1))
        cur_ratio = min(cur_max_pt_diffs / max_diff, 1.0)

        result["ratios"].append(cur_ratio)
        cur_f = emd_start * (1.0 - cur_ratio) + emd_end * cur_ratio
        result["total"] += (cur_f + prev_f) / 2.0 * np.abs(cur_ratio - prev_ratio)
        result["weighted_emd"].append(
            (cur_f + prev_f) / 2.0 * np.abs(cur_ratio - prev_ratio)
        )
        prev_ratio = cur_ratio
        prev_f = cur_f
        print(
            f"Iteration: {i}, Start EMD {emd_start}, End EMD {emd_end}, Total Weighted EMD: {result['total']}, current weighted EMD {result['weighted_emd'][-1]}, current ratio {cur_ratio}"
        )

    return result


def si_mped(points, start, end, neighbors):
    # convert to torch tensors
    start = torch.from_numpy(start).unsqueeze(0)
    end = torch.from_numpy(end).unsqueeze(0)
    points = [torch.from_numpy(points[i]).unsqueeze(0) for i in range(len(points))]
    
    cur_pts = points[0]
    prev_f = MPED_VALUE(start, cur_pts, points[0], neighbors)
    max_diff = np.sum(np.linalg.norm(np.array(points[0].squeeze(0)) - np.array(points[-1].squeeze(0)), axis=1))

    result = {}
    result["weighted_mped"] = []
    result["start_mped"] = []
    result["end_mped"] = []
    result["ratios"] = []
    prev_ratio = 0.0

    result["total"] = 0.0

    total_diff = 0.0
    for i in range(1, len(points)):
        total_diff += np.sum(np.linalg.norm(points[i-1].squeeze(0) - points[i].squeeze(0), axis=1))
    
    for i in tqdm(range(1, len(points))):
        cur_pts = points[i]

        mped_start = MPED_VALUE(start, cur_pts, points[0], neighbors)
        mped_end = MPED_VALUE(end, cur_pts, points[-1], neighbors)
        result["start_mped"].append(mped_start)
        result["end_mped"].append(mped_end)

        cur_max_pt_diffs = np.sum(np.linalg.norm(points[0].squeeze(0) - points[i].squeeze(0), axis=1))
        cur_ratio = min(cur_max_pt_diffs / max_diff, 1.0)

        result["ratios"].append(cur_ratio)
        cur_f = mped_start * (1.0 - cur_ratio) + mped_end * cur_ratio
        result["total"] += (cur_f + prev_f) / 2.0 * np.abs(cur_ratio - prev_ratio)
        result["weighted_mped"].append(
            (cur_f + prev_f) / 2.0 * np.abs(cur_ratio - prev_ratio)
        )
        prev_ratio = cur_ratio
        prev_f = cur_f
        print(
            f"Iteration: {i}, Start MPED {mped_start}, End MPED {mped_end}, Total Weighted MPED: {result['total']}, current weighted MPED {result['weighted_mped'][-1]}, current ratio {cur_ratio}"
        )

    return result