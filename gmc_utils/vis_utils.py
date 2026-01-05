import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import os
import torchvision
from tqdm import tqdm

from gmc_utils.pose_utils import interpolate_rotations
from gaussian_renderer import render

from utils.image_utils import psnr
from utils.loss_utils import ssim
import io
from PIL import Image


def vis_interpolation(gaussians1, potentials1, rot_matrix1, scene1, pipe1,
                      gaussians2, potentials2, rot_matrix2, scene2, pipe2, 
                      scale_xyz, scalar_lst, out_dir_vis, nn_idx, i, args, device, fg_ids):
    if fg_ids is None:
        fg_ids1 = np.arange(gaussians1.get_xyz.shape[0])
        fg_ids2 = np.arange(gaussians2.get_xyz.shape[0])
    else:
        fg_ids1 = fg_ids
        fg_ids2 = fg_ids

    bg_color_vis = [1,1,1] if args.vis.white_background else [0, 0, 0]
    background_vis = torch.tensor(bg_color_vis, dtype=torch.float32, device=device)

    ### Scene Flowed
    pc1_flowed_dir = f"{out_dir_vis}/iter{i}_interpolation_render"
    os.makedirs(pc1_flowed_dir, exist_ok=True)

    if args.database.xyz_dist_weight == 0:
        pred_scene_flow = gaussians2.get_xyz[fg_ids2][nn_idx] - gaussians1.get_xyz[fg_ids1]
    else:
        m2_inv = rot_matrix2[nn_idx].transpose(1, 2)
        relative_rot_1 = m2_inv @ rot_matrix1
        relative_trans_1 = torch.bmm(m2_inv, (potentials1 - potentials2[nn_idx]).unsqueeze(-1)).squeeze(-1)

    for scalar in scalar_lst:
        interpolated_rotations = interpolate_rotations(relative_rot_1, scalar)
        interpolated_translations = relative_trans_1 * scalar

        flowed_pc1 = torch.bmm(interpolated_rotations, gaussians1.get_xyz[fg_ids1].unsqueeze(-1) * scale_xyz).squeeze(-1) + interpolated_translations

        flowed_pc1 = flowed_pc1 / scale_xyz
            
        curr_t_dir = f"{pc1_flowed_dir}/time{scalar:.2f}"
        os.makedirs(curr_t_dir, exist_ok=True)
        
        original_xyz = gaussians1.get_xyz[fg_ids1]
        gaussians1.set_xyz(flowed_pc1, fg_ids1)

        ### Render from Flowed Gaussians
        for idx, view in enumerate(tqdm(scene1.getTestCameras(), desc="Rendering progress")):
            if (idx + 1) % args.vis.render_skip_every == 0:
                render_pkg = render(view, gaussians1, background_vis, render_feature=False) 
                torchvision.utils.save_image(render_pkg["render"], os.path.join(curr_t_dir, '{0:05d}'.format(idx) + ".png"))

        # change back to the original positions
        gaussians1.set_xyz(original_xyz, fg_ids1)


def vis_interpolation_video(gaussians1, potentials1, rot_matrix1, scene1, pipe1,
                            gaussians2, potentials2, rot_matrix2, scene2, pipe2, 
                            scale_xyz, scalar_lst, out_dir_vis, nn_idx, i, args, device, fg_ids):

    if fg_ids is None:
        fg_ids1 = np.arange(gaussians1.get_xyz.shape[0])
        fg_ids2 = np.arange(gaussians2.get_xyz.shape[0])
    else:
        fg_ids1 = fg_ids
        fg_ids2 = fg_ids

    bg_color_vis = [1, 1, 1] if args.vis.white_background else [0, 0, 0]
    background_vis = torch.tensor(bg_color_vis, dtype=torch.float32, device=device)

    ### Scene Flowed
    pc1_flowed_dir = f"{out_dir_vis}/tmp_interpolation_render"
    os.makedirs(pc1_flowed_dir, exist_ok=True)

    m2_inv = rot_matrix2[nn_idx].transpose(1, 2)
    relative_rot_1 = m2_inv @ rot_matrix1
    relative_trans_1 = torch.bmm(m2_inv, (potentials1 - potentials2[nn_idx]).unsqueeze(-1)).squeeze(-1)

    image_path_lst = []
    for scalar in scalar_lst:
        interpolated_rotations = interpolate_rotations(relative_rot_1, scalar)
        interpolated_translations = relative_trans_1 * scalar

        flowed_pc1 = torch.bmm(interpolated_rotations, gaussians1.get_xyz[fg_ids1].unsqueeze(-1) * scale_xyz).squeeze(-1) + interpolated_translations
        
        flowed_pc1 = flowed_pc1 / scale_xyz

        original_xyz = gaussians1.get_xyz
        gaussians1.set_xyz(flowed_pc1, fg_ids1)

        ### Render from Flowed Gaussians
        idx = args.vis.video_view_id
        view = scene1.getTestCameras()[idx]
        render_pkg = render(view, gaussians1, background_vis, render_feature=False) 

        file_name = f"{pc1_flowed_dir}/scalar{scalar:.2f}.png"
        torchvision.utils.save_image(render_pkg["render"], file_name)
        image_path_lst.append(file_name)

        # change back to the original positions
        gaussians1.set_xyz(original_xyz)

    # Create a list to store images
    images = []

    for idx, image_path in enumerate(image_path_lst):
        img = cv2.imread(image_path)
        
        os.remove(image_path)

        # Convert BGR to RGB (imageio uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Append image to list
        images.append(img_rgb)
        
        # If it's the first or last frame, freeze for some time
        if idx == 0 or idx == len(image_path_lst) - 1 or scalar_lst[idx] == 0 or scalar_lst[idx] == 1:
            for _ in range(5):
                images.append(img_rgb)

    output_mp4_path = os.path.join(out_dir_vis, f'interpolation_iter{i}.mp4')
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(output_mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    video.release()

    # remove the temporary directory
    os.rmdir(pc1_flowed_dir)

