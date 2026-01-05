#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from gsplat import rasterization


def render(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    render_feature=True,
):
    device = pc.get_xyz.device

    img_height = int(viewpoint_camera.image_height)
    img_width = int(viewpoint_camera.image_width)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    fx = img_width / (2.0 * tanfovx)
    fy = img_height / (2.0 * tanfovy)
    cx = img_width * 0.5
    cy = img_height * 0.5

    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )[None]  # [1, 3, 3]

    # gsplat expects viewmats = inverse(camtoworld)
    viewmat = viewpoint_camera.world_view_transform.T
    viewmats = viewmat[None]  # [1, 4, 4]

    means = pc.get_xyz                         # [N, 3]
    quats = pc.get_rotation                   # [N, 4]
    scales = pc.get_scaling * scaling_modifier  # [N, 3]
    opacities = pc.get_opacity                # [N]

    render_rgb, render_alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.squeeze(-1),
        colors=pc.get_features,
        sh_degree=pc.active_sh_degree,
        viewmats=viewmats,
        Ks=K,
        width=img_width,
        height=img_height,
        backgrounds=bg_color.unsqueeze(0),  # [1, 3]
        packed=False
    )

    # [1, H, W, 3] -> [3, H, W]
    rgb = render_rgb[0].permute(2, 0, 1)

    feature_map = None
    if render_feature:
        semantic_features = pc.get_semantic_feature  # [N, F]
        # Background for semantic features
        semantic_bg = torch.ones(1, semantic_features.shape[-1], dtype=torch.float, device=device) * bg_color[0]  # [1, F]

        render_feat, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities.squeeze(-1).detach(),  # matches your distilling=True
            colors=semantic_features,
            viewmats=viewmats,
            Ks=K,
            width=img_width,
            height=img_height,
            backgrounds=semantic_bg,  # [1, H, W, F]
            packed=False
        )

        # [1, H, W, F] -> [F, H, W]
        feature_map = render_feat[0].permute(2, 0, 1)

    viewspace_points = info["means2d"]  # [1, N, 2]
    try:
        viewspace_points.retain_grad()  
    except:
        pass   

    return {
        "render": rgb,
        "feature_map": feature_map,
        "viewspace_points": viewspace_points, # [1, N, 2]
        "visibility_filter": (info["radii"] > 0).all(-1).any(0),
        "radii": info["radii"][0].max(-1).values,
    }