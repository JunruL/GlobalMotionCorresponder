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

import wandb
import pickle
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim 
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch.nn.functional as F


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.isotropic)
    scene = Scene(dataset, gaussians)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    args_dir = f'{scene.model_path}/args'
    os.makedirs(args_dir, exist_ok=True)
    
    with open(f'{args_dir}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
        
    with open(f'{args_dir}/opt.pkl', 'wb') as f:
        pickle.dump(opt, f)
        
    with open(f'{args_dir}/pipe.pkl', 'wb') as f:
        pickle.dump(pipe, f)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    first_iter += 1

    wandb.init(project="GMC",
               name=args.model_path.split("/")[-1],
               tags= ['single_gs'] + [args.model_path.split("/")[-2]])

    for iteration in range(first_iter, opt.iterations + 1):

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, bg)
        
        feature_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        gt_feature_map = viewpoint_cam.semantic_feature.cuda()
        
        ### Interpolate the gt feature map to match the shape of the rendered feature map and mask
        mask = viewpoint_cam.mask.bool().cuda()
        gt_feature_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(feature_map.shape[1], feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
        
        Ll1_feature = l1_loss(feature_map[:, mask], gt_feature_map[:, mask]) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + args.f_weight * Ll1_feature 

        loss.backward()

        with torch.no_grad():
            # Log and save
            training_report(wandb.run, iteration, Ll1, Ll1_feature, loss, l1_loss, testing_iterations, scene, background) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, densify=True)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(wandb_run, iteration, Ll1, Ll1_feature, loss, l1_loss, testing_iterations, scene : Scene, background):
    if wandb_run:
        wandb_run.log({
            'train/l1_loss_color': Ll1.item(),
            'train/l1_loss_feature': Ll1_feature.item(),
            'train/total_loss': loss.item(),
        }, step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'train/l1_loss_color: {Ll1.item()}')
        print(f'train/l1_loss_feature: {Ll1_feature.item()}')
        print(f'train/total_loss: {loss.item()}')

        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, background)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if wandb_run and (idx < 5):
                        wandb_run.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): wandb.Image(image)}, step=iteration)
                        if iteration == testing_iterations[0]:
                            wandb_run.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): wandb.Image(gt_image)}, step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if wandb_run:
                    wandb_run.log({
                        config['name'] + '/avg - l1_loss': l1_test,
                        config['name'] + '/avg - psnr': psnr_test
                    }, step=iteration)

        if wandb_run:
            wandb_run.log({
                "scene/opacity_histogram": wandb.Histogram(scene.gaussians.get_opacity.cpu().numpy()),
                "scene/gaussian_size": wandb.Histogram(scene.gaussians.get_scaling.cpu().numpy()),
                'scene/total_points': scene.gaussians.get_xyz.shape[0]
            }, step=iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training 3DGS model with features.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--f_weight", type=float, default=0.01)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")