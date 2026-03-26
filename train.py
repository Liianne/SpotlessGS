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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.shading import ShadingModel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from scipy.spatial import KDTree
import numpy as np

from scene.light import LightMLP1D


# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

TENSORBOARD_FOUND = True

SCALE_FACTOR = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate_missing_normals(pred_normals, gt_normals):
    """
    Interpolates missing normals by padding 
    """
    if pred_normals.shape[0] < gt_normals.shape[0]:
        diff = gt_normals.shape[0] - pred_normals.shape[0]
        pad = F.pad(pred_normals, (0, 0, 0, diff), mode='replicate')  # Replicate padding
        return pad
    elif pred_normals.shape[0] > gt_normals.shape[0]:
        pred_normals = pred_normals[:gt_normals.shape[0]]  # Trim excess values
    return pred_normals

def match_normals(pred_normals, gt_normals, pred_positions, gt_positions):
    """
    Matches predicted normals to ground truth using nearest neighbor search.
    
    Args:
        pred_normals (torch.Tensor): Predicted normals [N, 3].
        gt_normals (torch.Tensor): Ground truth normals [M, 3].
        pred_positions (torch.Tensor): 3D positions of predicted normals [N, 3].
        gt_positions (torch.Tensor): 3D positions of GT normals [M, 3].

    Returns:
        torch.Tensor: Reordered predicted normals, aligned with gt_normals.
    """
    if pred_normals.shape[0] == gt_normals.shape[0]: 
        return gt_normals # If same size, no need for alignment

    print("The shapes don't match.")

    pred_positions_np = pred_positions.cpu().detach().numpy()
    gt_positions_np = gt_positions.cpu().detach().numpy()

    tree = KDTree(gt_positions_np)  # Build KD-tree on GT positions
    _, indices = tree.query(pred_positions_np)  # Find nearest GT index for each pred

    aligned_gt_normals = gt_normals[torch.tensor(indices, device=gt_normals.device)]
    return aligned_gt_normals


def cosine_loss(pred_normals, gt_normals, pred_positions, gt_positions):
    """
    Computes cosine loss, aligning predicted normals with GT using nearest neighbors.
    """
    gt_normals_matched = match_normals(pred_normals, gt_normals, pred_positions, gt_positions)

    # Normalize both normal vectors
    pred_normals = F.normalize(pred_normals, dim=-1)
    gt_normals_matched = F.normalize(gt_normals_matched, dim=-1)

    print("the shape of predicted normal: ",pred_normals.shape)
    print("the shape of gt normal: ", gt_normals_matched.shape)

    # Compute cosine similarity loss
    loss = 1.0 - F.cosine_similarity(pred_normals, gt_normals_matched, dim=-1)

    return loss.mean(), gt_normals_matched



class EarlyStopping:
    def __init__(self, patience=500, min_iterations=1000, mode='min'):
        self.patience = patience
        self.min_iterations = min_iterations
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_iter = 0

    def check(self, loss_value, current_iter):
        if current_iter < self.min_iterations:
            return False  # Never stop before minimum iterations

        improved = (loss_value < self.best_score) if self.mode == 'min' else (loss_value > self.best_score)

        if improved:
            self.best_score = loss_value
            self.counter = 0
            self.best_iter = current_iter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint: str, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    early_stopper = EarlyStopping(patience=500, min_iterations=1000, mode='min')

    #gaussians.set_normal()

    # #saved_normals_path = os.path.join(dataset.model_path, f"saved_normals_{iteration}.pth")
    # normals_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/outdoor_1_new_new/normals.pth"
    # if os.path.exists(normals_path):
    #     # saved_normals = torch.load(normals_path)
    #     # print(f"Loaded normals from {normals_path}")
    #     saved_data = torch.load(normals_path)
    #     saved_normals = saved_data["normals"]
    #     saved_xyz = saved_data["xyz"]

    #     # Find matching Gaussians by position
    #     current_xyz = gaussians.get_xyz
    #     matching_indices = torch.all(torch.isclose(current_xyz[:, None], saved_xyz[None, :], atol=1e-5), dim=-1).any(dim=1)

    #     # Apply only to matching indices
    #     if matching_indices.sum() == saved_normals.shape[0]:  # Ensure matching count
    #         gaussians.set_normal(saved_normals[matching_indices])
    #         print("Loaded and assigned normals correctly!")
    #     else:
    #         print("Mismatch in Gaussian positions. Normals not assigned.")


    # print("Saved Normal shape: ", saved_normals.shape)
    # print("Ori Normal shape: ", gaussians.get_normal.shape)
    # covariance, other_data = gaussians.get_covariance()
    # print("Ori Convariance shape: ", covariance.shape)


    # # Assign them to the Gaussian model
    # if gaussians.get_normal.shape == saved_normals.shape:
    #     gaussians.set_normal(saved_normals)
    #     print("Normals assigned to Gaussians!")
    # else:
    #     print("Warning: Normal shape mismatch! Skipping normal assignment.")

    # print("Normal: ", gaussians.get_normal)
    
    

    shader = ShadingModel(light = "1DMLP")
    shader = shader.cuda()

    # #newly added
    # shader = shader.to("cuda:0")

    # model_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/sh_6"

    # finetuned_shadermodel_path = model_path+"/shader30000.pth"
    # shader_dict = torch.load(finetuned_shadermodel_path)
    # #res = shader.load_state_dict(shader_dict['model_state_dict'])
    
    # # Apply the weights tbut what should I o the model
    # shader.load_state_dict(shader_dict['model_state_dict'], strict=False)
    
    dict = torch.load('model_parameters.pth')
   
    # res = shader.load_state_dict(dict['model_state_dict'], strict=False)

    shader = shader.to("cuda:0")
    # r_vec = dict['so3'].squeeze()
    # t_vec = dict['model_state_dict']['light._t_vec'].squeeze()
    # r_vec = shader_dict['so3'].squeeze()
    # t_vec = shader_dict['model_state_dict']['light._t_vec'].squeeze()

    r_vec = torch.zeros_like(dict['so3'].squeeze())
    # t_vec = torch.tensor([0.0, 0.0, 0.023], dtype=torch.float32).squeeze()
    # t_vec = torch.tensor([-0.05, 0.0, 0.01], dtype=torch.float32).squeeze()
    t_vec = torch.tensor([0.145, 0.03, 0.08], dtype=torch.float32).squeeze()
    #t_vec = torch.tensor([0.00092, 0.00092, 0.00092], dtype=torch.float32).squeeze()
    #t_vec = torch.tensor([0.0238, 0.0238, 0.0238], dtype=torch.float32).squeeze()
    #t_vec = torch.tensor([-0.0238, -0.0238, 0.0238], dtype=torch.float32).squeeze()
    
    # print(r_vec)
    # print(t_vec)

    # Check /home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/37d24fba-4if BRDF parameters are included in optimization
    brdf_in_optimization = any(
        id(p) in {id(param) for group in shader.optimizer.param_groups for param in group['params']}
        for p in shader.brdf.parameters()
    ) if hasattr(shader, 'brdf') else False

    if brdf_in_optimization:
        print("BRDF parameters are included in the optimizer.")
    else:
        print("WARNING: BRDF parameters are NOT included in the optimizer!")

    # An initial (human) guess of scaling factor (by looking at the colmap vizualization).
    #shader.set_scaling_factor(0.1)
    shader.set_scaling_factor(SCALE_FACTOR)
    #shader.set_scaling_factor(0.238)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        # gaussians._albedo.requires_grad_(False)
        
        # for pg in gaussians.optimizer.param_groups:
        #     if pg.get("name") == "albedo":
        #         pg["lr"] = 0.0
        #         for p in pg["params"]:
        #             state = gaussians.optimizer.state[p]
        #             if "exp_avg" in state:
        #                 state["exp_avg"].zero_()
        #             if "exp_avg_sq" in state:
        #                 state["exp_avg_sq"].zero_()
        
        # freeze_params = ["albedo", "xyz", "scaling", "rotation", "opacity", "normal", "roughness", "metalness"]

   
        # for name in freeze_params:
        #     getattr(gaussians, f"_{name}").requires_grad_(False)

        # for pg in gaussians.optimizer.param_groups:
        #     if pg.get("name") in freeze_params:
        #         pg["lr"] = 0.0
        #         for p in pg["params"]:
        #             state = gaussians.optimizer.state[p]
        #             if "exp_avg" in state:
        #                 state["exp_avg"].zero_()
        #             if "exp_avg_sq" in state:
        #                 state["exp_avg_sq"].zero_()


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    #print("first iteration: ", first_iter)
    first_iter = 0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    warmup_factors  = torch.linspace(0, 1, opt.warmup_until_itr-opt.warmup_start_itr, device="cuda:0", requires_grad=False)
    shader.warmup_factor = 0.0

    shader.light.set_r_vec(tuple([r_vec[0], r_vec[1], r_vec[2]]))
    # shader.light.set_t_vec(tuple([t_vec[0], t_vec[1], t_vec[2]]))

    # torch.save(gt_normals, "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/normals/gt_xyz.pth")
    #gt_xyz = torch.load("/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/normals/gt_xyz.pth").to(device)

    for iteration in range(first_iter, opt.iterations + 1):     

        # # WARM-UP STAGE ########################################
        # if iteration >= opt.warmup_start_itr and iteration < opt.warmup_until_itr:
        #     warmup_factor = warmup_factors[iteration-opt.warmup_start_itr]
        #     # shader.warmup_factor = warmup_factor
        #     shader.light.set_r_vec(tuple([r_vec[0]*warmup_factor, r_vec[1]*warmup_factor, r_vec[2]*warmup_factor]))
        #     shader.light.set_t_vec(tuple([t_vec[0]*warmup_factor, t_vec[1]*warmup_factor, t_vec[2]*warmup_factor]))
        #     # shader.light.set_r_vec(tuple([r_vec[0], r_vec[1], r_vec[2]]))
        #     # shader.light.set_t_vec(tuple([t_vec[0], t_vec[1], t_vec[2]]))       
        # ########################################################

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, shader, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        #print the normal vector 
        #print("Normal: ", gaussians.get_normal.requires_grad)
        #normals = gaussians.get_normal
        # normals = F.normalize(normals, dim=-1, eps=1e-6)
        print("r is: ", shader.light.get_r_vec())
        print("t is: ", shader.light.get_t_vec())


        # gaussians.set_normals(normals)
        pred_normals = gaussians.get_normal
        #print("Normal: ", gaussians.get_normal)
        #print("Roughness: ", gaussians.get_roughness)


        gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            print("Gaussian Size: ", scene.gaussians.get_size)
            print("Scaling factor: ", shader.scaling_factor.item())
            print("Ambient Light: ", shader.ambient_light.item())
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        #set the relit shader
        shader_relit = ShadingModel(brdf = "MLP", light = "relit_1DMLP").cuda()
        #shader_relit = ShadingModel(brdf = "Disney", light = "relit_1DMLP").cuda()
        #shader_relit = ShadingModel(brdf = "LambertianMLP", light = "relit_1DMLP").cuda()
        #shader_relit = Sh/home/stud/hlia/storage/master_thesis/full_full_darkgs/darkgs/output/4dfd7536-1adingModel(brdf = "Lambertian", light = "relit_1DMLP").cuda()
        shader_relit.load_state_dict(shader.state_dict())
        shader_relit.set_scaling_factor(shader.scaling_factor)
        
        # Freeze shader_relit
        for param in shader_relit.parameters():
            param.requires_grad = False
        
        render_pkg_relit = render(viewpoint_cam, gaussians, pipe, bg, shader=shader_relit)
        image_relit, viewspace_point_tensor, visibility_filter, radii = render_pkg_relit["render"], render_pkg_relit["viewspace_points"], render_pkg_relit["visibility_filter"], render_pkg_relit["radii"]

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, shader=shader)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # #newly added: relit images
        # render_pkg_relit = render(viewpoint_cam, gaussians, pipe, bg, shader=shader_relit)
        # relit_image, viewspace_point_tensor_relit, visibility_filter_relit, radii_relit = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        #ori_image = viewpoint_cam.ori_original_image.cuda()


        Ll1 = l1_loss(image, gt_image)
        #Ll1 = l1_loss(image.detach().clamp(0,1), gt_image.detach().clamp(0,1))
        #Ll1_relit = l1_loss(image_relit.detach().clamp(0,1), ori_image.detach().clamp(0,1))
        #Ll1_relit = l1_loss(image_relit, ori_image)
    
        # pred_positions = gaussians.get_xyz
        # gt_positions = gt_xyz
        
        #loss_normals = 0.0

        # if pred_normals.shape[0] == gt_normals.shape[0]: 
        #     loss_normals, gt_normals_matched = cosine_loss(pred_normals, gt_normals, pred_positions, gt_positions)
        #     gt_normals = gt_normals_matched
        # else:
        #     gt_normals = ori_gt_normals
        #     loss_normals, gt_normals_matched = cosine_loss(pred_normals, gt_normals, pred_positions, gt_positions)
        #     gt_normals = gt_normals_matched
        
        #loss = Ll1  
        #loss = 0.2 * Ll1 + 0.8 * Ll1_relit
        #loss = 0.1 * Ll1 + (1.0 - opt.lambda_dssim) * Ll1_relit + opt.lambda_dssim * (1 - ssim(image_relit, ori_image))

        # if image.shape[1] == 1:
        #     image = image.repeat(1, 3, 1, 1)
        # if gt_image.shape[1] == 1:
        #     gt_image = gt_image.repeat(1, 3, 1, 1)

        #target_translation = torch.tensor([-0.068, 0.027, 0.120])
        target_translation = torch.tensor([0.1, 0.1, 0.1])  
        target_rotation = torch.tensor([0.0, 0.0, 0.0])

        # Convert and move to same device
        t_vec = torch.from_numpy(shader.light.get_t_vec()).to(target_translation.device)
        r_vec = torch.from_numpy(shader.light.get_r_vec()).to(target_rotation.device)

        # L2 norm regularization
        light_pose_reg = torch.norm(t_vec - target_translation) #+ torch.norm(r_vec - target_rotation)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #+ 0.05 * light_pose_reg
        #+ 0.005 * Ll1_relit#+ 0.1 * loss_normals 

        #loss = Ll1

        # loss = (0.2 * (1.0 - opt.lambda_dssim) * Ll1 
        #         + 0.2 * opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
        #         + 0.8 * (1.0 - opt.lambda_dssim) * Ll1_relit
        #         + 0.8 * opt.lambda_dssim * (1.0 - ssim(image_relit, ori_image))
        #           )

        #loss.backward(retain_graph=True)
        # Use relit loss or main loss as the metric
        #val_loss = loss.item() 

        # if early_stopper.check(val_loss, iteration):
        #     print(f"\n[EARLY STOPPING] Training stopped at iteration {iteration}. Best iteration was {early_stopper.best_iter}.")
        #     break

        loss.backward()
        
        # # Gradient Clipping (to avoid exploding gradients)
        # max_grad_norm = 1.0 
        # torch.nn.utils.clip_grad_norm_(shader.parameters(), max_grad_norm)

        # for param in gaussians.__dict__.values():
        #     if isinstance(param, torch.Tensor) and param.requires_grad and param.grad is not None:
        #         param.grad.data.clamp_(-max_grad_norm, max_grad_norm)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, Ll1_relit, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, shader), (pipe, background, shader_relit))
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, shader), (pipe, background, shader_relit))
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, shader), (pipe, background, shader_relit), shader)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Some magic hyperparameters here
                    size_threshold = None if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent*shader.scaling_factor, size_threshold)
                    
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            #avoid gradient explosure
            # max_grad_norm = 1.0
            # torch.nn.utils.clip_grad_norm_(shader.parameters(), max_grad_norm)
            # for name, param in gaussians.__dict__.items():
            #     if isinstance(param, torch.Tensor) and param.requires_grad and param.grad is not None:
            #         param.grad.data.clamp_(-max_grad_norm, max_grad_norm)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            #if iteration < opt.shader_optimize_until:
                shader.optimizer.step()
                shader.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations or (iteration % 10000 == 0)):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                save_dict = {
                    'model_state_dict': shader.state_dict(),
                    #'so3': shader.light._r_l2c_SO3.log()
                    'so3': shader.light.get_r_vec(),
                    'light_t': shader.light.get_t_vec(),
                    }
                res = torch.save(save_dict, scene.model_path + "/shader" + str(iteration) + ".pth")
                print("Parameters saved!")

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

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

#def training_report(tb_writer, iteration, Ll1, Ll1_relit, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, renderArgs_relit):
# def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, renderArgs_relit):
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, renderArgs_relit, shader):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        #tb_writer.add_scalar('train_loss_patches/l1_relit_loss', Ll1_relit.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        t_vec_np = shader.light.get_t_vec() 
        tb_writer.add_scalar('light_pose/translation_x', t_vec_np[0], iteration)
        tb_writer.add_scalar('light_pose/translation_y', t_vec_np[1], iteration)
        tb_writer.add_scalar('light_pose/translation_z', t_vec_np[2], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                l1_relit_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    relit_image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs_relit)["render"], 0.0, 1.0)
                    #ori_gt_image = torch.clamp(viewpoint.ori_original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/relit".format(viewpoint.image_name), relit_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        #tb_writer.add_images(config['name'] + "_view_{}/ori_ground_truth".format(viewpoint.image_name), ori_gt_image[None], global_step=iteration)
                        # if iteration == testing_iterations[0]:
                        #     tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        #     tb_writer.add_images(config['name'] + "_view_{}/ori_ground_truth".format(viewpoint.image_name), ori_gt_image[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    #l1_relit_test += l1_loss(relit_image.clamp(0,1), ori_gt_image.clamp(0,1))
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                #l1_relit_test /= len(config['cameras'])
                       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    #tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_relit_loss', l1_relit_test, iteration)
                    
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    #parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[80_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    args.test_iterations = list(range(0, 30001, 1000))
    #args.test_iterations = list(range(0, 80001, 1000))
    #args.start_checkpoint = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/71ee52dd-d/chkpnt7000.pth"
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

    def load_model_param(self)->None:
        print("Loading model parameters...")
        dict = torch.load('model_parameters.pth')
        res = self.shading_model.load_state_dict(dict['model_state_dict'])
        print("loaded model parameters: \n", self.shading_model.state_dict())
        print("load res: \n", res)
        r_vec = dict['so3'].squeeze()
        print(r_vec)
        self.shading_model.light.set_r_vec(tuple([r_vec[0], r_vec[1], r_vec[2]]))
        if hasattr(self.shading_model.light, 'sigma'):
            if self.shading_model.light.sigma.ndim == 0:
                self.update_shading_model_param(self.sh05ading_model.albedo, self.shading_model.light.gamma, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma, 0])
            else:
                self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [self.shading_model.light.sigma[0], self.shading_model.light.sigma[1]])
        else:  
            self.update_shading_model_param(self.shading_model.albedo, self.shading_model.light.gamma, self.shading_model.ambient_light, self.shading_model.light._t_vec, self.shading_model.light._r_l2c_SO3.log(), [0, 0])

