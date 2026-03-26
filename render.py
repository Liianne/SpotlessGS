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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.shading import ShadingModel
import numpy as np
import matplotlib.pyplot as plt
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    #render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rendering_after_relighting_new")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "relit_new")
    #render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "reconstructed")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    #added a shader
    #shader = ShadingModel(brdf = "MLP", light = "1DMLP")

    shader = ShadingModel(brdf = "MLP", light = "relit_1DMLP")
    #shader = ShadingModel(brdf = "Lambertian", light = "relit_1DMLP")
    #shader = ShadingModel(brdf = "LambertianMLP", light = "relit_1DMLP")
    #shader = ShadingModel(brdf = "Disney", light = "relit_1DMLP")
    shader = shader.to("cuda:0")

    finetuned_shadermodel_path = model_path+"/shader30000.pth"
    #finetuned_shadermodel_path = model_path+"/shader10000.pth"
    #finetuned_shadermodel_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/ab11320f-f/shader3000.pth"
    shader_dict = torch.load(finetuned_shadermodel_path)
    res = shader.load_state_dict(shader_dict['model_state_dict'])
    
    r_vec = shader_dict['so3'].squeeze()
    shader.light.set_r_vec(tuple([r_vec[0], r_vec[1], r_vec[2]]))
    

    t_vec = shader_dict['model_state_dict']['light._t_vec'].squeeze()
    shader.light.set_t_vec(tuple([t_vec[0], t_vec[1], t_vec[2]]))

    # r_vec = shader_dict['so3']
    # shader.light.set_r_vec(tuple(r_vec))

    # t_vec = shader_dict['light_t']
    # shader.light.set_t_vec(tuple(t_vec))


    print("r vector: ", r_vec)
    print("t vector: ", t_vec)

    #draw the mlp pattern
    light_mlp = shader.light
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a 2D grid of points in front of the light source
    grid_size = 401
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-2.0, 2.0, grid_size),  # X-axis range
        torch.linspace(-2.0, 2.0, grid_size),  # Y-axis range
        indexing="xy"
    )

    # Define Z-coordinate (distance from light source)
    grid_z = 0.7 * torch.ones_like(grid_x)  # Constant Z-distance from the light

    # Add extra dimensions to match the trained MLP's expected input
    extra1 = torch.zeros_like(grid_x)
    extra2 = torch.zeros_like(grid_x)

    # # Stack into (x, y, z, extra1, extra2) coordinates
    # grid = torch.stack([grid_x, grid_y, grid_z, extra1, extra2], dim=-1)
    # grid = grid.view(-1, 5).to(device)  # Reshape to (N, 5) and move to GPU

    # Stack into (x, y, z) coordinates
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    grid = grid.view(-1, 3).to(device)  # Reshape to (N, 3)

    # Pass grid points through MLP light model
    intensity = light_mlp(grid[None])  # MLP expects a batch dimension
    intensity = intensity.view(grid_size, grid_size).detach().cpu().numpy()  # Reshape to image format

    # Plot the intensity map
    plt.figure(figsize=(6, 6))
    #plt.imshow(intensity, cmap='hot', extent=[0, 480, 0, 360])
    plt.imshow(intensity, cmap='hot', extent=[0, 720, 0, 540])
    plt.colorbar(label="Light Intensity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("MLP Light Intensity Distribution")
    #plt.show()
    plt.savefig('/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/rid_curves/colmap_100_before.png', dpi=300)
    print("Image saved!")

    render_times = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()

        rendering = render(view, gaussians, pipeline, background, shader)["render"]

        end_time = time.time()
    
        render_times.append(end_time - start_time)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # Log memory usage for this frame
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)  # in MB
        mem_reserved = torch.cuda.memory_reserved() / (1024**2)  # in MB
        print(f"Frame {idx}: Memory allocated = {mem_alloc:.2f} MB, Memory reserved = {mem_reserved:.2f} MB")

    avg_time = sum(render_times) / len(render_times)
    fps = 1.0 / avg_time
    print(f"Average FPS: {fps:.2f}")
    print(f"Average rendering time per frame: {avg_time:.3f} s")

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak GPU memory usage during rendering: {peak_mem:.2f} MB")



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        
        #print("The roughness is: ", gaussians.get_roughness)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # #extract normals from the gaussian model
        # normals = gaussians.get_normal
        # print("Normal Shape:", normals.shape)
        # print("Normal Values:", normals if normals.numel() > 0 else "No normals found!")
        # #print("Normal: ", gaussians.get_normal)

        #  # Save normals to a file
        # save_path = os.path.join("/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/outdoor_1_new_new", f"normals.pth")
        # saved_data = {
        #     "normals": gaussians.get_normal, 
        #     "xyz": gaussians.get_xyz  # Save Gaussian positions too
        # }
        # torch.save(saved_data, save_path)
        # print(f"Normals and positions saved to {save_path}")

        # torch.save(normals, save_path)
        # print(f"Normals saved to {save_path}")


        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)



    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.iteration = 30000
    #args.iteration = 7000
    

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    
