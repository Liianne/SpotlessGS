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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    ori_image: np.array
    ori_image_path: str
    ori_image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    #novel_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    n = 0
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        #ori_images_folder = "/home/stud/hlia/storage/guided_research/Cycle_gan/PyTorch-CycleGAN/output/final_test_results_real_world"
        #ori_images_folder = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/real_world_2/train/ours_30000/ori_images"
        #ori_images_folder = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/mf2_5/train/ours_30000/cycle_gan"
        #ori_images_folder = "/home/stud/hlia/Downloads/anymal_1/cam3/cycle-gan"
        #ori_images_folder = "/home/stud/hlia/Downloads/blackfly/subset_6/cycle_gan_processed_images"
        ori_images_folder = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/real_world_mask_6/train/ours_3000/renders_3000"
        ori_image_path = os.path.join(ori_images_folder, f"{n:05d}.png")
        ori_image_name = str(n)
        ori_image = Image.open(ori_image_path)

        n = n + 1

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                    image=image, image_path=image_path, image_name=image_name, 
                    ori_image=ori_image, ori_image_path=ori_image_path, ori_image_name = ori_image_name,
                    width=width, height=height)
        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readNovelColmapCameras(cam_extrinsics, cam_intrinsics):
    cam_infos = []
    #n = 0
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)

        # #ori_images_folder = "/home/stud/hlia/storage/guided_research/Cycle_gan/PyTorch-CycleGAN/output/final_test_results_real_world"
        # #ori_images_folder = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/real_world_2/train/ours_30000/ori_images"
        # ori_images_folder = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/mf2_5/train/ours_30000/cycle_gan"
        # ori_image_path = os.path.join(ori_images_folder, f"{n}.png")
        # ori_image_name = str(n)
        # ori_image = Image.open(ori_image_path)

        #n = n + 1

        # Use dummy black image as placeholder
        # dummy_np = np.zeros((height, width, 3), dtype=np.uint8)
        # dummy_img = Image.fromarray(dummy_np)

        #image = torch.zeros((3, height, width))

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                    image=None, image_path="", image_name=f"novel_{idx}", 
                    ori_image=None, ori_image_path="", ori_image_name = f"novel_{idx}",
                    width=width, height=height)
        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

# def fetchPly(path):
#     plydata = PlyData.read(path)
#     vertices = plydata['vertex']
#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
#     colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
#     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
#     return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPly(path):
    try:
        plydata = PlyData.read(path)
    except Exception as e:
        print(f"Error reading .ply file: {e}")
        return None

    # Check if 'vertex' element exists in plydata
    if 'vertex' not in plydata:
        print("Error: 'vertex' element not found in the .ply file.")
        return None

    vertices = plydata['vertex']

    # Check if the required vertex attributes exist
    required_attrs = ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz']
    #required_attrs = ['x', 'y', 'z', 'ar', 'ag', 'ab', 'nx', 'ny', 'nz']
    for attr in required_attrs:
        if attr not in vertices:
            print(f"Error: Missing required attribute '{attr}' in the 'vertex' element.")
            return None

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    #colors = np.vstack([vertices['ar'], vertices['ag'], vertices['ab']]).T / 255.0  # Normalize the RGB values
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    # Check if the point cloud is empty
    if len(positions) == 0:
        print("Error: Point cloud has no vertices.")
        return None

    print(f"Successfully loaded point cloud with {len(positions)} points.")
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=11):
    # try:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    #novel_cameras_extrinsic_file = os.path.join(path, "sparse/0", "novel_images.txt")
    novel_cameras_extrinsic_file = "/home/stud/hlia/Downloads/blackfly/subset_6/sparse/0/novel_images.txt"
    novel_cam_extrinsics = read_extrinsics_text(novel_cameras_extrinsic_file)

    # except:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    # cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    # cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    # cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    novel_cam_infos = readNovelColmapCameras(novel_cam_extrinsics, cam_intrinsics)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "sparse/0/filtered_he_sub_fused_colored_cloud_with_normals.ply")
    #ply_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/mf2_5/point_cloud/iteration_30000/point_cloud.ply"
    #ply_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/real_world_mask_1/point_cloud/iteration_3000/point_cloud.ply"
    ply_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/real_world_mask_6/point_cloud/iteration_3000/point_cloud.ply"
    
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        print("trying to fetch ply...")
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if pcd is None:
        print("Point cloud loading failed!")
    else:
        print(f"Colmap Point cloud loaded successfully with {len(pcd.points)} points.")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           novel_cameras=novel_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            #no extension here
            #cam_name = os.path.join(path, frame["file_path"] + extension)
            cam_name = os.path.join(path, frame["file_path"])
            ori_cam_name = os.path.join(path, frame["file_path_ori"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            
            ori_image_path = os.path.join(path, ori_cam_name)
            ori_image_name = Path(cam_name).stem
            ori_image = Image.open(ori_image_path)

            im_data = np.array(image.convert("RGBA"))
            ori_im_data = np.array(ori_image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            ori_norm_data = ori_im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
            arr_ori = ori_norm_data[:,:,:3] * ori_norm_data[:, :, 3:4] + bg * (1 - ori_norm_data[:, :, 3:4])
            ori_image = Image.fromarray(np.array(arr_ori*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, ori_image=ori_image, ori_image_path=ori_image_path, ori_image_name=ori_image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    #train_cam_infos = readCamerasFromTransforms(path, "transforms_train_new.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    #ply_path = os.path.join(path, "points3d.ply")
    #ply_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/sh_6/point_cloud/iteration_7000/point_cloud.ply"
    #ply_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/sh_6/input.ply"
    #ply_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/anymal_2/point_cloud/iteration_30000/point_cloud.ply"
    ply_path = "/home/stud/hlia/storage/master_thesis/dataset/GT/all_0b16abb1-4a59-4ce3-85b5-8ec10440d9dd/camera_poses/fused.ply"

    #print("File exists:", os.path.exists(ply_path))
    #ply_path = "/home/stud/hlia/storage/master_thesis/ori_ori_darkgs/darkgs/output/sh_6/point_cloud/iteration_30000/point_cloud.ply"

    #ply_path = os.path.join(path, "fused.ply")

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}