# SpotlessGS: Relightable 3D Gaussian Splatting under Dynamic Illumination for Robotic Perception

This implementation was based on the repositories:
- [DarkGS](https://github.com/tyz1030/darkgs)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## Cloning the Repository
```sh
# HTTPS
git clone https://github.com/Liianne/SpotlessGS --recursive
```
or 
```sh
# SSH
git clone git clone git@github.com/Liianne/SpotlessGS --recursive
```
## Optimizer
The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models.

## Setup
Our code has been tested on Ubuntu 20.04 and CUDA 12.2.
Please install the environment: 

```sh
conda env create --file environment.yml
conda activate SpotlessGS
```

## Data
To get the input camera poses and point cloud, there are two options:
- [COLMAP](https://colmap.github.io/)
Please first create an input_data folder and an images subfolder, save all the input (undistorted) images in the images folder, then run [COLMAP](https://colmap.github.io/) to get camera poses and a sparse point cloud. 
- [OKVIS2](https://github.com/ethz-mrl/okvis2)
Please note that the raw data from OKVIS2 has to be converted into COLMAP format. The input folder should look like:

```
├── input_data
|   ├── images
|   |   ├── cam_img_1.png
|   |   ├── ...
|   |   ├── cam_img_N.png
|   ├── sparse
|   |   ├── 0
|   |   |   ├── cameras.txt
|   |   |   ├── images.txt
|   |   |   ├── points3D.txt

```
- [Blender](https://github.com/DLR-RM/BlenderProc)
If you have the camera trajectory from Blender (e.g., synthetic dataset), then the input folder should look like:
```
├── input_data
|   ├── images
|   |   ├── cam_img_1.png
|   |   ├── ...
|   |   ├── cam_img_N.png
|   ├── cam_traj_train.json
|   ├── cam_traj_test.json
|   ├── points3d.ply

```
## Training
To train the model, first navigate to the Spotless folder, then:

```sh
python train.py -s <path to input_data>
```
If you want to train the model while having a test set for evaluation, please run:

```sh
python train.py -s <path to input_data> --eval
```

Please replace `<path to input_data>` with the path to the folder containing the COLMAP or OKVIS2 or Blender data. The trained model will be saved in the output folder.

## Evaluation
You can render training/test sets and produce error metrics as follows:
```sh
python render.py -m <path to trained model>
python metrics.py -m <path to trained model>
```
Please replace `<path to trained model>` with the path to the trained model directory. The rendered images will be saved in the same directory. 

## Note
- Input images should be undistorted before running COLMAP and training the model.
- Raw data from OKVIS2 must be converted into COLMAP format before training.
- When using data estimated from COLMAP, different scale factors may need to be tested due to the scale ambiguity in COLMAP.
