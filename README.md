# DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes

## 📖 Overview

<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## 🛠️ Installation

We test our code on Ubuntu 20.04 using Python 3.9 and PyTorch 2.0. We recommend using conda to install all the independencies. 


1. Create the conda environment and install requirements. 
```
# Clone the repo.
git clone ***
cd DeSiReGS

# Create the conda environment.
conda create -n DeSiReGS python==3.9
conda activate DeSiReGS

# Install torch. 
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 # replace with your own CUDA version

# Install requirements.
pip install -r requirements.txt
```

2. Install the submodules. The repository contains the same submodules as [PVG](https://github.com/fudan-zvg/PVG).
```
# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# a modified gaussian splatting (for feature rendering)
git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install nvdiffrast (for Envlight)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

## 💾 Data Preparation

### Waymo Dataset

We provide following subsets from Waymo Open Dataset.

| Source | Number of Sequences |      Scene Type      | Description |
|:----------:|:---------------------:|:-----------------:|----------------------------|
| [PVG](https://github.com/fudan-zvg/PVG) | 4 | Dynamic | • Refer to [this page](https://github.com/fudan-zvg/PVG?tab=readme-ov-file#data-preparation). |
| [OmniRe](https://ziyc.github.io/omnire/) | 8 | Dynamic | • Described as highly complex dynamic<br>• Refer to [this page](https://github.com/ziyc/drivestudio/blob/main/docs/Waymo.md). |
| NOTR from [EmerNeRF](https://github.com/NVlabs/EmerNeRF) | 64 | 32 dynamic<br>32 static | • Contains 32 static, 32 dynamic and 56 diverse scenes. <br> • We test our code on the 32 static and 32 dynamic scenes. <br> • See [this page](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md) for detailed instructions. |


### KITTI Dataset

| Source | Number of Sequences |      Scene Type      | Description |
|:----------:|:---------------------:|:-----------------:|----------------------------|
| [PVG](https://github.com/fudan-zvg/PVG) | 3 | Dynamic | • Refer to [this page](https://github.com/fudan-zvg/PVG?tab=readme-ov-file#kitti-dataset). |

## Training and Evaluation

### Training
Run the following command to train an uncertainty model in stage 1. You can define your scene type in the ```.yaml``` config file.
```
# Stage 1
python train.py \
--config configs/emer_reconstruction_stage1.yaml \
source_path=${SOURCE_PATH}/022 \
model_path=eval_output/waymo_reconstruction/022_stage1
```
After running the command, the uncertainty model will be saved in ```eval_output/waymo_reconstruction/022_stage1/uncertainty_model.pth``` by default. Then train the second stage by running

```
# Stage 2
python train.py \
--config configs/emer_reconstruction_stage2.yaml \
source_path=${SOURCE_PATH}/022 \
model_path=eval_output/waymo_reconstruction/022_stage2 \
uncertainty_model_path=eval_output/waymo_reconstruction/022_stage1/uncertainty_model30000.pth
```
### Evaluating

Here is an example to evaluate the results.

```
# Evaluating
python evaluate.py --config_path eval_output/waymo_reconstruction/022_stage2/config.yaml
```

## Static-Dynamic Decomposition

<div align="center">
  <img src="assets/separation.png"/>
</div><br/>


We provide code ```separate.py``` for static-dynamic decomposition. Run

```
python separate.py --config_path ${MODEL_PATH}/config.yaml
```
For instance, 
```
# example
python separate.py --config_path eval_output/waymo_reconstruction/022_stage2/config.yaml
```
The decomposition results will be saved in ```${MODEL_PATH}/separation```.

## Visualization

### 3D Gaussians Visualization

Following [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), we use [SIBR](https://sibr.gitlabpages.inria.fr/) framework, which is developed by GRAPHDECO group, as an interactive viewer to visualize the gaussian ellipsoids. Refer to [this page](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers) for more installation details.

We provide code ```visualize_gs.py``` for gaussian ellipsoids visualization. For example, run

```
# Save gaussian point cloud.
python visualize_gs.py --config_path eval_output/waymo_reconstruction/022_stage2/config.yaml
```

The ```.ply``` file which contains visible gaussians will be saved in  ```${MODEL_PATH}/point_cloud/point_cloud.ply```. You can use SIBR Viewer to visualize the gaussians directly in your model path folder. For example,

```
# Enter your SIBR folder.
cd ${SIBR_FOLDER}/SIBR_viewers/install/bin

# Visualize the gaussians.
./SIBR_gaussianViewer_app -m ${MODEL_PATH}/

# Example
./SIBR_gaussianViewer_app -m ${PROJECT_FOLDER}/eval_output/waymo_reconstruction/022_stage2/
```

### Visualization Results

<div align="center">
  <img src="assets/qualitative_comp.png"/>
</div><br/>