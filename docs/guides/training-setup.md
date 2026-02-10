# Training Backend Setup

This guide covers the integrated training panel inside the GS Capture addon.

## Data Requirements

Your training data folder must contain:

- `images/` with `image_0000.*` files
- One of the following:
  - `transforms.json` (Nerfstudio and gsplat)
  - `sparse/0/` COLMAP files (3DGS and GS-Lightning)

## 3D Gaussian Splatting (Original)

1. Clone the repository
   ```bash
   git clone https://github.com/graphdeco-inria/gaussian-splatting.git
   cd gaussian-splatting
   ```
2. Create and activate a conda environment
   ```bash
   conda create -n gaussian_splatting python=3.8
   conda activate gaussian_splatting
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install submodules/diff-gaussian-rasterization
   pip install submodules/simple-knn
   ```
4. In Blender preferences, set:
   - 3D Gaussian Splatting Path: the repo folder
   - 3D Gaussian Splatting Env: `gaussian_splatting` (or your env name)

## Nerfstudio (splatfacto)

1. Create and activate a conda environment
   ```bash
   conda create -n nerfstudio python=3.10
   conda activate nerfstudio
   ```
2. Install PyTorch (adjust CUDA version as needed)
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install Nerfstudio
   ```bash
   pip install nerfstudio
   ```
4. In Blender preferences, set:
   - Nerfstudio Env: `nerfstudio` (or your env name)

## GS-Lightning

1. Clone the repository
   ```bash
   git clone https://github.com/yzslab/gaussian-splatting-lightning.git
   cd gaussian-splatting-lightning
   ```
2. Create and activate a conda environment
   ```bash
   conda create -n gs_lightning python=3.10
   conda activate gs_lightning
   ```
3. Install PyTorch (adjust CUDA version as needed)
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
4. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
5. In Blender preferences, set:
   - GS-Lightning Path: the repo folder
   - GS-Lightning Env: `gs_lightning` (or your env name)
6. In GS Capture settings, enable Export Object Masks and set Mask Format to GS-Lightning.

## gsplat

1. Create and activate a conda environment
   ```bash
   conda create -n gsplat python=3.10
   conda activate gsplat
   ```
2. Install PyTorch (adjust CUDA version as needed)
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install gsplat
   ```bash
   pip install gsplat
   ```
4. Clone gsplat and locate the examples directory
   ```bash
   git clone https://github.com/nerfstudio-project/gsplat.git
   ```
5. In Blender preferences, set:
   - gsplat Examples Path: `.../gsplat/examples` (must contain `simple_trainer.py`)
   - gsplat Env: `gsplat` (or your env name)

## Start Training in Blender

1. Open the Training panel (N key -> GS Capture -> Training)
2. Select a backend
3. Set Training Data Path (the capture output folder)
4. Set Training Output Path
5. Set iterations and optional extra arguments
6. Click Start Training
