# Gaussian Splatting Training Pipeline - Setup Guide

This repository includes two CLI scripts for training outside Blender:

1. `tools/train_gs_simple.py` - simple training for one folder or batch folders
2. `tools/gs_training_pipeline.py` - pipeline with optional COLMAP and batching

---

## Quick Start (Blender Addon Output)

If you exported `transforms.json`, you can train directly without COLMAP.

```bash
# Single capture
python tools/train_gs_simple.py ./gs_capture/MyObject

# Batch process all capture folders
python tools/train_gs_simple.py ./gs_capture --batch --output ./trained_models
```

---

## Option A - Original 3D Gaussian Splatting

The default path is the original 3DGS repo. You must run the script in an environment where 3DGS dependencies are installed.

```bash
# Clone
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting

# Create env (example)
conda create -n gaussian_splatting python=3.8
conda activate gaussian_splatting

# Install dependencies
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Run training from that environment:

```bash
conda activate gaussian_splatting
python /path/to/GS_Blender/tools/train_gs_simple.py /path/to/capture
```

You can also set a repo path:

```bash
python tools/train_gs_simple.py ./gs_capture/MyObject --gs-path /path/to/gaussian-splatting
```

---

## Option B - Nerfstudio (splatfacto)

`tools/train_gs_simple.py` supports Nerfstudio with the `--nerfstudio` flag:

```bash
pip install nerfstudio
python tools/train_gs_simple.py ./gs_capture/MyObject --nerfstudio
```

---

## Using the Pipeline Script

`tools/gs_training_pipeline.py` is for larger batches and optional COLMAP processing.

```bash
# Process folders with default settings
python tools/gs_training_pipeline.py --input ./captures --output ./trained

# Skip COLMAP (recommended for Blender output)
python tools/gs_training_pipeline.py --input ./captures --output ./trained --skip-colmap

# Recursive scan
python tools/gs_training_pipeline.py --input ./captures --output ./trained --recursive
```

You can also use a YAML config:

```yaml
# pipeline_config.yaml
input_folders:
  - "./captures"

output_base: "./trained"

colmap:
  enabled: false
  skip_if_transforms_exist: true

training:
  implementation: "original"  # or "nerfstudio"
  iterations: 30000
  white_background: true
```

```bash
python tools/gs_training_pipeline.py --config pipeline_config.yaml
```

Note: `implementation: "gsplat"` is not implemented in the pipeline script and will log a warning.

---

## COLMAP (Only for Real Photos)

Blender output already includes camera parameters if you export `transforms.json`. COLMAP is only required for real photographs or video frames.

---

## Output Structure Reference

```
output/
  ObjectName/
    images/
      image_0000.png
      image_0001.png
    transforms.json
    sparse/0/           # optional (COLMAP)
      cameras.txt
      images.txt
      points3D.txt
```

---

## Troubleshooting

- "gaussian-splatting not found": clone the repo or use `--gs-path`
- "nerfstudio not found": install nerfstudio or remove `--nerfstudio`
- No images found: make sure your capture folder has `images/` and images inside
