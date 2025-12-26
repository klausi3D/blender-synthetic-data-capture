# Gaussian Splatting Training Pipeline - Setup Guide

## Overview

This pipeline automates the process of converting captured images (from the Blender GS Capture addon) into trained 3D Gaussian Splatting models.

**Two scripts are provided:**

1. `train_gs_simple.py` - Simple, direct training (recommended for Blender addon output)
2. `gs_training_pipeline.py` - Full pipeline with COLMAP support, queuing, checkpoints

---

## Quick Start (For Blender Addon Output)

Since the Blender addon exports `transforms.json` with camera parameters, you can skip COLMAP entirely and train directly:

```bash
# Single object
python train_gs_simple.py ./gs_capture/MyObject

# Batch process all captures
python train_gs_simple.py ./gs_capture --batch --output ./trained_models
```

---

## Installation Options

### Option 1: Original 3DGS (Recommended for Quality)

The original implementation from INRIA gives the best results:

```bash
# Clone the repository
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting

# Create conda environment
conda env create --file environment.yml
conda activate gaussian_splatting

# Set environment variable (add to ~/.bashrc or ~/.zshrc)
export GAUSSIAN_SPLATTING_PATH="$HOME/gaussian-splatting"
```

**Requirements:**
- CUDA 11.6+
- 24GB+ VRAM recommended (RTX 3090/4090, A5000, etc.)
- Your RTX 3090 is perfect for this

### Option 2: Nerfstudio (Easier Setup)

Nerfstudio has splatfacto which is easier to install:

```bash
# Create environment
conda create -n nerfstudio python=3.10
conda activate nerfstudio

# Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install nerfstudio
pip install nerfstudio

# Verify
ns-train --help
```

**Pros:** Easier install, built-in viewer, active development
**Cons:** Slightly different results than original 3DGS

### Option 3: gsplat (Library-Based)

For custom implementations:

```bash
pip install gsplat
```

Note: gsplat is a library, not a training script. Good if you want to build custom training loops.

---

## COLMAP Installation (Only for Real Photos)

**Not needed for Blender addon output** since we export camera parameters directly.

Only install if you want to process real photographs:

```bash
# Ubuntu/Debian
sudo apt install colmap

# Or build from source for GPU support
# See: https://colmap.github.io/install.html

# macOS
brew install colmap

# Windows
# Download from: https://github.com/colmap/colmap/releases
```

---

## Pipeline Configuration

For complex workflows, use the YAML config:

```yaml
# pipeline_config.yaml
input_folders:
  - "./gs_capture"

output_base: "./trained_models"

colmap:
  enabled: false  # Not needed for Blender output
  skip_if_transforms_exist: true

training:
  implementation: "original"  # or "nerfstudio"
  iterations: 30000
  white_background: true
```

Then run:
```bash
python gs_training_pipeline.py --config pipeline_config.yaml
```

---

## Recommended Workflow

### For Individual Objects from Blender:

1. In Blender, select object and run GS Capture
2. Train directly:
   ```bash
   python train_gs_simple.py ~/gs_capture/MyObject -i 30000
   ```
3. View result:
   ```bash
   # With original 3DGS viewer
   python gaussian-splatting/render.py -m ~/trained_models/MyObject/splat
   
   # Or with nerfstudio
   ns-viewer --load-config ~/trained_models/MyObject/config.yml
   ```

### For Batch Processing Multiple Objects:

1. Export all objects from Blender (each to separate folder)
2. Run batch training:
   ```bash
   python train_gs_simple.py ~/gs_capture --batch -o ~/trained_models
   ```
3. Script automatically:
   - Skips already-completed training
   - Handles failures
   - Continues with next object

### For Large Scenes with Many Collections:

1. Use Blender addon's "Analyze Scene" to plan
2. Run batch by collection
3. Use full pipeline for checkpointing:
   ```bash
   python gs_training_pipeline.py -i ~/gs_capture -o ~/trained --recursive
   ```

---

## Training Parameters Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| iterations | 30000 | Total training steps. 30k is standard, 50k+ for complex scenes |
| densify_until_iter | 15000 | When to stop adding gaussians |
| white_background | true | Enable for objects on white bg (from addon) |
| resolution | -1 (auto) | Training resolution. -1 uses image resolution |

**For Your RTX 3090 (24GB):**
- Can train up to ~4K resolution
- ~2-3 minutes per 1000 iterations at 1080p
- 30k iterations takes ~60-90 minutes per object

---

## Troubleshooting

### Out of Memory
```bash
# Reduce resolution
python train_gs_simple.py ./capture --resolution 1080

# Or use less densification
# Edit the training config to reduce densify_grad_threshold
```

### Training Stalls
- Check GPU utilization with `nvidia-smi`
- Ensure CUDA is properly installed
- Try smaller batch of images first

### Bad Results
- More cameras usually help (100+ recommended)
- Check image quality in the capture folder
- Ensure consistent lighting (use addon's neutral lighting)
- Try longer training (50k iterations)

### COLMAP Fails (Real Photos Only)
- Ensure images have good overlap (60%+)
- Check for motion blur
- Try sequential matcher for video frames:
  ```yaml
  colmap:
    matcher: "sequential"
  ```

---

## Viewing Results

### Original 3DGS Viewer
```bash
cd gaussian-splatting
python render.py -m /path/to/output/splat
```

### SIBR Viewer (Real-time)
```bash
# Build SIBR viewer (included in 3DGS repo)
cd gaussian-splatting/SIBR_viewers
cmake -B build .
cmake --build build --target install

# Run
./install/bin/SIBR_gaussianViewer_app -m /path/to/output/splat
```

### Nerfstudio Viewer
```bash
ns-viewer --load-config /path/to/output/config.yml
```

### Web Viewers
- [Luma AI WebGL Viewer](https://lumalabs.ai/luma-web-library)
- [Three.js Gaussian Splat Viewer](https://github.com/mkkellogg/GaussianSplats3D)
- [WebGPU Splat Viewer](https://github.com/cvlab-epfl/gaussian-splatting-web)

---

## Converting Results for Use in Game Engines

### For Your Grandma's House Project (Godot/UE5):

The trained `.ply` files can be converted:

```bash
# Location of trained model
ls /trained_models/MyObject/splat/point_cloud/iteration_30000/

# Output: point_cloud.ply
```

**For UE5:**
- Use 3DGS UE5 plugins (several available)
- Or convert to compatible format

**For Godot (your custom implementation):**
- Load the PLY directly
- Parse gaussian parameters (position, scale, rotation, SH coefficients)
- Render with your custom shader

---

## File Structure Reference

```
gs_capture/                      # From Blender addon
  ObjectName/
    images/
      image_0000.png
      image_0001.png
      ...
    transforms.json              # Camera parameters
    sparse/0/                    # COLMAP format (optional)
      cameras.txt
      images.txt
      points3D.txt

trained_models/                  # Training output
  ObjectName/
    splat/
      point_cloud/
        iteration_7000/
          point_cloud.ply
        iteration_30000/
          point_cloud.ply        # Final model
      cameras.json
      cfg_args
      input.ply
```
