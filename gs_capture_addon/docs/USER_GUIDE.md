# GS Capture - User Guide

A professional Blender addon for generating training data for 3D Gaussian Splatting and NeRF models.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Framework Presets](#framework-presets)
4. [Camera Configuration](#camera-configuration)
5. [Capture Settings](#capture-settings)
6. [Training Integration](#training-integration)
7. [Export Formats](#export-formats)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Blender 4.0 or newer
- GPU with OpenGL 3.3+ support (for viewport preview)
- For training: Python environment with training framework installed

### Installing the Addon

1. Download the `gs_capture_addon` folder
2. In Blender: **Edit → Preferences → Add-ons**
3. Click **Install...** and select the folder (or zip file)
4. Enable "GS Capture - Gaussian Splatting Training Data Generator"
5. Configure backend paths in Preferences (optional, for training)

### Configuring Training Backends (Optional)

To use integrated training, configure paths in addon preferences:

1. **Edit → Preferences → Add-ons → GS Capture**
2. Set paths for installed backends:
   - **3D Gaussian Splatting**: Path to repository containing `train.py`
   - **Nerfstudio**: Name of conda environment
   - **gsplat**: Path to examples directory

---

## Quick Start

### Basic Workflow

1. **Select your object(s)** in the 3D viewport
2. **Open the GS Capture panel** in the sidebar (N-key → GS Capture)
3. **Choose a framework preset** matching your target training framework
4. **Click "Apply Preset"** to configure optimal settings
5. **Set the output path** for captured images
6. **Click "Preview Cameras"** to visualize camera positions
7. **Click "Capture Selected"** to start rendering

### Recommended Settings by Use Case

| Use Case | Camera Count | Resolution | Format |
|----------|--------------|------------|--------|
| Quick test | 30-50 | 800x800 | PNG |
| Standard quality | 100-150 | 1280x1280 | PNG |
| High quality | 200-300 | 1920x1920 | PNG |
| Production | 300+ | 2048x2048 | PNG |

---

## Framework Presets

Framework presets configure capture settings optimized for specific training frameworks.

### Available Presets

#### 3D Gaussian Splatting (INRIA)
- **Cameras**: 100-300
- **Resolution**: 1920x1080
- **Format**: PNG with transparent background
- **Export**: COLMAP format
- Best for: Original 3DGS implementation

#### Instant-NGP
- **Cameras**: 50-200
- **Resolution**: 1920x1080
- **Format**: PNG
- **Export**: transforms.json
- Best for: Fast training with NVIDIA hardware

#### Nerfstudio
- **Cameras**: 100-200
- **Resolution**: 1280x720
- **Format**: PNG
- **Export**: transforms.json
- Best for: splatfacto and other Nerfstudio methods

#### Postshot
- **Cameras**: 50-150
- **Resolution**: 1920x1080
- **Format**: JPEG (quality 95)
- **Export**: COLMAP format
- Best for: Postshot's optimized pipeline

#### Polycam
- **Cameras**: 30-100
- **Resolution**: 1920x1080
- **Format**: JPEG
- **Export**: Both COLMAP and transforms.json
- Best for: Polycam-compatible exports

#### Luma AI
- **Cameras**: 100-300
- **Resolution**: 1920x1080
- **Format**: PNG
- **Export**: transforms.json
- Best for: Luma AI's API

#### gsplat
- **Cameras**: 100-300
- **Resolution**: 1920x1080
- **Format**: PNG
- **Export**: Both formats
- Best for: Nerfstudio's gsplat library

### Applying a Preset

1. Select the preset from the dropdown in the **Framework Preset** panel
2. Click the checkmark button to apply settings
3. Review the "Current Settings" subpanel to verify configuration
4. Adjust individual settings as needed

---

## Camera Configuration

### Distribution Patterns

#### Fibonacci Sphere
Evenly distributed cameras on a sphere using the Fibonacci spiral pattern.
Best for: Most general use cases, uniform coverage.

#### Top Hemisphere
Cameras positioned only above the object.
Best for: Objects meant to be viewed from above (e.g., tabletop items).

#### Bottom Hemisphere
Cameras positioned only below the object.
Best for: Objects viewed from below (e.g., ceiling fixtures).

#### Single Ring
Cameras in a horizontal ring around the object.
Best for: Quick 360° capture, turntable-style views.

#### Multi Ring
Multiple horizontal rings at different elevations.
Best for: Balanced coverage with controlled vertical distribution.

### Camera Parameters

- **Camera Count**: Total number of cameras to generate
- **Min/Max Elevation**: Vertical angle limits (degrees)
- **Distance Mode**: Auto (based on object size) or Manual
- **Distance Multiplier**: Scale factor for auto-calculated distance
- **Focal Length**: Camera focal length in mm (default: 50mm)

### Previewing Cameras

1. Click **"Preview Cameras"** to generate camera positions
2. Camera frustrums appear in the viewport as blue wireframes
3. Adjust settings and preview again as needed
4. Click **"Clear Preview"** to remove preview cameras

---

## Capture Settings

### Render Settings

Render settings are controlled through Blender's native Render Properties panel:
- **Render Engine**: Cycles or EEVEE
- **Resolution**: Set in Output Properties
- **Samples**: Set in Render Properties

### GS-Specific Settings

#### Output Options
- **Export COLMAP**: Generate COLMAP-compatible files
- **Export transforms.json**: Generate NeRF-format camera data
- **Export Depth Maps**: Save normalized depth for each view
- **Export Normal Maps**: Save world-space normals
- **Export Object Masks**: Save binary masks for target objects

#### Lighting Modes
- **White Background**: Pure white environment (recommended)
- **Gray Background**: Neutral gray environment
- **HDR Environment**: Custom HDR image for lighting
- **Keep Scene Lighting**: Preserve existing scene lights

#### Material Overrides
- **Original Materials**: Use object's materials
- **Neutral Diffuse**: Gray diffuse for geometry capture
- **Vertex Colors**: Display vertex color data
- **Matcap**: Matcap-style shading

### Checkpoint/Resume

For long capture sessions:

1. Enable **"Enable Checkpoints"** in settings
2. Set **"Checkpoint Interval"** (default: every 10 images)
3. If interrupted, the addon will prompt to resume from last checkpoint
4. Enable **"Auto Resume"** to automatically continue

---

## Training Integration

### Supported Backends

#### 3D Gaussian Splatting
Original implementation from INRIA/Graphdeco.

**Requirements:**
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

#### Nerfstudio
Nerfstudio framework with splatfacto.

**Requirements:**
```bash
pip install nerfstudio
```

#### gsplat
Optimized Gaussian Splatting library.

**Requirements:**
```bash
pip install gsplat
git clone https://github.com/nerfstudio-project/gsplat.git
```

### Starting Training

1. Complete a capture with valid data
2. Open the **Training** panel
3. Select your training backend
4. Set the training data path (your capture output)
5. Set the training output path
6. Configure iterations and other parameters
7. Click **"Start Training"**

### Monitoring Progress

During training:
- Progress bar shows current iteration
- Loss and PSNR metrics update in real-time
- ETA estimates remaining time
- Log panel shows raw output

### Stopping Training

Click **"Stop Training"** to terminate the process.
Partial results are preserved in the output directory.

---

## Export Formats

### COLMAP Format

Directory structure:
```
output/
├── images/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt
```

### transforms.json Format

```json
{
  "camera_angle_x": 0.7853,
  "frames": [
    {
      "file_path": "images/00000.png",
      "transform_matrix": [[...], [...], [...], [...]],
      "depth_path": "depth/00000.png",
      "mask_path": "masks/00000.png"
    }
  ]
}
```

### Optional Exports

#### Depth Maps
- Normalized 16-bit PNG or EXR
- Values: 0 = near, 1 = far
- Path: `depth/00000.png`

#### Normal Maps
- World-space normals as RGB
- Path: `normals/00000.png`

#### Object Masks
- Binary masks (white = object)
- Path: `masks/00000.png`

---

## Troubleshooting

### Common Issues

#### "No objects to capture"
- Ensure objects are selected before clicking capture
- Check that objects are mesh type (not empties, lights, etc.)

#### "Missing sparse/0 directory"
- Enable COLMAP export in settings
- Ensure capture completed successfully

#### Training won't start
- Verify backend path in addon preferences
- Check that required Python packages are installed
- Ensure CUDA is available for GPU training

#### Black renders
- Check lighting mode (try "White Background")
- Verify render engine is compatible with scene
- Check object visibility settings

#### Slow rendering
- Reduce sample count for faster preview
- Use EEVEE for quick tests
- Lower resolution for initial tests

### Getting Help

1. Check the console for error messages (Window → Toggle System Console)
2. Review the IMPLEMENTATION_PLAN.md for feature status
3. Submit issues at the project repository

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| N | Toggle sidebar (to access GS Capture panel) |
| Ctrl+Z | Undo last action |
| Esc | Cancel running operation |

---

## Version History

### v2.1.0 (Current)
- Added framework presets
- Integrated training support
- Added validation system
- Viewport camera preview
- Coverage analysis

### v2.0.0
- Complete modular rewrite
- Depth, normal, and mask export
- Checkpoint/resume system
- Batch processing

### v1.0.0
- Initial release
- Basic capture functionality
