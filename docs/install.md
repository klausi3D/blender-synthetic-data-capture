# Install

## Requirements

- Blender 4.0 or newer
- Windows recommended for training backend integration
- Optional: conda environments for training backends

## Addon Installation

1. Open Blender.
2. Go to `Edit -> Preferences -> Add-ons`.
3. Click `Install...` and select `gs_capture_addon-<version>.zip`.
4. Enable `GS Capture - Gaussian Splatting Training Data Generator`.

## Optional Backend Preferences

Open `Edit -> Preferences -> Add-ons -> GS Capture`.

Set paths/env names only for backends you plan to use:

- 3D Gaussian Splatting
  - Path to repo containing `train.py`
  - Conda env name
- Nerfstudio
  - Conda env name
- GS-Lightning
  - Path to repo containing `main.py`
  - Conda env name
- gsplat
  - Path to `gsplat/examples` containing `simple_trainer.py`
  - Conda env name

## Verify Installation

1. In 3D Viewport press `N`.
2. Open the `GS Capture` tab.
3. Confirm capture panels load without errors.

