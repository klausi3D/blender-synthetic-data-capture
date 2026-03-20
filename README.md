# Blender Synthetic Data Capture

[![CI](https://github.com/klausi3D/blender-synthetic-data-capture/actions/workflows/ci.yml/badge.svg)](https://github.com/klausi3D/blender-synthetic-data-capture/actions/workflows/ci.yml)

A professional-grade Blender addon for generating synthetic training datasets for 3D Gaussian Splatting and NeRF pipelines.

## Features

- **Multi-distribution camera placement** — Fibonacci sphere, hemisphere, ring, multi-ring, and adaptive hotspot-biased distributions
- **Framework presets** — One-click setup for 3DGS, Nerfstudio, Postshot, Polycam, and Luma AI
- **Rich export outputs** — RGB images, depth maps, normal maps, and object masks
- **COLMAP & transforms.json** — Direct export of camera parameters for training without COLMAP
- **Checkpoint/resume** — Resume interrupted captures without re-rendering completed frames
- **Coverage analysis** — Viewport heatmap showing camera coverage of mesh surfaces
- **Scene validation** — Pre-capture checks for scene issues, settings, output paths, and disk space
- **Material & scene analysis** — Detects material problems and estimates capture quality
- **Batch capture** — Headless CLI pipeline for asset libraries with per-collection capture
- **Splat cleanup** — Proxy-hull-based point cloud cleanup to remove outlier splats

## Supported Blender Versions

- **Blender 4.5.x LTS** (stable)
- **Blender 5.0.x** (compatible)

## Install

Use the packaged addon zip named `gs_capture_addon-<version>.zip`.
Do **not** use GitHub's auto-generated `Source code (zip)` archive — Blender will not detect the addon module from that layout.

1. Download `gs_capture_addon-<version>.zip` from [Releases](https://github.com/klausi3D/blender-synthetic-data-capture/releases) (or build with `python tools/package_addon.py`).
2. In Blender, open **Edit > Preferences > Add-ons**.
3. Click **Install...** and select the zip.
4. Enable **Blender Synthetic Data Capture**.

The addon panel appears in **View3D > Sidebar > GS Capture**.

## Documentation

Full user documentation: <https://klausi3d.github.io/blender-synthetic-data-capture/>

Topics covered: installation, quick start, capture and export options, training backends, splat cleanup, CLI pipeline setup, and troubleshooting.

## Architecture Overview

```
gs_capture_addon/
├── core/           # Camera math, analysis, validation, training backends
├── operators/      # Blender operators (capture, preview, export, training)
├── ui/             # Panels and UI components
└── utils/          # Checkpoint, paths, lighting, materials, coverage
tools/              # CLI scripts (training, pipeline, headless capture, packaging)
tests/
├── python/         # Pure Python unit tests (no Blender required)
└── smoke/          # Blender smoke tests (run in CI with xvfb)
```

## CLI Training Scripts

Two standalone scripts for training outside Blender:

```bash
# Simple single-folder training
python tools/train_gs_simple.py ./gs_capture/MyObject

# Batch pipeline with optional COLMAP
python tools/gs_training_pipeline.py --input ./captures --output ./trained --skip-colmap
```

See [Pipeline Setup](https://klausi3d.github.io/blender-synthetic-data-capture/guides/pipeline-setup/) for details.

## Branches

| Branch | Purpose |
|--------|---------|
| `master` | Main development branch |
| `release/4.5-lts` | Stable Blender 4.5.1 LTS release |
| `feature/blender-5.0-compat` | Blender 5.0 compatibility work |

## Contributing

Issues and feature requests: <https://github.com/klausi3D/blender-synthetic-data-capture/issues>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
