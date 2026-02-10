# Quick Start

## Basic Capture Flow

1. Select one or more mesh objects.
2. Open `N -> GS Capture`.
3. Set `Output Path`.
4. Optionally apply a framework preset.
5. Optionally click `Preview` to inspect camera positions.
6. Click `Capture Selected`.

## Minimum Recommended Settings

- Camera Count: `100+` for training quality
- Distribution: `Fibonacci Sphere` for uniform coverage
- Output format: PNG for images
- Export COLMAP or `transforms.json` based on your training backend

## Useful Preset Mapping

| Backend | Typical Export |
|---|---|
| 3D Gaussian Splatting | COLMAP (`sparse/0`) |
| Nerfstudio (splatfacto) | `transforms.json` |
| GS-Lightning | COLMAP (+ optional masks) |
| gsplat | COLMAP or `transforms.json` |

## Resume Support

- Enable checkpoints to save progress every N images.
- If capture is interrupted, restart capture with the same settings to resume.

