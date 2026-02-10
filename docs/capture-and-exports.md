# Capture And Exports

## Camera Controls

- Distribution: sphere/hemisphere/ring patterns
- Camera Count: total samples
- Elevation limits: minimum and maximum vertical angle
- Distance mode: auto or manual
- Focal length: camera lens in mm

## Export Types

- RGB images: `images/image_0000.png`
- COLMAP: `sparse/0/cameras.txt`, `images.txt`, `points3D.txt`
- `transforms.json`: NeRF-compatible camera transforms
- Depth maps: `depth/depth_0000.<ext>`
- Normal maps: `normals/normal_0000.exr`
- Masks: `masks/mask_0000.<ext>` or GS-Lightning naming

## Blender 4.5 vs 5.0 Output Notes

Depth and object-index mask compositor outputs can differ by Blender major version:

| Blender Version | Depth Extension | Object-Index Mask Extension |
|---|---|---|
| 4.x / 4.5.1 LTS | `.png` | `.png` |
| 5.0 | `.exr` | `.exr` |

`transforms.json` paths follow the captured output extensions automatically.

## Alpha vs Object-Index Masks

- Alpha masks:
  - Require transparent background in render settings
  - Work with PNG/EXR image output
- Object-index masks:
  - Set pass indices on objects
  - Use `Mask Source: Object Index`

## Coverage Analysis

1. Generate preview cameras.
2. Run coverage analysis or heatmap.
3. Increase camera count or adjust elevation if low coverage is reported.

