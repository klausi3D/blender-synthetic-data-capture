# Training Backends

## Supported Backends

### 3D Gaussian Splatting

- Input: COLMAP export (`sparse/0`)
- Preference path: repository containing `train.py`

### Nerfstudio (splatfacto)

- Input: `transforms.json`
- Preference: conda env name

### GS-Lightning

- Input: COLMAP export
- Optional masks supported (use GS-Lightning mask format when needed)
- Preference path: repository containing `main.py`

### gsplat

- Input: COLMAP or `transforms.json`
- Preference path: `gsplat/examples` containing `simple_trainer.py`

## Start Training From Blender

1. Open the `Training` panel.
2. Choose backend.
3. Set training data path and output path.
4. Configure iterations/arguments.
5. Click `Start Training`.

## Common Validation Requirements

- COLMAP backends: `sparse/0` must exist
- `transforms.json` backends: JSON file must exist and match image paths
- Windows: keep output paths short to avoid path-length issues

## Post-Training Cleanup

- The addon can run proxy-hull cleanup after training for all built-in backends.
- Cleanup is geometry-based (`cleanup/proxy_hulls.json`) and does not depend on backend-specific prune behavior.
- Backend-native knobs can still help reduce floaters:
  - Nerfstudio: `cull_alpha_thresh`, `continue_cull_post_densification`, `use_scale_regularization`
  - gsplat: strategy prune parameters such as `prune_opa`
  - GS-Lightning / 3DGS: regularization and densification tuning
