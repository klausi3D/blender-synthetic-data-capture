# Headless Capture (CLI)

Drive multi-view captures from the command line without a GUI.

---

## Quick Start

```bash
# Single capture with CLI args
blender scene.blend --python tools/headless_capture.py -- \
    --output /tmp/capture --cameras 100 --preset 3dgs

# Batch: capture every collection separately
blender scene.blend --python tools/headless_capture.py -- \
    --output /tmp/capture --batch-mode COLLECTIONS --cameras 150

# Config file mode
blender scene.blend --python tools/headless_capture.py -- \
    --config jobs/my_job.yaml
```

On headless Linux (e.g. CI), prefix with `xvfb-run -a`:

```bash
xvfb-run -a blender scene.blend --python tools/headless_capture.py -- ...
```

---

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | required | Output directory for images, depth, COLMAP, etc. |
| `--cameras` | `50` | Number of viewpoints to render |
| `--distribution` | `FIBONACCI` | Camera placement: `FIBONACCI`, `RANDOM`, `UNIFORM` |
| `--preset` | `3dgs` | Export preset: `3dgs`, `nerfstudio`, `colmap` |
| `--engine` | `EEVEE` | Render engine: `EEVEE`, `CYCLES` |
| `--resolution` | `1920x1080` | Image resolution (WxH) |
| `--batch-mode` | none | `COLLECTIONS` or `SCENE` |
| `--config` | none | YAML config file (overrides CLI args) |

---

## Config File

See [`jobs/example_capture.yaml`](https://github.com/klausi3D/blender-synthetic-data-capture/blob/master/jobs/example_capture.yaml) for the full config schema. The `capture` section maps to the CLI arguments above.

---

## Output Structure

```
output/
  images/
    image_0000.png
    image_0001.png
  depth/           (if depth export enabled)
  masks/           (if mask export enabled)
  sparse/0/        (COLMAP format)
    cameras.txt
    images.txt
    points3D.txt
  transforms.json  (Nerfstudio format)
```

---

## Notes

- The script runs inside Blender's Python environment via `--python`.
- Requires a `.blend` file as the scene source.
- Use `--factory-startup` to avoid user addon conflicts in CI.
