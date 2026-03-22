# Automation Pipeline (CLI)

End-to-end orchestrator that takes a `.blend` file through capture, GS training, and splat cleanup in one command.

---

## Quick Start

```bash
# Single job from config
python tools/pipeline.py --config jobs/my_job.yaml

# Multiple jobs
python tools/pipeline.py --config jobs/a.yaml jobs/b.yaml

# Preview without executing
python tools/pipeline.py --config jobs/my_job.yaml --dry-run
```

---

## Three Phases

1. **Capture** — launches Blender with `headless_capture.py` to render multi-view images and export COLMAP/transforms.json.
2. **Training** — runs a Gaussian Splatting backend (3DGS, Nerfstudio, gsplat, etc.) on the captured images.
3. **Cleanup** — filters the trained `.ply` model using proxy hull volumes to remove floating splats.

Each phase is optional and can be enabled/disabled in the config.

---

## Config File

See [`jobs/example_capture.yaml`](https://github.com/klausi3D/blender-synthetic-data-capture/blob/master/jobs/example_capture.yaml) for the full config schema. Key sections:

```yaml
blender_path: "blender"
scene: "scene.blend"
output_base: "/output/path"

capture:
  cameras: 150
  preset: "3dgs"
  engine: EEVEE

training:
  enabled: true
  backend: "original"
  iterations: 30000

cleanup:
  enabled: true
  proxy_hulls: true
```

---

## Pipeline Report

Each run writes `pipeline_report.json` to the output directory with per-phase timings, success/failure status, and error details.

---

## Relationship to Other Tools

| Tool | Purpose |
|------|---------|
| `tools/pipeline.py` | Orchestrates a single .blend through all phases |
| `tools/headless_capture.py` | Capture-only (called by pipeline.py) |
| `tools/library_capture.py` | Batch-processes a folder of .blend files using pipeline.py |

For batch processing many `.blend` files, use [Library Batch Capture](library-batch-capture.md) instead.

---

## Notes

- Blender is auto-detected if on PATH, or set `blender_path` in config.
- The pipeline runs outside Blender (standalone Python).
- Training requires the chosen backend to be installed in the active Python environment.
