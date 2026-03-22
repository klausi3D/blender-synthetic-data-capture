# Library Batch Capture (CLI)

Scan a folder of `.blend` files and run capture, training, and cleanup for every collection.

---

## Quick Start

```bash
# Full batch with config file
python tools/library_capture.py --config jobs/example_library.yaml

# Inline config (no file)
python tools/library_capture.py --library-path /assets --output /data

# Preview without executing
python tools/library_capture.py --config lib.yaml --dry-run

# Introspection only (discover + analyze, no capture)
python tools/library_capture.py --config lib.yaml --introspect-only

# Parallel execution on multiple GPUs
python tools/library_capture.py --config lib.yaml --max-workers 4
```

---

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | none | YAML config file |
| `--library-path` | none | Folder of `.blend` files to scan |
| `--output` | none | Base output directory |
| `--blender-path` | `blender` | Path to Blender executable |
| `--granularity` | `COLLECTIONS` | `COLLECTIONS`, `SCENE`, or `EACH_OBJECT` |
| `--max-workers` | `1` | Parallel workers (1 = sequential) |
| `--dry-run` | off | Preview pipeline without executing |
| `--introspect-only` | off | Analyze collections only, no capture |
| `--adaptive` | off | Auto-detect optimal camera count/resolution |
| `--no-skip-existing` | off | Force re-capture of already-processed items |

---

## Workflow

The pipeline runs in five phases:

1. **Discovery** — scans `library_path` for `.blend` files, applies include/exclude filters.
2. **Introspection** — launches Blender headless per file to discover collections, mesh counts, vertex totals.
3. **Job Generation** — creates a pipeline config per collection, respecting filters and skip-existing logic.
4. **Execution** — runs capture/train/cleanup jobs (optionally in parallel with GPU load-balancing).
5. **Reporting** — writes `library_report.json` with per-file status, timings, and error messages.

---

## Config File

See [`jobs/example_library.yaml`](https://github.com/klausi3D/blender-synthetic-data-capture/blob/master/jobs/example_library.yaml) for the full config schema, including filtering, adaptive analysis, and capture/training/cleanup settings.

---

## Notes

- Use `--introspect-only` to preview what would be captured before committing to a long batch.
- `skip_existing: true` (default) avoids re-processing items that already have a `headless_capture_summary.json`.
- GPU assignment uses round-robin across `gpu_ids` for parallel workers.
