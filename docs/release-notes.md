# Release Notes

## Current Addon Version

`2.3.0`

## 2.3.0 (2026-03-22)

1. Added end-to-end pipeline orchestrator (`tools/pipeline.py`) for config-driven capture, training, and cleanup.
2. Added headless capture script (`tools/headless_capture.py`) for GUI-free multi-view rendering.
3. Added asset library batch capture (`tools/library_capture.py`) with introspection, adaptive analysis, and parallel execution.
4. Added proxy-hull splat cleanup with automatic post-training PLY filtering.
5. Added config schema validation for capture and library job configs.
6. Added 700+ unit tests across 8 test suites (cleanup, camera math, checkpoints, paths, validation, config schema, library capture, realistic PLY).
7. Hardened config loading with error handling for missing files and parse failures.
8. Added documentation for headless capture, library batch capture, and automation pipeline.

## 2.2.2 (2026-02-21)

1. Training startup now uses normalized preflight paths and robust extra-arg parsing.
2. Training stop now terminates full process trees to prevent orphaned backend workers.
3. Checkpoint resume now validates expected exported files before skipping completed frames.
4. Capture/export now handles directory creation failures with explicit user-facing errors.
5. Release workflow now enforces tag/version matching and blocks publish until verification gates pass.

## 2.2.1 (2026-02-07)

1. Hardened capture pipeline with safer checkpoint/resume and output validation.
2. Added coverage validation with large-mesh safeguards.
3. Improved Windows compatibility for training backend discovery and path warnings.
4. Updated documentation, release checklist, and packaging guidance.

## Branch Policy

- `release/4.5-lts`: stable Blender 4.5.1 LTS release branch
- `feature/blender-5.0-compat`: Blender 5.0 compatibility branch

## CI Coverage

- Ruff linting (Pyflakes + pycodestyle errors and warnings, pinned to 0.12.5)
- 8 unit test suites (700+ assertions)
- Python compile checks and addon packaging
- mkdocs build with `--strict` mode
- Linux smoke tests in Blender 4.5.1 and 5.0.0 (10 smoke scripts)
- Tracked .blend file validation
