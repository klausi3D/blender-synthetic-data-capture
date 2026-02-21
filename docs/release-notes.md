# Release Notes

## Current Addon Version

`2.2.2`

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

- Python sanity checks and packaging
- Windows smoke tests in Blender 4.5.1 and 5.0
- Checkpoint-only and object-index mask focused smoke tests
