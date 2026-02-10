# GS Capture Branching and CI/CD

## Recommended branch model

1. Stable release branch:
`release/4.5-lts`

2. Ongoing development:
`main`

3. Blender 5.0 compatibility work:
`feature/blender-5.0-compat`

## How to use this model

1. Keep `release/4.5-lts` pinned to Blender 4.5.1 LTS verified behavior.
2. Merge only tested bug fixes and release metadata changes into `release/4.5-lts`.
3. Develop and test Blender 5.0 API changes in `feature/blender-5.0-compat`.
4. Merge `feature/blender-5.0-compat` into `main` only after green CI and manual verification.
5. Cherry-pick only safe fixes from `main` into `release/4.5-lts` when needed.

## CI workflow

Workflow file: `.github/workflows/ci.yml`

It runs on push/PR to `main` and `release/**`:

1. Python sanity checks:
- compile all addon Python files
- package addon zip (`tools/package_addon.py`)

2. Blender smoke tests on Windows (Blender 4.5.1):
- `tests/smoke/smoke_release_verification.py`
- `tests/smoke/smoke_checkpoint_resume_only.py`
- `tests/smoke/smoke_object_index_mask.py`
- report validation with `tests/smoke/verify_ci_smoke_reports.py`

Artifacts:
- packaged zip
- smoke test reports and generated outputs

## CD workflow

Workflow file: `.github/workflows/release.yml`

1. Trigger:
- manual (`workflow_dispatch`)
- git tag push matching `v*` (for example `v2.2.2`)

2. Actions:
- build addon zip
- upload artifact
- publish GitHub release with zip attachment (tag-triggered runs)

## Branch protection recommendations

Set branch protection on `release/4.5-lts` and `main`:

1. Require status checks:
- `python-sanity`
- `blender-smoke-windows-4-5`

2. Require pull request reviews.
3. Restrict direct pushes.
