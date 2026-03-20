# Splat Cleanup

## Goal

Splat cleanup removes floating gaussians after training by filtering points outside object proxy hulls.

## How It Works

1. Capture exports `cleanup/proxy_hulls.json` (enable `Export Proxy Hulls`).
2. Training produces a `.ply` model.
3. Cleanup filters splat points whose centers are outside all proxy hull volumes.
4. Outputs:
   - `*.cleaned.ply`
   - `cleanup_report.json`

## Automatic vs Manual

- Automatic: enable `Auto Cleanup After Training` in the `Training` panel.
- Manual: click `Run Cleanup Now` in the `Training` panel.

## Safety Controls

- `Hull Margin`: expands the keep volume slightly to avoid over-pruning.
- `Max Removal Ratio`: aborts cleanup if too many splats would be removed.

## Import Behavior

When `Prefer Cleaned Splat Import` is enabled, the importer uses `*.cleaned.ply` when available.

## Troubleshooting

- `Missing cleanup/proxy_hulls.json`:
  - Re-run capture with `Export Proxy Hulls` enabled.
- `Cleanup aborted because removal ratio exceeded safety limit`:
  - Increase `Max Removal Ratio` cautiously or raise `Hull Margin`.
- `No trained .ply found`:
  - Verify backend training output path and model export.
