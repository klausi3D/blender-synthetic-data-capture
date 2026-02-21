# Troubleshooting

## No Objects To Capture

- Select at least one mesh object before running capture.

## Missing COLMAP Files

- Enable `Export COLMAP Format`.
- Re-run capture and confirm `sparse/0` contains the three `.txt` files.

## Mask Export Issues

- Alpha mask path:
  - Enable transparent background.
  - Use PNG or EXR output.
- Object-index path:
  - Assign object pass indices.
  - Use `Mask Source: Object Index`.

## Capture Does Not Resume

- Enable checkpoints.
- Keep output path/settings consistent between runs.
- Do not delete checkpoint or output files mid-run.

## Path Too Long On Windows

- Use short output roots, for example `C:\<capture_root>\scene01`.
- Avoid deep nested directories in output location.

## Backend Not Found

- Re-check addon preference paths and conda env names.
- Verify required scripts exist (`train.py`, `main.py`, `simple_trainer.py`).
