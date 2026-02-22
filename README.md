# Blender Synthetic Data Capture

Blender Synthetic Data Capture is a Blender addon for capturing synthetic training datasets for 3D Gaussian Splatting and NeRF pipelines.

## Documentation

- User docs site: https://klausi3d.github.io/blender-synthetic-data-capture/
- In-repo addon guide: `gs_capture_addon/docs/USER_GUIDE.md`

## Install

Use the packaged addon zip named `gs_capture_addon-<version>.zip`.
Do not use GitHub's auto-generated `Source code (zip)` / `blender-synthetic-data-capture-*.zip` archive, because Blender will not detect the addon module from that archive layout.

1. Download `gs_capture_addon-<version>.zip` from Releases (or build with `python3 tools/package_addon.py`).
2. In Blender open `Edit -> Preferences -> Add-ons`.
3. Click `Install...` and select the zip.
4. Enable `Blender Synthetic Data Capture`.

## Branches

- `release/4.5-lts`: stable Blender 4.5.1 LTS branch
- `feature/blender-5.0-compat`: Blender 5.0 compatibility work

## License

This project is licensed under the MIT License. See `LICENSE`.
