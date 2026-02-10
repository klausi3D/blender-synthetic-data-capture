# GS Capture - Release Checklist

This checklist is for preparing a public release of the GS Capture addon.

**Scope**
1. Confirm the release scope and target Blender versions.
2. Confirm supported training backends and required dependencies.

**Pre-Release QA**
1. Run a clean capture with default settings and confirm output folders and files.
2. Run a capture with masks (alpha and object index) and confirm mask naming.
3. Run a capture with depth and normals enabled and confirm files are written.
4. Verify checkpoint resume by interrupting and resuming a capture.
5. Validate Windows path length warnings by testing a long output path.
6. Verify transforms.json loads in Nerfstudio (splatfacto) and/or gsplat.
7. Verify COLMAP files load in the original 3DGS pipeline.

**Documentation**
1. Update `gs_capture_addon/docs/USER_GUIDE.md` with current behavior and outputs.
2. Update `docs/guides/training-setup.md` and `docs/guides/pipeline-setup.md` if scripts changed.
3. Update `CHANGELOG.md` with the release summary and date.
4. Ensure version and feature list match the release.

**Versioning**
1. Update `bl_info` version in `gs_capture_addon/__init__.py`.
2. If you ship a ZIP, update the filename and any version references.
3. Confirm `doc_url` and `tracker_url` in `gs_capture_addon/__init__.py`.

**Packaging**
1. Ensure the addon package includes `gs_capture_addon/` and `gs_capture_addon.py` shim.
2. Build the zip with `python scripts/package_addon.py`.
3. Exclude large training repos or sample data from the release zip.
4. Verify the zip installs in Blender without errors.

**Release**
1. Draft release notes from `CHANGELOG.md`.
2. Tag the release in git.
3. Upload the release zip and release notes.

**Post-Release**
1. Install the release build in a clean Blender profile and smoke test.
2. Verify user guide links and training instructions.
3. Monitor issues and collect regression reports.
