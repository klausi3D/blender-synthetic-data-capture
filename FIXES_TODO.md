# GS Capture Pro - Fixes & Improvements TODO

This document tracks known issues and improvements for the gs_capture_addon.
Generated: 2024-12-26

---

## HIGH PRIORITY (Data loss, crashes, silent failures)

### 1. Checkpoint System - Data Loss Risk

- [x] **1.1 Race condition in atomic write** ✅ FIXED
  - File: `utils/checkpoint.py` lines 42-50
  - Problem: Uses `os.remove()` then `os.rename()` - if crash between them, checkpoint lost
  - Fix: Use `os.replace()` which is atomic on both Windows and Unix

- [x] **1.2 Checkpoint saved BEFORE image verified** ✅ FIXED
  - File: `operators/capture.py` line 427
  - Problem: Image marked complete before file is verified on disk
  - Fix: Verify file exists and size > 0 before updating checkpoint

- [x] **1.3 Partial/corrupt files not detected on resume** ✅ FIXED
  - File: `utils/checkpoint.py` line 174
  - Problem: Only checks `os.path.exists()`, not file integrity
  - Fix: Check file size > 0, optionally validate PNG header

- [x] **1.4 No error handling on checkpoint write** ✅ FIXED
  - File: `utils/checkpoint.py` lines 42-50
  - Problem: Disk full or permission denied causes silent failure
  - Fix: Wrap in try-except, clean up temp file on failure, return status

### 2. Image Save - Silent Failures

- [x] **2.1 No error handling on image save** ✅ FIXED
  - File: `operators/capture.py` lines 417-424
  - Problem: `save_render()` can fail silently (disk full, permission denied)
  - Fix: Wrap in try-except, verify file exists after save

- [ ] **2.2 No disk full handling in export**
  - File: `core/export.py` lines 95, 112, 201, 281
  - Problem: File writes have no error handling
  - Fix: Try-except around all file operations, report errors to user
  - **Note:** File keeps getting modified externally - needs manual fix

### 3. Validation Gaps - Potential Crashes

- [x] **3.1 Array access without length check: cameras[0]** ✅ FIXED
  - File: `core/export.py` lines 101, 226
  - Problem: Assumes cameras list is not empty
  - Fix: Add `if not cameras: raise ValueError("No cameras")`

- [x] **3.2 Array access without length check: color_attributes[0]** ✅ ALREADY SAFE
  - File: `utils/materials.py` lines 93, 207
  - Problem: Assumes color_attributes exists
  - Note: Already had proper validation in place

- [ ] **3.3 ViewLayer "ViewLayer" assumed to exist**
  - File: `operators/capture.py` line 116
  - Problem: Hardcoded name may not exist
  - Fix: Use `scene.view_layers[0]` or check existence

- [x] **3.4 object_groups index without bounds check** ✅ FIXED
  - File: `operators/batch.py` lines 141, 165
  - Problem: `object_groups[active_group_index]` can be out of bounds
  - Fix: Validate `0 <= index < len(object_groups)`

---

## MEDIUM PRIORITY (Performance bottlenecks)

### 4. O(n^2) Algorithms - Slow Analysis

- [x] **4.1 Nested loop iterates ALL edges for each vertex** ✅ FIXED
  - File: `core/analysis.py` lines 122-135 and 284-305
  - Problem: `for edge in mesh.edges` inside vertex loop is O(V * E)
  - Fix: Use adjacency map for O(V * avg_neighbors)
  - Impact: 10-50x faster on high-poly meshes

- [ ] **4.2 O(n^2) clustering in coverage suggestions**
  - File: `utils/coverage.py` lines 261-301
  - Problem: Compares every position to every other position
  - Fix: Use KD-tree or spatial hash for O(N log N)

- [ ] **4.3 Vertex coverage calculation is expensive**
  - File: `utils/coverage.py` lines 49-68
  - Problem: O(Vertices * Cameras) with BVH raycast per check
  - Fix: Frustum cull cameras first, cache BVH, batch queries

### 5. Blocking I/O on Main Thread

- [ ] **5.1 Checkpoint writes block render loop**
  - File: `operators/capture.py` line 432
  - Problem: JSON write happens on main thread every N frames
  - Fix: Move to background thread or increase interval

- [x] **5.2 JSON formatted with indent=2** ✅ FIXED
  - File: `utils/checkpoint.py` line 45
  - Problem: Larger file size, slower to write
  - Fix: Use `indent=None` for compact JSON

### 6. Redundant Computations

- [ ] **6.1 Two separate loops over objects in analysis**
  - File: `core/analysis.py` lines 312-399
  - Problem: First loop for mesh analysis, second for texture analysis
  - Fix: Combine into single pass over objects

- [ ] **6.2 Format dict recreated every frame**
  - File: `operators/capture.py` lines 400-414
  - Problem: `format_to_ext` dict and path formatting done per render
  - Fix: Cache extension and base path in `execute()`

- [ ] **6.3 Surface area uses matrix multiply per triangle vertex**
  - File: `core/analysis.py` lines 62-93
  - Problem: Expensive matrix operations in loop
  - Fix: Transform all vertices once, then calculate areas

---

## LOW PRIORITY (Edge cases, code quality)

### 7. Error Reporting Inconsistency

- [ ] **7.1 Uses print() instead of self.report()**
  - Files: `utils/checkpoint.py`, `utils/lighting.py`, `core/export.py`
  - Problem: Errors printed to console, users don't see them
  - Fix: Return error status, let calling operator report to user

### 8. Windows Compatibility

- [ ] **8.1 Path length > 260 chars not validated**
  - Files: Multiple (capture.py, export.py, checkpoint.py)
  - Problem: Windows MAX_PATH limit causes silent failures
  - Fix: Check path length before writing, warn user

- [ ] **8.2 Unix-style paths in search list**
  - File: `utils/lighting.py` lines 68-74
  - Problem: `/opt/gaussian-splatting` doesn't exist on Windows
  - Fix: Use platform-specific search paths

### 9. Resume Edge Cases

- [ ] **9.1 Camera count mismatch not handled**
  - File: `operators/capture.py` lines 320-326
  - Problem: If camera count changed, old indices may be invalid
  - Fix: Compare checkpoint total_cameras with current count

- [ ] **9.2 Settings hash doesn't cover all critical settings**
  - File: `utils/checkpoint.py` lines 183-213
  - Problem: Render engine, format, transparency not in hash
  - Fix: Include all settings that affect output

- [ ] **9.3 Deleted output folder not detected on resume**
  - File: `operators/capture.py`
  - Problem: Checkpoint exists but output folder was deleted
  - Fix: Verify directory structure before resuming

---

## COMPLETED FIXES

- [x] Fix `render_resolution_x/y` non-existent properties (analysis.py)
- [x] Remove unused numpy import (capture.py)
- [x] Fix bare except clause (capture.py)
- [x] Move `import random` to module level (camera.py)
- [x] Add render speed presets (properties.py, capture.py, panels/render.py)
- [x] Add size estimation feature (validation.py, panels/output.py)

---

## Implementation Notes

### Quick Wins (< 5 min each)
1. `os.replace()` - 1 line change in checkpoint.py
2. `indent=None` - 1 line change in checkpoint.py
3. `if not cameras:` guards - 2-3 lines in export.py
4. `vert.link_edges` - Replace nested loop pattern

### Requires More Care
1. Error handling in capture loop - need to handle retry/skip logic
2. Background checkpoint writes - threading considerations
3. Coverage optimization - algorithmic change

### Testing Checklist
- [ ] Test checkpoint resume after simulated crash
- [ ] Test with empty camera list
- [ ] Test with high-poly mesh (100K+ verts)
- [ ] Test on Windows with long paths
- [ ] Test disk full scenario
