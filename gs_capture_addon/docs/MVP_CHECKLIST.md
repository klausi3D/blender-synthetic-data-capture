# GS Capture - MVP Checklist

**Goal:** Minimal, polished, sellable product at $29-39 price point

---

## What MVP INCLUDES (Core Value)

### ✅ Already Done
- [x] Multi-view camera generation (Fibonacci, hemisphere, ring, multi-ring)
- [x] COLMAP export (cameras.txt, images.txt, points3D.txt)
- [x] transforms.json export (NeRF/Instant-NGP format)
- [x] Framework presets (3DGS, Nerfstudio, Postshot, Polycam, Luma AI, gsplat)
- [x] Camera preview in viewport
- [x] Checkpoint/resume for long captures
- [x] Depth map export
- [x] Normal map export
- [x] Object mask export
- [x] Lighting overrides (white/gray/HDR background)
- [x] Material overrides (diffuse, vertex color)

### 🔴 MVP Must-Have (Not Done Yet)
- [ ] **Material Problem Detector** - Warns about reflective/transparent materials
- [ ] **Scene Complexity Score** - Simple metric showing if scene is suitable
- [ ] **Coverage Heatmap** - Visual feedback on camera coverage
- [ ] **UI Polish** - Clean, consistent, no confusion
- [ ] **End-to-end test** - Verify full workflow actually works
- [ ] **Documentation** - Clear quick-start guide

---

## What MVP EXCLUDES (Post-Launch)

### Remove from Current Build (Simplify)
- [ ] Training integration (3DGS, Nerfstudio, gsplat) → External training is fine
- [ ] Training panel UI → Not needed
- [ ] Addon preferences for training paths → Not needed
- [ ] Batch processing panel → Advanced feature
- [ ] Adaptive capture analysis → Nice-to-have, not core
- [ ] Object groups → Over-engineered

### Future Features (Not MVP)
- Cloud integration (Luma AI, Polycam API)
- Quality prediction
- Splat preview/viewer
- 4DGS animation support
- Batch asset processing

---

## MVP User Flow

```
1. Select objects
2. Choose preset (or adjust settings manually)
3. Set output path
4. [NEW] See warnings if problems detected
5. Preview cameras
6. [NEW] See coverage heatmap
7. Click "Capture"
8. Done → Export ready for training
```

---

## Files to Simplify/Remove for MVP

### REMOVE (not needed for MVP)
```
panels/training.py          → Training integration
operators/training.py       → Training operators
panels/batch.py            → Batch processing
panels/adaptive.py         → Adaptive capture (keep basic version)
core/training/             → Entire training module
preferences.py             → Training paths not needed
```

### SIMPLIFY
```
__init__.py                → Remove training/batch registrations
properties.py              → Remove training properties
panels/presets.py          → Simplify, remove "quick settings" subpanel
```

### ADD
```
core/material_analyzer.py  → Material problem detection
core/scene_score.py        → Simple complexity score
panels/warnings.py         → Warning display panel
operators/coverage.py      → Coverage heatmap operator
```

---

## MVP Panel Layout

```
┌─────────────────────────────────┐
│ GS Capture                      │
├─────────────────────────────────┤
│ Framework: [3DGS ▼] [Apply]     │
│                                 │
│ ⚠️ Warnings (if any)            │
│   • Glass material: transparent │
│   • Chrome: highly reflective   │
│   [Fix All] [Ignore]            │
│                                 │
│ Scene Score: ████████░░ Good    │
│                                 │
│ [Preview Cameras]  [Clear]      │
│ Coverage: ███████░░░ 72%        │
│                                 │
│ [═══ CAPTURE SELECTED ═══]      │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│ ▶ Camera Settings               │
├─────────────────────────────────┤
│ Count: [100]                    │
│ Distribution: [Fibonacci ▼]     │
│ ...                             │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│ ▶ Output Settings               │
├─────────────────────────────────┤
│ Path: [//output/]         [📁]  │
│ [x] COLMAP  [x] transforms.json │
│ [ ] Depth   [ ] Normals         │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│ ▶ Render Settings (use Blender) │
└─────────────────────────────────┘
```

---

## Implementation Priority

### Day 1: Material Analyzer
```python
# Simple implementation - check for common problems
- Detect blend_method != OPAQUE → Transparency warning
- Detect metallic > 0.5 → Reflective warning
- Detect transmission > 0 → Glass warning
- Detect emission strength > 0 → Emissive warning
```

### Day 2: Scene Score + Coverage
```python
# Scene score (simple heuristic)
- Count vertices, faces
- Check material issues
- Calculate bounding box ratio
- Output: "Excellent / Good / Fair / Poor"

# Coverage heatmap
- Use existing coverage.py code
- Add operator to visualize
- Show percentage in UI
```

### Day 3: UI Consolidation
```python
# Merge into single clean main panel
- Remove training panels
- Remove batch panel
- Simplify presets (dropdown + apply, no subpanel)
- Add warnings section
- Add scene score display
```

### Day 4: Testing & Polish
```
- Test full workflow with real scene
- Fix any bugs
- Clean up code
- Update documentation
```

---

## Success Criteria for MVP

- [ ] User can capture a scene in < 5 clicks
- [ ] Warnings prevent obvious mistakes
- [ ] UI is self-explanatory (no manual needed for basics)
- [ ] Output works with 3DGS, Nerfstudio, Instant-NGP
- [ ] No crashes or errors on typical scenes
- [ ] Documentation covers quick-start in < 2 minutes

---

## Pricing for MVP

**$29 - Single User License**
- All capture features
- All presets
- Material warnings
- Coverage visualization
- Email support
- 1 year updates

This is competitive with Camera Array Tool (€36) but offers:
- GS-specific features (presets, warnings)
- Multiple export formats
- Intelligence (not just cameras)
