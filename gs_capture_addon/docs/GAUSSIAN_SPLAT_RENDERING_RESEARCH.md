# Gaussian Splat Rendering in Blender - Research Notes

**Date:** December 2024
**Status:** Research Complete - Future Enhancement
**Priority:** Medium (post-launch feature)

---

## Executive Summary

Native Gaussian Splatting rendering in Blender is **NOT possible** with compute shaders through the Python API. Current implementations use a **Geometry Nodes + Material Shader workaround** that provides visualization but not true 3DGS quality.

---

## Research Sources

### Analyzed Projects

1. **Mediastorm 3DGS/4DGS Viewer Node**
   - Repository: https://github.com/mediastormDev/Blender-3DGS-4DGS-Viewer-Node
   - Creator: Zhi Wang / 4DV.ai (SIGGRAPH 2025 Best in Show)
   - Technique: Geometry Nodes with instanced Ico Spheres
   - File analyzed: `Blender-GSViewer-Node.blend`

2. **KIRI 3DGS Render Addon**
   - Repository: https://github.com/Kiri-Innovation/3dgs-render-blender-addon
   - Technique: Geometry Nodes (`KIRI_3DGS_Render_GN`)
   - Supports: EEVEE only (not Cycles for real-time)

3. **Blender 5.0 4D Gaussian Splatting Demo**
   - Source: https://www.blender.org/download/demo-files/
   - File: `4D_Gaussian_Splatting-Nunchucks_and_cat.blend` (1GB)
   - Works with both EEVEE and Cycles

---

## Technical Findings

### Blender GPU API Limitations

```python
# What Blender's gpu module provides:
import gpu
gpu.shader.from_builtin()      # Vertex + Fragment shaders only
gpu.types.GPUShader            # No compute shader support
gpu.types.GPUOffScreen         # Offscreen rendering
gpu.state                      # Blend modes, depth test

# What's MISSING:
# - Compute shaders
# - Shader Storage Buffer Objects (SSBOs)
# - glDispatchCompute()
# - Atomic operations
```

**The `bgl` module is deprecated** and will be removed in Blender 6.0.

### Workaround: ModernGL for Compute Shaders

```python
# External library approach (NOT integrated with EEVEE/Cycles)
import moderngl
ctx = moderngl.create_context()  # Requires OpenGL 4.30+

compute_shader = ctx.compute_shader("""
#version 430
layout(local_size_x=256) in;
layout(std430, binding=0) buffer Data { vec4 points[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    // Process Gaussian splat...
}
""")

buffer = ctx.buffer(data)
buffer.bind_to_storage_buffer(0)
compute_shader.run(groups_x=num_splats//256)
```

**Limitation:** Data must be copied between ModernGL and Blender contexts. Cannot render directly to EEVEE/Cycles viewport.

---

## Current Implementation Technique (Geometry Nodes)

### Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PLY Point Cloud │ --> │  Geometry Nodes  │ --> │  EEVEE/Cycles   │
│  (Gaussian data) │     │  (190 nodes!)    │     │  Material       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Input Attributes (from 3DGS PLY file)

| Attribute | Type | Description |
|-----------|------|-------------|
| `x, y, z` | float | Gaussian center position |
| `scale_0, scale_1, scale_2` | float | Covariance scale (log space) |
| `rot_0, rot_1, rot_2, rot_3` | float | Rotation quaternion (w,x,y,z) |
| `f_dc_0, f_dc_1, f_dc_2` | float | DC spherical harmonics (base color) |
| `f_rest_0` ... `f_rest_44` | float | Higher-order SH coefficients |
| `opacity` | float | Alpha (before sigmoid) |

### Geometry Nodes Processing

```
1. GeometryNodeInputNamedAttribute
   - Read: scale_0, scale_1, scale_2, rot_0-3, f_dc_0-2, opacity

2. Math Operations
   - exp(scale) → linear scale
   - sigmoid(opacity) → alpha [0,1]
   - SH to RGB conversion

3. FunctionNodeQuaternionToRotation
   - Convert rot_0-3 to Blender rotation

4. GeometryNodeMeshIcoSphere
   - Create template sphere (subdivisions=1)

5. GeometryNodeInstanceOnPoints
   - Instance sphere on each Gaussian position
   - Apply scale and rotation per-instance

6. GeometryNodeStoreNamedAttribute
   - Store computed: SH_0 (color), computeAlpha
```

### Material Setup

```
Blend Method: HASHED (dithered alpha, avoids sorting)
Shadow Method: NONE or HASHED

Shader Nodes:
├── Attribute Node (reads "SH_0") → Base Color
├── Attribute Node (reads "computeAlpha") → Alpha
├── Transparent BSDF
├── Principled BSDF or Emission
└── Mix Shader → Material Output
```

---

## Comparison: True 3DGS vs Geometry Nodes Approach

| Aspect | True 3DGS Rendering | Geometry Nodes Approach |
|--------|---------------------|-------------------------|
| **Splat Shape** | 2D Gaussian in screen space | 3D Ico Sphere meshes |
| **Sorting** | Per-tile, per-pixel depth sort | No sorting (HASHED dither) |
| **Rasterization** | Custom tile-based compute shader | Standard EEVEE/Cycles |
| **Performance** | Millions of splats @ 60fps | ~50K-100K splats max |
| **View-Dependent Color** | Full SH evaluation per-view | DC term only (approximation) |
| **Memory** | Compact point buffer | Full mesh instances |
| **Quality** | Production-ready | Preview/visualization only |

---

## Implementation Roadmap for GS Viewer Feature

### Phase 1: Basic PLY Viewer (Low Effort)
- [ ] Import PLY with Gaussian attributes
- [ ] Simple point cloud visualization
- [ ] Color from f_dc (no SH)
- **Effort:** 1-2 days

### Phase 2: Geometry Nodes Splat Viewer (Medium Effort)
- [ ] Create GN node tree programmatically
- [ ] Instance Ico Spheres on points
- [ ] Apply scale/rotation transforms
- [ ] Basic alpha blending (HASHED)
- **Effort:** 3-5 days

### Phase 3: Enhanced Viewer (High Effort)
- [ ] Spherical harmonics evaluation (view-dependent color)
- [ ] LOD system for large splat counts
- [ ] Decimation/culling controls
- [ ] Animation support for 4DGS
- **Effort:** 1-2 weeks

### Phase 4: True Compute Shader Rendering (Research)
- [ ] Investigate Blender C++ modification
- [ ] ModernGL integration experiments
- [ ] Wait for Blender 6.0 compute shader API
- **Effort:** Unknown (depends on Blender development)

---

## Code Snippets for Future Implementation

### Loading 3DGS PLY in Blender

```python
import bpy
import numpy as np
from plyfile import PlyData

def load_gaussian_splat_ply(filepath):
    """Load 3DGS PLY file as Blender mesh with attributes."""
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']

    # Create mesh
    mesh = bpy.data.meshes.new("GaussianSplat")
    n_points = len(vertex['x'])

    # Add vertices
    mesh.vertices.add(n_points)
    positions = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
    mesh.vertices.foreach_set('co', positions.flatten())

    # Add Gaussian attributes
    attr_mapping = {
        'scale_0': 'FLOAT', 'scale_1': 'FLOAT', 'scale_2': 'FLOAT',
        'rot_0': 'FLOAT', 'rot_1': 'FLOAT', 'rot_2': 'FLOAT', 'rot_3': 'FLOAT',
        'f_dc_0': 'FLOAT', 'f_dc_1': 'FLOAT', 'f_dc_2': 'FLOAT',
        'opacity': 'FLOAT',
    }

    for attr_name, attr_type in attr_mapping.items():
        if attr_name in vertex.dtype.names:
            attr = mesh.attributes.new(attr_name, attr_type, 'POINT')
            attr.data.foreach_set('value', vertex[attr_name])

    mesh.update()
    return mesh
```

### Creating Splat Material

```python
def create_splat_material():
    """Create material for Gaussian splat visualization."""
    mat = bpy.data.materials.new("GaussianSplatMaterial")
    mat.use_nodes = True
    mat.blend_method = 'HASHED'
    mat.shadow_method = 'NONE'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Attribute nodes for color and alpha
    attr_color = nodes.new('ShaderNodeAttribute')
    attr_color.attribute_name = 'SH_0'
    attr_color.attribute_type = 'INSTANCER'

    attr_alpha = nodes.new('ShaderNodeAttribute')
    attr_alpha.attribute_name = 'computeAlpha'
    attr_alpha.attribute_type = 'INSTANCER'

    # Emission shader (no lighting dependency)
    emission = nodes.new('ShaderNodeEmission')

    # Transparent shader
    transparent = nodes.new('ShaderNodeBsdfTransparent')

    # Mix based on alpha
    mix = nodes.new('ShaderNodeMixShader')

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')

    # Connect
    links.new(attr_color.outputs['Color'], emission.inputs['Color'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(attr_alpha.outputs['Fac'], mix.inputs['Fac'])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    return mat
```

### Geometry Nodes Setup (Pseudocode)

```python
def create_splat_geometry_nodes():
    """Create Geometry Nodes modifier for splat instancing."""
    node_tree = bpy.data.node_groups.new("GaussianSplatGN", 'GeometryNodeTree')

    # Input/Output
    group_input = node_tree.nodes.new('NodeGroupInput')
    group_output = node_tree.nodes.new('NodeGroupOutput')

    # Read attributes
    read_scale0 = create_attribute_node(node_tree, 'scale_0')
    read_scale1 = create_attribute_node(node_tree, 'scale_1')
    read_scale2 = create_attribute_node(node_tree, 'scale_2')
    # ... rotation, color, opacity

    # Math: exp(scale)
    exp_scale0 = create_math_node(node_tree, 'EXPONENT', read_scale0)
    # ...

    # Combine into vector
    combine_scale = node_tree.nodes.new('ShaderNodeCombineXYZ')
    # ...

    # Create Ico Sphere template
    ico = node_tree.nodes.new('GeometryNodeMeshIcoSphere')
    ico.inputs['Radius'].default_value = 1.0
    ico.inputs['Subdivisions'].default_value = 1

    # Instance on Points
    instance = node_tree.nodes.new('GeometryNodeInstanceOnPoints')
    # Connect: points -> instance, ico -> instance, scale -> instance, rotation -> instance

    # Store computed attributes for material
    store_color = node_tree.nodes.new('GeometryNodeStoreNamedAttribute')
    store_color.inputs['Name'].default_value = 'SH_0'
    # ...

    return node_tree
```

---

## External Resources

### Documentation
- Blender GPU Module: https://docs.blender.org/api/current/gpu.html
- 3DGS Paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- gsplat Library: https://github.com/nerfstudio-project/gsplat

### Reference Implementations
- Original 3DGS CUDA: https://github.com/graphdeco-inria/gaussian-splatting
- Web Viewer (WebGL): https://github.com/antimatter15/splat
- UnrealGaussianSplatting: https://github.com/xverse-engine/XV3DGS-UEPlugin

### Blender Addons
- KIRI 3DGS Render: https://github.com/Kiri-Innovation/3dgs-render-blender-addon
- Mediastorm Viewer: https://github.com/mediastormDev/Blender-3DGS-4DGS-Viewer-Node

---

## Conclusion

For the GS Capture addon, we recommend:

1. **Short-term:** Bundle a basic PLY viewer using the Geometry Nodes approach
2. **Medium-term:** Enhance with LOD and decimation for large splat counts
3. **Long-term:** Monitor Blender development for native compute shader support

The Geometry Nodes approach is sufficient for **preview and verification** of captured/trained data, even if not suitable for final production rendering.

---

*This document should be updated when Blender adds compute shader support or new rendering techniques become available.*
