"""
Deep analysis of Gaussian Splatting rendering technique in the demo file.
"""
import bpy
import json

def analyze_material_shaders():
    """Analyze material shader nodes for GS rendering."""
    print("\n" + "="*80)
    print("MATERIAL SHADER ANALYSIS")
    print("="*80)

    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            continue

        print(f"\n### Material: {mat.name}")
        print(f"Blend Method: {mat.blend_method}")
        print(f"Shadow Method: {mat.shadow_method}")
        print(f"Use Backface Culling: {mat.use_backface_culling}")

        # Look for attribute nodes (these read GS data)
        print("\nAttribute nodes (reading GS data):")
        for node in mat.node_tree.nodes:
            if node.type == 'ATTRIBUTE':
                attr_name = node.attribute_name if hasattr(node, 'attribute_name') else 'unknown'
                print(f"  - {node.name}: reads '{attr_name}'")

        # List all shader nodes
        print("\nAll shader nodes:")
        for node in mat.node_tree.nodes:
            print(f"  - {node.name} ({node.type})")

def analyze_geometry_nodes():
    """Analyze geometry node setup for GS rendering."""
    print("\n" + "="*80)
    print("GEOMETRY NODES ANALYSIS")
    print("="*80)

    for nt in bpy.data.node_groups:
        if 'Geometry' not in nt.bl_idname:
            continue

        # Look for key rendering nodes
        has_instance_on_points = False
        has_mesh_to_points = False
        has_ico_sphere = False
        has_simulation = False

        attribute_reads = []
        attribute_stores = []

        for node in nt.nodes:
            if node.bl_idname == 'GeometryNodeInstanceOnPoints':
                has_instance_on_points = True
            if node.bl_idname == 'GeometryNodeMeshToPoints':
                has_mesh_to_points = True
            if node.bl_idname == 'GeometryNodeMeshIcoSphere':
                has_ico_sphere = True
            if 'Simulation' in node.bl_idname:
                has_simulation = True
            if node.bl_idname == 'GeometryNodeInputNamedAttribute':
                for inp in node.inputs:
                    if inp.name == 'Name' and hasattr(inp, 'default_value'):
                        attribute_reads.append(inp.default_value)
            if node.bl_idname == 'GeometryNodeStoreNamedAttribute':
                for inp in node.inputs:
                    if inp.name == 'Name' and hasattr(inp, 'default_value'):
                        attribute_stores.append(inp.default_value)

        print(f"\n### Node Tree: {nt.name}")
        print(f"Total nodes: {len(nt.nodes)}")
        print(f"\nRendering technique indicators:")
        print(f"  - Instance on Points: {has_instance_on_points}")
        print(f"  - Mesh to Points: {has_mesh_to_points}")
        print(f"  - Ico Sphere (for billboards): {has_ico_sphere}")
        print(f"  - Simulation nodes: {has_simulation}")

        if attribute_reads:
            print(f"\nGaussian attributes READ:")
            for attr in sorted(set(attribute_reads)):
                print(f"  - {attr}")

        if attribute_stores:
            print(f"\nAttributes STORED (computed):")
            for attr in sorted(set(attribute_stores)):
                print(f"  - {attr}")

def analyze_objects():
    """Analyze scene objects with GS data."""
    print("\n" + "="*80)
    print("SCENE OBJECT ANALYSIS")
    print("="*80)

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        mesh = obj.data
        if not mesh.attributes:
            continue

        # Check for GS-specific attributes
        gs_attrs = ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
                   'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity']

        found_gs = False
        for attr in mesh.attributes:
            if attr.name in gs_attrs or attr.name.startswith('f_rest') or attr.name.startswith('sh_'):
                found_gs = True
                break

        if found_gs:
            print(f"\n### Object: {obj.name}")
            print(f"Vertex count: {len(mesh.vertices)}")
            print(f"Modifiers: {[m.type for m in obj.modifiers]}")

            print("\nMesh attributes:")
            for attr in mesh.attributes:
                print(f"  - {attr.name}: {attr.data_type}, domain={attr.domain}")

def print_summary():
    """Print summary of findings."""
    print("\n" + "="*80)
    print("SUMMARY: HOW THIS RENDERS GAUSSIAN SPLATS")
    print("="*80)

    summary = """
This is NOT true Gaussian Splatting rendering. It is a visualization approximation.

TECHNIQUE: "Point Cloud with Instanced Geometry"

1. INPUT: PLY file with Gaussian attributes per vertex
   - Position (x,y,z)
   - Scale (scale_0, scale_1, scale_2) - log space
   - Rotation (rot_0-3) - quaternion
   - Color (f_dc_0-2) - spherical harmonics DC term
   - Opacity - sigmoid-transformed alpha

2. GEOMETRY NODES PROCESSING:
   - Read all attributes from mesh/point cloud
   - Convert quaternion to rotation
   - Apply exp() to scales
   - Store computed values as new attributes

3. INSTANCING:
   - Create low-poly sphere (Ico Sphere subdivisions=1)
   - Instance on every Gaussian point
   - Scale/rotate each instance by Gaussian covariance

4. MATERIAL SHADING:
   - Read color from f_dc attributes (converted to RGB)
   - Apply opacity via alpha blending
   - Transparent BSDF mixed with color

5. LIMITATIONS vs TRUE 3DGS:
   - No tile-based rasterization (slow for millions of splats)
   - No proper depth sorting per-pixel
   - No 2D Gaussian evaluation in screen space
   - No EWA splatting
   - Instanced meshes, not actual splats

This approach works for visualization but is NOT production-quality 3DGS.
"""
    print(summary)

def main():
    blend_path = "C:/Projects/GS_Blender/Blender-3DGS-4DGS-Viewer-Node/Blender-GSViewer-Node.blend"
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    analyze_geometry_nodes()
    analyze_material_shaders()
    analyze_objects()
    print_summary()

if __name__ == "__main__":
    main()
