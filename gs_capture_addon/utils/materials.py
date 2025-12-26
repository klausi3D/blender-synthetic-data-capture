"""
Material override utilities for neutral capture.
Handles material replacement and restoration for GS training data.
"""

import bpy


def override_materials(objects, override_type='DIFFUSE'):
    """Override materials on objects for neutral capture.

    Args:
        objects: List of Blender mesh objects
        override_type: One of 'DIFFUSE', 'VERTEX_COLOR', 'MATCAP'

    Returns:
        dict: Dictionary mapping object names to original material lists
    """
    original_materials = {}

    for obj in objects:
        if obj.type != 'MESH':
            continue

        original_materials[obj.name] = list(obj.data.materials)

        if override_type == 'DIFFUSE':
            _apply_diffuse_material(obj)

        elif override_type == 'VERTEX_COLOR':
            _apply_vertex_color_material(obj)

        elif override_type == 'MATCAP':
            _apply_matcap_material(obj)

    return original_materials


def _apply_diffuse_material(obj):
    """Apply neutral diffuse material to object.

    Args:
        obj: Blender mesh object
    """
    mat_name = "GS_Neutral_Diffuse"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        bsdf = nodes.new('ShaderNodeBsdfDiffuse')
        bsdf.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)
        bsdf.location = (0, 0)

        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (200, 0)

        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    else:
        mat = bpy.data.materials[mat_name]

    # Apply to all slots
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _apply_vertex_color_material(obj):
    """Apply vertex color material to object.

    Uses vertex colors if present, otherwise falls back to neutral gray.

    Args:
        obj: Blender mesh object
    """
    mat_name = f"GS_VertexColor_{obj.name}"

    # Check if object has vertex colors
    mesh = obj.data
    has_vertex_colors = len(mesh.color_attributes) > 0

    if has_vertex_colors:
        # Create material that uses vertex colors
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Vertex color node
        vc_node = nodes.new('ShaderNodeVertexColor')
        vc_node.layer_name = mesh.color_attributes[0].name
        vc_node.location = (-200, 0)

        # Diffuse shader
        bsdf = nodes.new('ShaderNodeBsdfDiffuse')
        bsdf.location = (0, 0)

        # Output
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (200, 0)

        # Connect
        links.new(vc_node.outputs['Color'], bsdf.inputs['Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    else:
        # No vertex colors, use neutral diffuse
        mat_name = "GS_Neutral_Diffuse"
        if mat_name not in bpy.data.materials:
            _apply_diffuse_material(obj)
            return
        mat = bpy.data.materials[mat_name]

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _apply_matcap_material(obj):
    """Apply matcap-style material for visualization.

    Creates a simple emission material based on normals.

    Args:
        obj: Blender mesh object
    """
    mat_name = "GS_Matcap"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Geometry node for normals
        geometry = nodes.new('ShaderNodeNewGeometry')
        geometry.location = (-400, 0)

        # Vector transform to view space
        transform = nodes.new('ShaderNodeVectorTransform')
        transform.vector_type = 'NORMAL'
        transform.convert_from = 'WORLD'
        transform.convert_to = 'CAMERA'
        transform.location = (-200, 0)

        # Map range to 0-1
        map_range = nodes.new('ShaderNodeMapRange')
        map_range.inputs['From Min'].default_value = -1
        map_range.inputs['From Max'].default_value = 1
        map_range.inputs['To Min'].default_value = 0
        map_range.inputs['To Max'].default_value = 1
        map_range.location = (0, 0)

        # Combine XYZ for RGB
        combine = nodes.new('ShaderNodeCombineXYZ')
        combine.location = (200, 0)

        # Emission shader
        emission = nodes.new('ShaderNodeEmission')
        emission.location = (400, 0)

        # Output
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (600, 0)

        # Connect
        links.new(geometry.outputs['Normal'], transform.inputs['Vector'])
        links.new(transform.outputs['Vector'], emission.inputs['Color'])
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
    else:
        mat = bpy.data.materials[mat_name]

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def restore_materials(original_materials):
    """Restore original materials.

    Args:
        original_materials: Dictionary from override_materials()
    """
    for obj_name, materials in original_materials.items():
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            if obj.type == 'MESH':
                obj.data.materials.clear()
                for mat in materials:
                    obj.data.materials.append(mat)


def create_vertex_color_material(obj, color_attribute_name=None):
    """Create a material that displays vertex colors.

    Args:
        obj: Blender mesh object
        color_attribute_name: Name of color attribute to use (default: first available)

    Returns:
        Material or None
    """
    mesh = obj.data

    if not mesh.color_attributes:
        return None

    attr_name = color_attribute_name or mesh.color_attributes[0].name

    mat_name = f"GS_VC_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Vertex color node
    vc = nodes.new('ShaderNodeVertexColor')
    vc.layer_name = attr_name
    vc.location = (-200, 0)

    # Principled BSDF for better shading
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Connect
    links.new(vc.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def cleanup_gs_materials():
    """Remove all materials created by GS Capture.

    Cleans up temporary materials to avoid clutter.
    """
    mats_to_remove = [mat for mat in bpy.data.materials
                      if mat.name.startswith('GS_')]

    for mat in mats_to_remove:
        if mat.users == 0:
            bpy.data.materials.remove(mat)
