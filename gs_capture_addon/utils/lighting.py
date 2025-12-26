"""
Lighting setup utilities for neutral capture environments.
Provides various background and lighting configurations for GS training.
"""

import bpy


def get_eevee_engine_name():
    """Get the correct Eevee engine name for this Blender version.

    Blender 4.2+ uses BLENDER_EEVEE_NEXT, but 5.0 reverted to BLENDER_EEVEE.

    Returns:
        str: The correct Eevee engine identifier
    """
    scene = bpy.context.scene if hasattr(bpy.context, 'scene') else None
    if scene:
        try:
            render_prop = scene.bl_rna.properties['render'].fixed_type.properties['engine']
            available_engines = [item.identifier for item in render_prop.enum_items]
            if 'BLENDER_EEVEE_NEXT' in available_engines:
                return 'BLENDER_EEVEE_NEXT'
            elif 'BLENDER_EEVEE' in available_engines:
                return 'BLENDER_EEVEE'
        except Exception as e:
            print(f"Engine detection warning: {e}")

    # Fallback: version-based detection
    if bpy.app.version >= (4, 2, 0) and bpy.app.version < (5, 0, 0):
        return 'BLENDER_EEVEE_NEXT'
    else:
        return 'BLENDER_EEVEE'


def store_lighting_state(context):
    """Store current lighting state for restoration.

    Args:
        context: Blender context

    Returns:
        dict: Dictionary mapping object names to (hide_render, hide_viewport) tuples
    """
    states = {}
    for obj in context.scene.objects:
        if obj.type == 'LIGHT':
            states[obj.name] = (obj.hide_render, obj.hide_viewport)
    return states


def restore_lighting(context, stored_light_states):
    """Restore original lighting state.

    Args:
        context: Blender context
        stored_light_states: Dictionary from store_lighting_state()
    """
    for obj_name, (hide_render, hide_viewport) in stored_light_states.items():
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            obj.hide_render = hide_render
            obj.hide_viewport = hide_viewport


def setup_neutral_lighting(context, settings):
    """Setup neutral lighting environment for capture.

    Supports WHITE, GRAY, and HDR background modes.

    Args:
        context: Blender context
        settings: GSCaptureSettings with lighting configuration
    """
    world = context.scene.world
    if not world:
        world = bpy.data.worlds.new("GS_Capture_World")
        context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Create output node
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (300, 0)

    if settings.lighting_mode == 'WHITE':
        # Pure white background
        background = nodes.new('ShaderNodeBackground')
        background.location = (0, 0)
        background.inputs['Color'].default_value = (1, 1, 1, 1)
        background.inputs['Strength'].default_value = settings.background_strength

        links.new(background.outputs['Background'], output.inputs['Surface'])

    elif settings.lighting_mode == 'GRAY':
        # Gray background with configurable value
        background = nodes.new('ShaderNodeBackground')
        background.location = (0, 0)
        gray = settings.gray_value
        background.inputs['Color'].default_value = (gray, gray, gray, 1)
        background.inputs['Strength'].default_value = settings.background_strength

        links.new(background.outputs['Background'], output.inputs['Surface'])

    elif settings.lighting_mode == 'HDR':
        # HDR environment map
        if settings.hdr_path:
            # Create environment texture
            env_tex = nodes.new('ShaderNodeTexEnvironment')
            env_tex.location = (-300, 0)

            # Load HDR image
            try:
                img = bpy.data.images.load(bpy.path.abspath(settings.hdr_path))
                env_tex.image = img
            except Exception as e:
                print(f"Failed to load HDR: {e}")
                # Fallback to white
                background = nodes.new('ShaderNodeBackground')
                background.inputs['Color'].default_value = (1, 1, 1, 1)
                links.new(background.outputs['Background'], output.inputs['Surface'])
                return

            background = nodes.new('ShaderNodeBackground')
            background.location = (0, 0)
            background.inputs['Strength'].default_value = settings.hdr_strength

            links.new(env_tex.outputs['Color'], background.inputs['Color'])
            links.new(background.outputs['Background'], output.inputs['Surface'])
        else:
            # No HDR specified, use white
            background = nodes.new('ShaderNodeBackground')
            background.inputs['Color'].default_value = (1, 1, 1, 1)
            links.new(background.outputs['Background'], output.inputs['Surface'])

    # Disable scene lights if requested
    if settings.disable_scene_lights:
        for obj in context.scene.objects:
            if obj.type == 'LIGHT':
                obj.hide_render = True
                obj.hide_viewport = True


def create_studio_lighting(context, settings):
    """Create a simple 3-point lighting setup.

    Args:
        context: Blender context
        settings: GSCaptureSettings

    Returns:
        list: Created light objects
    """
    lights = []

    # Key light
    key_data = bpy.data.lights.new(name="GS_Key_Light", type='AREA')
    key_data.energy = 1000
    key_data.size = 5
    key_obj = bpy.data.objects.new("GS_Key_Light", key_data)
    key_obj.location = (5, -5, 8)
    key_obj.rotation_euler = (0.8, 0, 0.8)
    context.scene.collection.objects.link(key_obj)
    lights.append(key_obj)

    # Fill light
    fill_data = bpy.data.lights.new(name="GS_Fill_Light", type='AREA')
    fill_data.energy = 500
    fill_data.size = 3
    fill_obj = bpy.data.objects.new("GS_Fill_Light", fill_data)
    fill_obj.location = (-5, -3, 5)
    fill_obj.rotation_euler = (0.6, 0, -0.6)
    context.scene.collection.objects.link(fill_obj)
    lights.append(fill_obj)

    # Back light
    back_data = bpy.data.lights.new(name="GS_Back_Light", type='AREA')
    back_data.energy = 300
    back_data.size = 2
    back_obj = bpy.data.objects.new("GS_Back_Light", back_data)
    back_obj.location = (0, 5, 6)
    back_obj.rotation_euler = (2.4, 0, 0)
    context.scene.collection.objects.link(back_obj)
    lights.append(back_obj)

    return lights


def remove_gs_lights(context):
    """Remove all lights created by GS Capture.

    Args:
        context: Blender context
    """
    lights_to_remove = [obj for obj in context.scene.objects
                        if obj.type == 'LIGHT' and obj.name.startswith('GS_')]

    for light in lights_to_remove:
        bpy.data.objects.remove(light, do_unlink=True)
