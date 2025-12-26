"""
Framework presets for popular 3DGS/NeRF training pipelines.
Each preset configures optimal settings for a specific framework.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class FrameworkPreset:
    """Configuration preset for a training framework."""

    # Identification
    id: str
    name: str
    description: str
    website: str

    # Camera settings
    recommended_cameras: Tuple[int, int]  # (min, max)
    camera_distribution: str
    min_elevation: float
    max_elevation: float

    # Render settings
    recommended_resolution: Tuple[int, int]
    aspect_ratio: Optional[str]  # None = any, "1:1", "16:9", etc.
    file_format: str
    jpeg_quality: int
    transparent_background: bool

    # Export settings
    export_colmap: bool
    export_transforms_json: bool
    colmap_binary: bool
    aabb_scale: int

    # Special requirements
    requires_test_set: bool
    test_set_percentage: float
    white_background: bool
    notes: List[str]


# Framework preset definitions
PRESETS = {
    'GAUSSIAN_SPLATTING': FrameworkPreset(
        id='GAUSSIAN_SPLATTING',
        name='3D Gaussian Splatting',
        description='Original 3DGS by Kerbl et al. (SIGGRAPH 2023)',
        website='https://github.com/graphdeco-inria/gaussian-splatting',

        recommended_cameras=(100, 300),
        camera_distribution='FIBONACCI',
        min_elevation=-60,
        max_elevation=60,

        recommended_resolution=(1920, 1080),
        aspect_ratio=None,
        file_format='PNG',
        jpeg_quality=95,
        transparent_background=False,

        export_colmap=True,
        export_transforms_json=False,
        colmap_binary=False,
        aabb_scale=16,

        requires_test_set=False,
        test_set_percentage=0,
        white_background=True,
        notes=[
            'Requires COLMAP format (sparse/0/)',
            'White background recommended for object captures',
            'More cameras = better quality but longer training',
        ]
    ),

    'INSTANT_NGP': FrameworkPreset(
        id='INSTANT_NGP',
        name='InstantNGP',
        description='NVIDIA Instant Neural Graphics Primitives',
        website='https://github.com/NVlabs/instant-ngp',

        recommended_cameras=(100, 200),
        camera_distribution='FIBONACCI',
        min_elevation=-45,
        max_elevation=45,

        recommended_resolution=(800, 800),
        aspect_ratio='1:1',
        file_format='PNG',
        jpeg_quality=95,
        transparent_background=False,

        export_colmap=False,
        export_transforms_json=True,
        colmap_binary=False,
        aabb_scale=16,

        requires_test_set=False,
        test_set_percentage=0,
        white_background=True,
        notes=[
            'Uses transforms.json format',
            'Square images often work best',
            'aabb_scale controls scene bounds',
            'Fast training (~5 minutes on RTX 3090)',
        ]
    ),

    'NERFSTUDIO': FrameworkPreset(
        id='NERFSTUDIO',
        name='Nerfstudio (splatfacto)',
        description='Nerfstudio framework with Gaussian Splatting',
        website='https://docs.nerf.studio/',

        recommended_cameras=(150, 300),
        camera_distribution='FIBONACCI',
        min_elevation=-60,
        max_elevation=60,

        recommended_resolution=(1920, 1080),
        aspect_ratio=None,
        file_format='PNG',
        jpeg_quality=95,
        transparent_background=False,

        export_colmap=False,
        export_transforms_json=True,
        colmap_binary=False,
        aabb_scale=16,

        requires_test_set=True,
        test_set_percentage=10,
        white_background=True,
        notes=[
            'Supports both NeRF and Gaussian Splatting',
            'Use "ns-train splatfacto" for 3DGS',
            'Requires transforms_train.json and transforms_test.json',
            'Good balance of quality and speed',
        ]
    ),

    'POSTSHOT': FrameworkPreset(
        id='POSTSHOT',
        name='Postshot',
        description='Jawset Postshot Gaussian Splatting',
        website='https://jawset.com/',

        recommended_cameras=(50, 150),
        camera_distribution='FIBONACCI',
        min_elevation=-45,
        max_elevation=60,

        recommended_resolution=(1920, 1080),
        aspect_ratio=None,
        file_format='JPEG',
        jpeg_quality=95,
        transparent_background=False,

        export_colmap=True,
        export_transforms_json=True,
        colmap_binary=False,
        aabb_scale=16,

        requires_test_set=False,
        test_set_percentage=0,
        white_background=True,
        notes=[
            'Commercial software with GUI',
            'Supports both COLMAP and transforms.json',
            'JPEG recommended to reduce file sizes',
            'Good for beginners',
        ]
    ),

    'POLYCAM': FrameworkPreset(
        id='POLYCAM',
        name='Polycam',
        description='Polycam 3D Gaussian Splatting',
        website='https://poly.cam/',

        recommended_cameras=(100, 200),
        camera_distribution='FIBONACCI',
        min_elevation=-30,
        max_elevation=60,

        recommended_resolution=(1080, 1080),
        aspect_ratio='1:1',
        file_format='JPEG',
        jpeg_quality=90,
        transparent_background=False,

        export_colmap=True,
        export_transforms_json=False,
        colmap_binary=False,
        aabb_scale=16,

        requires_test_set=False,
        test_set_percentage=0,
        white_background=True,
        notes=[
            'Mobile-first platform',
            'Square images preferred',
            'JPEG for smaller uploads',
            'Web-based training',
        ]
    ),

    'LUMA_AI': FrameworkPreset(
        id='LUMA_AI',
        name='Luma AI',
        description='Luma AI Gaussian Splatting',
        website='https://lumalabs.ai/',

        recommended_cameras=(100, 200),
        camera_distribution='FIBONACCI',
        min_elevation=-45,
        max_elevation=60,

        recommended_resolution=(1920, 1080),
        aspect_ratio=None,
        file_format='PNG',
        jpeg_quality=95,
        transparent_background=False,

        export_colmap=True,
        export_transforms_json=False,
        colmap_binary=False,
        aabb_scale=16,

        requires_test_set=False,
        test_set_percentage=0,
        white_background=True,
        notes=[
            'High camera overlap recommended',
            'Web-based training',
            'Good for high-quality captures',
        ]
    ),

    'GSPLAT': FrameworkPreset(
        id='GSPLAT',
        name='gsplat',
        description='Nerfstudio gsplat library',
        website='https://github.com/nerfstudio-project/gsplat',

        recommended_cameras=(100, 300),
        camera_distribution='FIBONACCI',
        min_elevation=-60,
        max_elevation=60,

        recommended_resolution=(1920, 1080),
        aspect_ratio=None,
        file_format='PNG',
        jpeg_quality=95,
        transparent_background=False,

        export_colmap=True,
        export_transforms_json=True,
        colmap_binary=False,
        aabb_scale=16,

        requires_test_set=False,
        test_set_percentage=0,
        white_background=True,
        notes=[
            'Optimized CUDA implementation',
            'Compatible with Nerfstudio data format',
            'Faster training than original 3DGS',
        ]
    ),
}


def get_preset(preset_id: str) -> Optional[FrameworkPreset]:
    """Get a framework preset by ID.

    Args:
        preset_id: Preset identifier

    Returns:
        FrameworkPreset or None if not found
    """
    return PRESETS.get(preset_id)


def get_all_presets() -> dict:
    """Get all available presets.

    Returns:
        Dictionary of preset_id -> FrameworkPreset
    """
    return PRESETS.copy()


def apply_preset_to_settings(preset: FrameworkPreset, settings, scene):
    """Apply a framework preset to GS Capture settings.

    Args:
        preset: FrameworkPreset to apply
        settings: GSCaptureSettings instance
        scene: Blender scene for render settings
    """
    # Camera settings
    settings.camera_count = (preset.recommended_cameras[0] + preset.recommended_cameras[1]) // 2
    settings.camera_distribution = preset.camera_distribution
    settings.min_elevation = preset.min_elevation
    settings.max_elevation = preset.max_elevation

    # Render settings (applied to Blender's settings)
    scene.render.resolution_x = preset.recommended_resolution[0]
    scene.render.resolution_y = preset.recommended_resolution[1]
    scene.render.image_settings.file_format = preset.file_format
    if preset.file_format == 'JPEG':
        scene.render.image_settings.quality = preset.jpeg_quality

    # GS Capture settings
    settings.transparent_background = preset.transparent_background

    # Export settings
    settings.export_colmap = preset.export_colmap
    settings.export_transforms_json = preset.export_transforms_json

    # Store current preset for reference
    settings.current_preset = preset.id


def get_preset_enum_items():
    """Get preset items for Blender EnumProperty.

    Returns:
        List of (id, name, description) tuples
    """
    items = [('CUSTOM', "Custom", "Manual settings configuration")]

    for preset_id, preset in PRESETS.items():
        items.append((
            preset_id,
            preset.name,
            preset.description
        ))

    return items
