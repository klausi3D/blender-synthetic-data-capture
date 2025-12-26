"""
Property Groups for GS Capture addon.
Defines all settings and state for the capture system.
"""

import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
    CollectionProperty,
)
from bpy.types import PropertyGroup


class GSCaptureObjectItem(PropertyGroup):
    """Single object reference for grouping."""
    obj: PointerProperty(
        type=bpy.types.Object,
        name="Object",
        description="Object in the group"
    )


class GSCaptureObjectGroup(PropertyGroup):
    """Group of objects for batch capture."""
    name: StringProperty(
        name="Group Name",
        default="Group"
    )
    objects: CollectionProperty(type=GSCaptureObjectItem)
    expanded: BoolProperty(default=True)


class GSCaptureCheckpoint(PropertyGroup):
    """Checkpoint state for resume functionality."""
    current_index: IntProperty(default=0)
    total_cameras: IntProperty(default=0)
    is_active: BoolProperty(default=False)
    output_path: StringProperty(default="")


class GSCaptureSettings(PropertyGroup):
    """Main settings for GS Capture addon."""

    # ==========================================================================
    # OUTPUT SETTINGS
    # ==========================================================================

    output_path: StringProperty(
        name="Output Path",
        description="Directory to save captured images",
        default="//gs_capture/",
        subtype='DIR_PATH'
    )

    export_colmap: BoolProperty(
        name="Export COLMAP Format",
        description="Export camera data in COLMAP format (sparse/0/)",
        default=True
    )

    export_transforms_json: BoolProperty(
        name="Export transforms.json",
        description="Export camera data in NeRF/3DGS format",
        default=True
    )

    export_depth: BoolProperty(
        name="Export Depth Maps",
        description="Export normalized depth maps for each view",
        default=False
    )

    export_normals: BoolProperty(
        name="Export Normal Maps",
        description="Export world-space normal maps",
        default=False
    )

    export_masks: BoolProperty(
        name="Export Object Masks",
        description="Export binary masks for target objects",
        default=False
    )

    # ==========================================================================
    # CAMERA SETTINGS
    # ==========================================================================

    camera_count: IntProperty(
        name="Camera Count",
        description="Number of cameras to generate",
        default=100,
        min=8,
        max=500
    )

    camera_distribution: EnumProperty(
        name="Distribution",
        description="How to distribute cameras around the object",
        items=[
            ('FIBONACCI', "Fibonacci Sphere", "Even distribution using Fibonacci spiral"),
            ('HEMISPHERE_TOP', "Top Hemisphere", "Cameras above the object"),
            ('HEMISPHERE_BOTTOM', "Bottom Hemisphere", "Cameras below the object"),
            ('RING', "Single Ring", "Cameras in a horizontal ring"),
            ('MULTI_RING', "Multi Ring", "Multiple horizontal rings at different elevations"),
        ],
        default='FIBONACCI'
    )

    ring_count: IntProperty(
        name="Ring Count",
        description="Number of horizontal rings for multi-ring distribution",
        default=5,
        min=2,
        max=20
    )

    min_elevation: FloatProperty(
        name="Min Elevation",
        description="Minimum camera elevation angle (degrees)",
        default=-60,
        min=-90,
        max=90
    )

    max_elevation: FloatProperty(
        name="Max Elevation",
        description="Maximum camera elevation angle (degrees)",
        default=60,
        min=-90,
        max=90
    )

    camera_distance_mode: EnumProperty(
        name="Distance Mode",
        items=[
            ('AUTO', "Auto", "Calculate distance based on object size"),
            ('MANUAL', "Manual", "Use specified distance"),
        ],
        default='AUTO'
    )

    camera_distance: FloatProperty(
        name="Camera Distance",
        description="Distance from object center (manual mode)",
        default=5.0,
        min=0.1,
        max=1000
    )

    camera_distance_multiplier: FloatProperty(
        name="Distance Multiplier",
        description="Multiplier for auto-calculated distance",
        default=2.5,
        min=1.0,
        max=10.0
    )

    focal_length: FloatProperty(
        name="Focal Length",
        description="Camera focal length in mm",
        default=50,
        min=10,
        max=300
    )

    # ==========================================================================
    # RENDER SETTINGS (GS-specific only)
    # ==========================================================================
    # Note: Resolution, samples, engine, and file format are controlled via
    # Blender's native render settings panel. This avoids duplication and
    # ensures settings are always in sync.

    render_speed_preset: EnumProperty(
        name="Render Speed",
        description="Optimize render settings for speed vs quality",
        items=[
            ('CUSTOM', "Custom", "Use current Blender render settings"),
            ('FAST', "Fast (Eevee)", "Eevee with low samples - fastest"),
            ('BALANCED', "Balanced (Eevee)", "Eevee with good quality"),
            ('QUALITY', "Quality (Cycles)", "Cycles with 128 samples"),
        ],
        default='CUSTOM'
    )

    transparent_background: BoolProperty(
        name="Transparent Background",
        description="Render with transparent background (PNG/EXR only)",
        default=True
    )

    # ==========================================================================
    # LIGHTING SETTINGS
    # ==========================================================================

    lighting_mode: EnumProperty(
        name="Lighting Mode",
        items=[
            ('WHITE', "White Background", "Pure white environment"),
            ('GRAY', "Gray Background", "Neutral gray environment"),
            ('HDR', "HDR Environment", "Use HDR image for lighting"),
            ('KEEP', "Keep Scene Lighting", "Don't modify lighting"),
        ],
        default='WHITE'
    )

    background_strength: FloatProperty(
        name="Background Strength",
        description="Brightness of the background",
        default=1.0,
        min=0.1,
        max=10.0
    )

    gray_value: FloatProperty(
        name="Gray Value",
        description="Gray level (0=black, 1=white)",
        default=0.5,
        min=0.0,
        max=1.0
    )

    hdr_path: StringProperty(
        name="HDR Path",
        description="Path to HDR environment image",
        default="",
        subtype='FILE_PATH'
    )

    hdr_strength: FloatProperty(
        name="HDR Strength",
        description="Intensity of HDR lighting",
        default=1.0,
        min=0.1,
        max=10.0
    )

    disable_scene_lights: BoolProperty(
        name="Disable Scene Lights",
        description="Hide existing lights during capture",
        default=True
    )

    # ==========================================================================
    # MATERIAL SETTINGS
    # ==========================================================================

    material_mode: EnumProperty(
        name="Material Mode",
        items=[
            ('ORIGINAL', "Original Materials", "Keep original materials"),
            ('DIFFUSE', "Neutral Diffuse", "Replace with neutral gray diffuse"),
            ('VERTEX_COLOR', "Vertex Colors", "Use vertex colors if available"),
            ('MATCAP', "Matcap", "Use matcap-style shading"),
        ],
        default='ORIGINAL'
    )

    # ==========================================================================
    # BATCH SETTINGS
    # ==========================================================================

    batch_mode: EnumProperty(
        name="Batch Mode",
        items=[
            ('SELECTED', "Selected Objects", "Capture each selected object separately"),
            ('COLLECTION', "Collection", "Capture entire collection as one"),
            ('EACH_SELECTED', "Each Selected", "Capture each selected object individually"),
            ('COLLECTIONS', "All Collections", "Capture each collection separately"),
            ('GROUPS', "Object Groups", "Use custom object groups"),
        ],
        default='SELECTED'
    )

    target_collection: StringProperty(
        name="Target Collection",
        description="Collection to capture"
    )

    include_nested_collections: BoolProperty(
        name="Include Nested",
        description="Include objects from nested collections",
        default=True
    )

    include_children: BoolProperty(
        name="Include Children",
        description="Include child objects of selected objects",
        default=True
    )

    object_groups: CollectionProperty(type=GSCaptureObjectGroup)
    active_group_index: IntProperty(default=0)

    # ==========================================================================
    # ADAPTIVE CAPTURE SETTINGS
    # ==========================================================================

    use_adaptive_capture: BoolProperty(
        name="Use Adaptive Capture",
        description="Automatically adjust settings based on object analysis",
        default=False
    )

    adaptive_quality_preset: EnumProperty(
        name="Quality Preset",
        items=[
            ('AUTO', "Auto", "Automatically determine quality"),
            ('DRAFT', "Draft", "Quick preview quality"),
            ('STANDARD', "Standard", "Good balance of quality and speed"),
            ('HIGH', "High", "High quality"),
            ('ULTRA', "Ultra", "Maximum quality"),
        ],
        default='AUTO'
    )

    adaptive_use_hotspots: BoolProperty(
        name="Use Detail Hotspots",
        description="Bias camera placement toward high-detail areas",
        default=True
    )

    adaptive_hotspot_bias: FloatProperty(
        name="Hotspot Bias",
        description="How much to bias cameras toward detail areas",
        default=0.3,
        min=0.0,
        max=1.0
    )

    # ==========================================================================
    # CHECKPOINT/RESUME SETTINGS
    # ==========================================================================

    enable_checkpoints: BoolProperty(
        name="Enable Checkpoints",
        description="Save progress for resume capability",
        default=True
    )

    checkpoint_interval: IntProperty(
        name="Checkpoint Interval",
        description="Save checkpoint every N images",
        default=10,
        min=1,
        max=100
    )

    auto_resume: BoolProperty(
        name="Auto Resume",
        description="Automatically resume from checkpoint if found",
        default=True
    )

    # ==========================================================================
    # PROGRESS/STATE (Runtime)
    # ==========================================================================

    is_rendering: BoolProperty(default=False)
    render_progress: FloatProperty(default=0.0)
    current_render_info: StringProperty(default="")

    # Analysis results storage
    analysis_vertex_count: IntProperty(default=0)
    analysis_face_count: IntProperty(default=0)
    analysis_surface_area: FloatProperty(default=0.0)
    analysis_detail_score: FloatProperty(default=0.0)
    analysis_texture_resolution: IntProperty(default=0)
    analysis_texture_score: FloatProperty(default=0.0)
    analysis_recommended_cameras: IntProperty(default=100)
    analysis_recommended_resolution: StringProperty(default="1920x1080")
    analysis_quality_preset: StringProperty(default="STANDARD")
    analysis_render_time_estimate: StringProperty(default="")
    analysis_warnings: StringProperty(default="")

    # Checkpoint state
    checkpoint: PointerProperty(type=GSCaptureCheckpoint)

    # ==========================================================================
    # FRAMEWORK PRESET SETTINGS
    # ==========================================================================

    framework_preset: EnumProperty(
        name="Framework Preset",
        description="Target training framework for optimized settings",
        items=[
            ('GAUSSIAN_SPLATTING', "3D Gaussian Splatting", "Original 3DGS by INRIA"),
            ('INSTANT_NGP', "Instant-NGP", "NVIDIA Instant Neural Graphics Primitives"),
            ('NERFSTUDIO', "Nerfstudio", "Nerfstudio framework (splatfacto)"),
            ('POSTSHOT', "Postshot", "Postshot Gaussian Splatting"),
            ('POLYCAM', "Polycam", "Polycam 3D capture format"),
            ('LUMA_AI', "Luma AI", "Luma AI capture format"),
            ('GSPLAT', "gsplat", "Optimized gsplat library"),
        ],
        default='GAUSSIAN_SPLATTING'
    )

    # ==========================================================================
    # TRAINING SETTINGS
    # ==========================================================================

    training_backend: EnumProperty(
        name="Training Backend",
        description="Training framework to use",
        items=[
            ('gaussian_splatting', "3D Gaussian Splatting", "Original 3DGS implementation"),
            ('nerfstudio', "Nerfstudio", "Nerfstudio splatfacto"),
            ('gsplat', "gsplat", "Optimized gsplat library"),
        ],
        default='gaussian_splatting'
    )

    training_data_path: StringProperty(
        name="Training Data",
        description="Path to captured training data",
        default="",
        subtype='DIR_PATH'
    )

    training_output_path: StringProperty(
        name="Training Output",
        description="Path to save trained model",
        default="",
        subtype='DIR_PATH'
    )

    training_iterations: IntProperty(
        name="Iterations",
        description="Number of training iterations",
        default=30000,
        min=1000,
        max=100000
    )

    training_save_every: IntProperty(
        name="Save Every",
        description="Save checkpoint every N iterations (0 = only at end)",
        default=7000,
        min=0,
        max=50000
    )

    training_white_background: BoolProperty(
        name="White Background",
        description="Use white background for training",
        default=True
    )

    training_gpu_id: IntProperty(
        name="GPU ID",
        description="GPU device ID for training",
        default=0,
        min=0,
        max=7
    )

    training_extra_args: StringProperty(
        name="Extra Arguments",
        description="Additional command-line arguments for training",
        default=""
    )

    # Densification settings (3DGS specific)
    densify_from_iter: IntProperty(
        name="Densify From",
        description="Start densification at this iteration",
        default=500,
        min=0,
        max=10000
    )

    densify_until_iter: IntProperty(
        name="Densify Until",
        description="Stop densification at this iteration",
        default=15000,
        min=1000,
        max=50000
    )

    densification_interval: IntProperty(
        name="Densification Interval",
        description="Densify every N iterations",
        default=100,
        min=10,
        max=1000
    )

    # ==========================================================================
    # VIEWPORT VISUALIZATION SETTINGS
    # ==========================================================================

    show_camera_preview: BoolProperty(
        name="Show Camera Preview",
        description="Display camera positions in viewport",
        default=True
    )

    show_coverage_heatmap: BoolProperty(
        name="Show Coverage Heatmap",
        description="Show vertex coverage as heatmap colors",
        default=False
    )

    preview_camera_size: FloatProperty(
        name="Camera Size",
        description="Size of camera preview in viewport",
        default=0.3,
        min=0.1,
        max=2.0
    )

    # ==========================================================================
    # SCENE ANALYSIS STATE (Runtime)
    # ==========================================================================

    # Coverage analysis results
    coverage_analyzed: BoolProperty(
        name="Coverage Analyzed",
        description="Whether coverage has been analyzed",
        default=False
    )

    coverage_percentage: FloatProperty(
        name="Coverage Percentage",
        description="Percentage of vertices with good coverage",
        default=0.0,
        min=0.0,
        max=100.0
    )

    # Scene score results
    scene_score: IntProperty(
        name="Scene Score",
        description="Overall scene suitability score (0-100)",
        default=0,
        min=0,
        max=100
    )

    scene_grade: EnumProperty(
        name="Scene Grade",
        items=[
            ('EXCELLENT', "Excellent", "Easy to capture, will work great"),
            ('GOOD', "Good", "Should work well with default settings"),
            ('FAIR', "Fair", "May need adjustments"),
            ('POOR', "Poor", "Challenging, may have issues"),
            ('NONE', "Not Analyzed", "Scene not yet analyzed"),
        ],
        default='NONE'
    )

    scene_analyzed: BoolProperty(
        name="Scene Analyzed",
        description="Whether scene has been analyzed",
        default=False
    )

    # Material problems count
    material_problems_count: IntProperty(
        name="Material Problems",
        description="Number of material issues detected",
        default=0,
        min=0
    )

    material_problems_high: IntProperty(
        name="High Severity Problems",
        description="Number of high severity material issues",
        default=0,
        min=0
    )
