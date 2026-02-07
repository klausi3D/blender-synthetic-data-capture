"""
Pre-capture Validation System.

Provides comprehensive validation of scene, settings, and output
before starting a capture to prevent common issues.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import bpy


class ValidationLevel(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Capture cannot proceed
    WARNING = "warning"  # Capture can proceed but results may be suboptimal
    INFO = "info"        # Informational message


@dataclass
class ValidationIssue:
    """Represents a single validation issue.

    Attributes:
        level: Severity of the issue
        category: Category for grouping (scene, settings, output, etc.)
        message: Human-readable description of the issue
        suggestion: Recommended fix or action
        auto_fixable: Whether the issue can be automatically fixed
        fix_operator: Operator ID to fix the issue (if auto_fixable)
    """
    level: ValidationLevel
    category: str
    message: str
    suggestion: str = ""
    auto_fixable: bool = False
    fix_operator: str = ""


@dataclass
class ValidationResult:
    """Complete validation result.

    Attributes:
        issues: List of all validation issues found
        can_proceed: Whether capture can proceed (no ERROR level issues)
    """
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def can_proceed(self) -> bool:
        """Check if capture can proceed (no errors)."""
        return not any(i.level == ValidationLevel.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)

    def add_error(self, category: str, message: str, suggestion: str = "",
                  auto_fixable: bool = False, fix_operator: str = "") -> None:
        """Add an error issue."""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            category=category,
            message=message,
            suggestion=suggestion,
            auto_fixable=auto_fixable,
            fix_operator=fix_operator
        ))

    def add_warning(self, category: str, message: str, suggestion: str = "",
                    auto_fixable: bool = False, fix_operator: str = "") -> None:
        """Add a warning issue."""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.WARNING,
            category=category,
            message=message,
            suggestion=suggestion,
            auto_fixable=auto_fixable,
            fix_operator=fix_operator
        ))

    def add_info(self, category: str, message: str) -> None:
        """Add an informational issue."""
        self.issues.append(ValidationIssue(
            level=ValidationLevel.INFO,
            category=category,
            message=message
        ))


class SceneValidator:
    """Validates scene configuration for capture.

    Checks for:
    - Target objects exist and are valid
    - Object visibility and renderability
    - Mesh geometry issues
    - Material configuration
    """

    def validate(self, context, settings) -> ValidationResult:
        """Validate scene configuration.

        Args:
            context: Blender context
            settings: GS Capture settings

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()

        # Get target objects
        target_objects = self._get_target_objects(context, settings)

        if not target_objects:
            result.add_error(
                "scene",
                "No objects to capture",
                "Select objects or set target collection"
            )
            return result

        # Check each object
        for obj in target_objects:
            self._validate_object(obj, result)

        # Check for hidden objects
        hidden = [obj for obj in target_objects if obj.hide_render]
        if hidden:
            result.add_warning(
                "scene",
                f"{len(hidden)} object(s) hidden from render",
                "Enable 'Render' visibility in Outliner"
            )

        return result

    def _get_target_objects(self, context, settings):
        """Get list of target objects for capture."""
        if settings.target_collection:
            collection = bpy.data.collections.get(settings.target_collection)
            if collection:
                return [obj for obj in collection.all_objects
                        if obj.type == 'MESH']

        return [obj for obj in context.selected_objects
                if obj.type == 'MESH']

    def _validate_object(self, obj, result: ValidationResult) -> None:
        """Validate a single object."""
        # Check mesh data exists
        if not obj.data:
            result.add_error(
                "scene",
                f"'{obj.name}' has no mesh data",
                "Remove or replace the object"
            )
            return

        mesh = obj.data

        # Check for geometry
        if len(mesh.vertices) == 0:
            result.add_error(
                "scene",
                f"'{obj.name}' has no vertices",
                "Remove or add geometry to the object"
            )
            return

        if len(mesh.polygons) == 0:
            result.add_warning(
                "scene",
                f"'{obj.name}' has no faces (vertices only)",
                "This may not render correctly"
            )

        # Check for non-manifold geometry
        if mesh.has_loose_vertices:
            result.add_warning(
                "scene",
                f"'{obj.name}' has loose vertices",
                "Clean up mesh in Edit Mode"
            )

        # Check scale
        scale = obj.scale
        if abs(scale.x - 1.0) > 0.01 or abs(scale.y - 1.0) > 0.01 or abs(scale.z - 1.0) > 0.01:
            result.add_info(
                "scene",
                f"'{obj.name}' has non-uniform scale ({scale.x:.2f}, {scale.y:.2f}, {scale.z:.2f})"
            )


class SettingsValidator:
    """Validates capture settings configuration.

    Checks for:
    - Camera count is reasonable
    - Resolution is appropriate
    - Export paths are valid
    - Format compatibility
    """

    def validate(self, context, settings) -> ValidationResult:
        """Validate capture settings.

        Args:
            context: Blender context
            settings: GS Capture settings

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()
        scene = context.scene

        # Camera count
        if settings.camera_count < 10:
            result.add_warning(
                "settings",
                f"Low camera count ({settings.camera_count})",
                "Recommend at least 20-30 cameras for good results"
            )
        elif settings.camera_count > 500:
            result.add_warning(
                "settings",
                f"Very high camera count ({settings.camera_count})",
                "This will take a long time to capture and train"
            )

        # Resolution
        res_x = scene.render.resolution_x
        res_y = scene.render.resolution_y

        if res_x < 512 or res_y < 512:
            result.add_warning(
                "settings",
                f"Low resolution ({res_x}x{res_y})",
                "Recommend at least 800x800 for good quality"
            )

        if res_x > 4096 or res_y > 4096:
            result.add_warning(
                "settings",
                f"Very high resolution ({res_x}x{res_y})",
                "This will require significant VRAM for training"
            )

        # Samples (for Cycles)
        if scene.render.engine == 'CYCLES':
            if scene.cycles.samples < 64:
                result.add_warning(
                    "settings",
                    f"Low sample count ({scene.cycles.samples})",
                    "May result in noisy renders"
                )

        # File format
        file_format = scene.render.image_settings.file_format
        if file_format not in ('PNG', 'JPEG', 'OPEN_EXR'):
            result.add_warning(
                "settings",
                f"Uncommon file format: {file_format}",
                "PNG or JPEG recommended for compatibility"
            )

        return result


class OutputValidator:
    """Validates output path configuration.

    Checks for:
    - Output directory exists or can be created
    - Write permissions
    - Disk space availability
    - Existing files that might be overwritten
    """

    def validate(self, context, settings) -> ValidationResult:
        """Validate output configuration.

        Args:
            context: Blender context
            settings: GS Capture settings

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()

        output_path = settings.output_path

        # Check path is set
        if not output_path:
            result.add_error(
                "output",
                "No output path specified",
                "Set an output directory for captured images"
            )
            return result

        # Normalize path
        output_path = bpy.path.abspath(output_path)

        # Check if path is valid
        try:
            os.makedirs(output_path, exist_ok=True)
        except PermissionError:
            result.add_error(
                "output",
                "Cannot create output directory",
                "Check folder permissions"
            )
            return result
        except Exception as e:
            result.add_error(
                "output",
                f"Invalid output path: {e}",
                "Choose a different location"
            )
            return result

        # Check for existing files
        images_dir = os.path.join(output_path, "images")
        if os.path.exists(images_dir):
            existing = os.listdir(images_dir)
            if existing:
                result.add_warning(
                    "output",
                    f"Output directory contains {len(existing)} existing files",
                    "Files may be overwritten"
                )

        # Check disk space (rough estimate)
        try:
            import shutil
            total, used, free = shutil.disk_usage(output_path)
            free_gb = free / (1024 ** 3)

            # Estimate required space based on enabled outputs
            estimate = estimate_capture_size(settings, context)
            estimated_size_gb = estimate['total_gb']

            if free_gb < 1.0:
                result.add_warning(
                    "output",
                    f"Low disk space: {free_gb:.1f} GB free",
                    "Free up space before capture"
                )
            elif estimated_size_gb > free_gb * 0.8:
                result.add_warning(
                    "output",
                    f"Capture may require ~{estimated_size_gb:.1f} GB",
                    f"Only {free_gb:.1f} GB available"
                )
        except Exception:
            pass  # Skip disk space check if it fails

        return result


class CoverageValidator:
    """Validates camera coverage of target objects.

    Checks for:
    - Adequate view coverage
    - Occluded areas
    - Camera distance appropriateness
    """

    def validate(self, context, settings, cameras=None) -> ValidationResult:
        """Validate camera coverage.

        Args:
            context: Blender context
            settings: GS Capture settings
            cameras: List of camera objects (optional)

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()

        if not cameras:
            result.add_info(
                "coverage",
                "Coverage validation requires generated cameras"
            )
            return result

        # Check camera count
        if len(cameras) < 10:
            result.add_warning(
                "coverage",
                f"Only {len(cameras)} cameras generated",
                "Increase camera count for better coverage"
            )

        # TODO: Implement detailed coverage analysis using utils/coverage.py
        # For now, just do basic validation

        return result


def _get_preview_cameras():
    """Return preview cameras created by the GS preview operator."""
    return [
        obj for obj in bpy.data.objects
        if obj.type == 'CAMERA' and obj.name.startswith('GS_Cam_')
    ]


def validate_all(context, settings, cameras=None) -> ValidationResult:
    """Run all validators and combine results.

    This is the main entry point for validation.

    Args:
        context: Blender context
        settings: GS Capture settings
        cameras: List of camera objects (optional)

    Returns:
        Combined ValidationResult from all validators
    """
    combined = ValidationResult()

    if cameras is None:
        cameras = _get_preview_cameras()

    # Run each validator
    validators = [
        ("Scene", SceneValidator()),
        ("Settings", SettingsValidator()),
        ("Output", OutputValidator()),
    ]

    if cameras:
        validators.append(("Coverage", CoverageValidator()))

    for name, validator in validators:
        if hasattr(validator, 'validate'):
            if name == "Coverage":
                result = validator.validate(context, settings, cameras)
            else:
                result = validator.validate(context, settings)

            combined.issues.extend(result.issues)

    return combined


def quick_validate(context, settings) -> Tuple[bool, str]:
    """Quick validation check for operators.

    Returns a simple pass/fail with message.

    Args:
        context: Blender context
        settings: GS Capture settings

    Returns:
        Tuple of (can_proceed, error_message)
    """
    result = validate_all(context, settings)

    if result.can_proceed:
        if result.warning_count > 0:
            return True, f"{result.warning_count} warnings"
        return True, ""
    else:
        # Get first error message
        for issue in result.issues:
            if issue.level == ValidationLevel.ERROR:
                return False, issue.message

        return False, "Validation failed"


def estimate_capture_size(settings, context) -> dict:
    """
    Estimate total capture size in MB.

    Returns dict with:
    - images_mb: RGB image size
    - depth_mb: Depth maps (if enabled)
    - normals_mb: Normal maps (if enabled)
    - total_mb: Total estimate
    - total_gb: Total in GB
    - warning: String if size is large
    """
    rd = context.scene.render
    width = rd.resolution_x
    height = rd.resolution_y
    num_cameras = settings.camera_count

    # Base pixel count
    pixels = width * height

    # PNG compression ratio (approximate)
    png_ratio = 0.4  # PNG compresses well for rendered images

    # RGB images: 3 bytes per pixel, with compression
    rgb_bytes = pixels * 3 * png_ratio
    images_mb = (rgb_bytes * num_cameras) / (1024 * 1024)

    # Depth maps (16-bit grayscale)
    depth_mb = 0
    if getattr(settings, 'export_depth', False):
        depth_bytes = pixels * 2 * 0.5  # 16-bit with compression
        depth_mb = (depth_bytes * num_cameras) / (1024 * 1024)

    # Normal maps (if enabled) - EXR is larger
    normals_mb = 0
    if getattr(settings, 'export_normals', False):
        normal_bytes = pixels * 6  # 16-bit float * 3 channels, minimal compression
        normals_mb = (normal_bytes * num_cameras) / (1024 * 1024)

    # Masks (if enabled) - single-channel PNG
    masks_mb = 0
    if getattr(settings, 'export_masks', False):
        mask_bytes = pixels * 1 * png_ratio
        masks_mb = (mask_bytes * num_cameras) / (1024 * 1024)

    total_mb = images_mb + depth_mb + normals_mb + masks_mb + 10  # +10MB for metadata
    total_gb = total_mb / 1024

    warning = None
    if total_gb > 50:
        warning = f"Very large capture: {total_gb:.1f} GB - consider reducing resolution or camera count"
    elif total_gb > 10:
        warning = f"Large capture: {total_gb:.1f} GB - ensure sufficient disk space"

    return {
        'images_mb': round(images_mb, 1),
        'depth_mb': round(depth_mb, 1),
        'normals_mb': round(normals_mb, 1),
        'masks_mb': round(masks_mb, 1),
        'total_mb': round(total_mb, 1),
        'total_gb': round(total_gb, 2),
        'warning': warning
    }
