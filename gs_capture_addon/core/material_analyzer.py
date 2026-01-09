"""
Material Problem Detector for Gaussian Splatting.

Analyzes materials in selected objects and warns about properties
that cause issues with 3DGS/NeRF training (transparency, reflections, etc).
"""

import bpy
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class ProblemSeverity(Enum):
    """Severity level for material problems."""
    HIGH = "high"      # Will likely cause major artifacts
    MEDIUM = "medium"  # May cause visible issues
    LOW = "low"        # Minor concern


@dataclass
class MaterialProblem:
    """Represents a single material problem."""
    material_name: str
    object_name: str
    problem_type: str
    severity: ProblemSeverity
    description: str
    suggestion: str


# Problem type definitions
PROBLEM_TYPES = {
    'TRANSPARENT': {
        'icon': 'MOD_OPACITY',
        'label': 'Transparent',
        'description': 'Transparency is not supported by 3DGS',
        'suggestion': 'Convert to opaque or remove transparent parts',
    },
    'REFLECTIVE': {
        'icon': 'MATERIAL',
        'label': 'Reflective',
        'description': 'Reflective surfaces cause view-dependent artifacts',
        'suggestion': 'Reduce metallic/specular or bake reflections',
    },
    'GLASS': {
        'icon': 'CUBE',
        'label': 'Glass/Refraction',
        'description': 'Refraction cannot be captured by 3DGS',
        'suggestion': 'Make opaque or remove glass objects',
    },
    'EMISSIVE': {
        'icon': 'LIGHT',
        'label': 'Emissive',
        'description': 'Bright emission may cause color bleeding',
        'suggestion': 'Reduce emission strength or bake lighting',
    },
    'SUBSURFACE': {
        'icon': 'OUTLINER_OB_ARMATURE',
        'label': 'Subsurface',
        'description': 'SSS effects are view-dependent',
        'suggestion': 'Disable subsurface scattering',
    },
}


def analyze_material(mat, obj_name: str) -> List[MaterialProblem]:
    """Analyze a single material for GS-incompatible properties.

    Args:
        mat: Blender material
        obj_name: Name of object using this material

    Returns:
        List of MaterialProblem instances
    """
    problems = []

    if not mat:
        return problems

    # Check blend method (transparency)
    if mat.blend_method in ('BLEND', 'HASHED', 'CLIP'):
        problems.append(MaterialProblem(
            material_name=mat.name,
            object_name=obj_name,
            problem_type='TRANSPARENT',
            severity=ProblemSeverity.HIGH,
            description=PROBLEM_TYPES['TRANSPARENT']['description'],
            suggestion=PROBLEM_TYPES['TRANSPARENT']['suggestion'],
        ))

    # Check node tree for specific properties
    if mat.use_nodes and mat.node_tree:
        principled = _find_principled_bsdf(mat.node_tree)

        if principled:
            # Check metallic (reflective)
            metallic = _get_socket_value(principled, 'Metallic', 0.0)
            if metallic > 0.5:
                problems.append(MaterialProblem(
                    material_name=mat.name,
                    object_name=obj_name,
                    problem_type='REFLECTIVE',
                    severity=ProblemSeverity.MEDIUM if metallic < 0.8 else ProblemSeverity.HIGH,
                    description=f"Metallic value: {metallic:.1%}",
                    suggestion=PROBLEM_TYPES['REFLECTIVE']['suggestion'],
                ))

            # Check transmission (glass)
            transmission = _get_socket_value(principled, 'Transmission', 0.0)
            # Also check "Transmission Weight" for newer Blender versions
            if transmission < 0.01:
                transmission = _get_socket_value(principled, 'Transmission Weight', 0.0)

            if transmission > 0.1:
                problems.append(MaterialProblem(
                    material_name=mat.name,
                    object_name=obj_name,
                    problem_type='GLASS',
                    severity=ProblemSeverity.HIGH,
                    description=f"Transmission: {transmission:.1%}",
                    suggestion=PROBLEM_TYPES['GLASS']['suggestion'],
                ))

            # Check emission
            emission_strength = _get_socket_value(principled, 'Emission Strength', 0.0)
            if emission_strength > 1.0:
                problems.append(MaterialProblem(
                    material_name=mat.name,
                    object_name=obj_name,
                    problem_type='EMISSIVE',
                    severity=ProblemSeverity.LOW if emission_strength < 5.0 else ProblemSeverity.MEDIUM,
                    description=f"Emission strength: {emission_strength:.1f}",
                    suggestion=PROBLEM_TYPES['EMISSIVE']['suggestion'],
                ))

            # Check subsurface
            subsurface = _get_socket_value(principled, 'Subsurface', 0.0)
            # Also check "Subsurface Weight" for newer Blender versions
            if subsurface < 0.01:
                subsurface = _get_socket_value(principled, 'Subsurface Weight', 0.0)

            if subsurface > 0.1:
                problems.append(MaterialProblem(
                    material_name=mat.name,
                    object_name=obj_name,
                    problem_type='SUBSURFACE',
                    severity=ProblemSeverity.LOW,
                    description=f"Subsurface: {subsurface:.1%}",
                    suggestion=PROBLEM_TYPES['SUBSURFACE']['suggestion'],
                ))

            # Check specular/roughness for high reflectivity
            specular = _get_socket_value(principled, 'Specular IOR Level', None)
            if specular is None:
                specular = _get_socket_value(principled, 'Specular', 0.5)
            if specular is None:
                specular = 0.5
            roughness = _get_socket_value(principled, 'Roughness', 0.5)

            if specular > 0.8 and roughness < 0.2:
                # Only add if not already flagged as metallic
                if not any(p.problem_type == 'REFLECTIVE' for p in problems):
                    problems.append(MaterialProblem(
                        material_name=mat.name,
                        object_name=obj_name,
                        problem_type='REFLECTIVE',
                        severity=ProblemSeverity.MEDIUM,
                        description=f"High specular ({specular:.1%}), low roughness ({roughness:.1%})",
                        suggestion=PROBLEM_TYPES['REFLECTIVE']['suggestion'],
                    ))

    return problems


def analyze_objects(objects) -> List[MaterialProblem]:
    """Analyze all materials on given objects.

    Args:
        objects: List of Blender objects

    Returns:
        List of all MaterialProblem instances found
    """
    all_problems = []
    analyzed_materials = set()  # Avoid duplicate analysis

    for obj in objects:
        if obj.type != 'MESH':
            continue

        for slot in obj.material_slots:
            mat = slot.material
            if not mat or mat.name in analyzed_materials:
                continue

            analyzed_materials.add(mat.name)
            problems = analyze_material(mat, obj.name)
            all_problems.extend(problems)

    return all_problems


def get_problem_summary(problems: List[MaterialProblem]) -> dict:
    """Get summary statistics of problems.

    Args:
        problems: List of MaterialProblem instances

    Returns:
        Dictionary with counts by type and severity
    """
    summary = {
        'total': len(problems),
        'by_severity': {
            'high': 0,
            'medium': 0,
            'low': 0,
        },
        'by_type': {},
        'affected_materials': set(),
        'affected_objects': set(),
    }

    for problem in problems:
        summary['by_severity'][problem.severity.value] += 1

        if problem.problem_type not in summary['by_type']:
            summary['by_type'][problem.problem_type] = 0
        summary['by_type'][problem.problem_type] += 1

        summary['affected_materials'].add(problem.material_name)
        summary['affected_objects'].add(problem.object_name)

    summary['affected_materials'] = len(summary['affected_materials'])
    summary['affected_objects'] = len(summary['affected_objects'])

    return summary


def _find_principled_bsdf(node_tree):
    """Find Principled BSDF node in node tree."""
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def _get_linked_socket_value(socket, default=0.0):
    """Attempt to read a constant value from a linked socket."""
    if not socket.links:
        return default

    from_socket = socket.links[0].from_socket
    if hasattr(from_socket, 'default_value'):
        val = from_socket.default_value
        if hasattr(val, '__iter__') and not isinstance(val, str):
            return val[0] if len(val) > 0 else default
        return val

    return default


def _get_socket_value(node, socket_name: str, default=0.0):
    """Get value from node socket, handling both connected and unconnected cases."""
    if socket_name not in node.inputs:
        return default

    socket = node.inputs[socket_name]

    # If socket is connected, try to read a constant value from the link.
    if socket.is_linked:
        return _get_linked_socket_value(socket, default)

    # Get default value
    if hasattr(socket, 'default_value'):
        val = socket.default_value
        # Handle color/vector types
        if hasattr(val, '__iter__') and not isinstance(val, str):
            return val[0] if len(val) > 0 else default
        return val

    return default


def fix_material_transparency(mat) -> bool:
    """Attempt to fix transparency issues in a material.

    Args:
        mat: Blender material

    Returns:
        True if fixed successfully
    """
    if mat.blend_method in ('BLEND', 'HASHED', 'CLIP'):
        mat.blend_method = 'OPAQUE'
        return True
    return False


def fix_all_problems(objects) -> Tuple[int, int]:
    """Attempt to automatically fix all detected problems.

    Args:
        objects: List of Blender objects

    Returns:
        Tuple of (fixed_count, unfixable_count)
    """
    fixed = 0
    unfixable = 0

    for obj in objects:
        if obj.type != 'MESH':
            continue

        for slot in obj.material_slots:
            mat = slot.material
            if not mat:
                continue

            # Fix transparency
            if mat.blend_method in ('BLEND', 'HASHED', 'CLIP'):
                mat.blend_method = 'OPAQUE'
                fixed += 1

            # Can't easily fix node-based issues automatically
            if mat.use_nodes and mat.node_tree:
                principled = _find_principled_bsdf(mat.node_tree)
                if principled:
                    # These would require more complex fixes
                    transmission = _get_socket_value(principled, 'Transmission', 0.0)
                    if transmission > 0.1:
                        unfixable += 1

    return fixed, unfixable
