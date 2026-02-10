"""
Extract Geometry Node tree structure from a blend file.
Run this with Blender: blender --background --python tools/extract_nodes.py
"""

import bpy
import json
import sys

def extract_node_tree(node_tree, depth=0):
    """Recursively extract node tree structure."""
    result = {
        'name': node_tree.name,
        'type': node_tree.bl_idname,
        'nodes': [],
        'links': []
    }
    
    for node in node_tree.nodes:
        node_info = {
            'name': node.name,
            'type': node.bl_idname,
            'label': node.label if node.label else node.name,
            'inputs': [],
            'outputs': []
        }
        
        # Get input sockets
        for inp in node.inputs:
            socket_info = {
                'name': inp.name,
                'type': inp.bl_idname,
            }
            # Try to get default value
            if hasattr(inp, 'default_value'):
                try:
                    val = inp.default_value
                    if hasattr(val, '__iter__') and not isinstance(val, str):
                        socket_info['default'] = list(val)[:4]  # Truncate vectors
                    else:
                        socket_info['default'] = val
                except:
                    pass
            node_info['inputs'].append(socket_info)
        
        # Get output sockets  
        for out in node.outputs:
            socket_info = {
                'name': out.name,
                'type': out.bl_idname,
            }
            node_info['outputs'].append(socket_info)
        
        # Check for nested node groups
        if node.bl_idname == 'GeometryNodeGroup' and node.node_tree:
            node_info['node_tree'] = extract_node_tree(node.node_tree, depth+1)
        
        result['nodes'].append(node_info)
    
    # Get links
    for link in node_tree.links:
        result['links'].append({
            'from_node': link.from_node.name,
            'from_socket': link.from_socket.name,
            'to_node': link.to_node.name,
            'to_socket': link.to_socket.name
        })
    
    return result

def main():
    # Open the blend file
    blend_path = "C:/Projects/GS_Blender/Blender-3DGS-4DGS-Viewer-Node/Blender-GSViewer-Node.blend"
    
    # Load the file
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    
    print("\n" + "="*80)
    print("BLEND FILE ANALYSIS: Gaussian Splatting Viewer Node")
    print("="*80)
    
    # List all node trees
    print("\n## Node Trees Found:")
    for nt in bpy.data.node_groups:
        print(f"  - {nt.name} ({nt.bl_idname})")
    
    # List all materials
    print("\n## Materials Found:")
    for mat in bpy.data.materials:
        print(f"  - {mat.name}")
        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if 'shader' in node.bl_idname.lower() or 'bsdf' in node.bl_idname.lower():
                    print(f"    -> {node.name} ({node.bl_idname})")
    
    # Extract geometry node trees in detail
    print("\n## Geometry Node Tree Details:")
    for nt in bpy.data.node_groups:
        if 'Geometry' in nt.bl_idname:
            print(f"\n### {nt.name}")
            print(f"Type: {nt.bl_idname}")
            print(f"Node count: {len(nt.nodes)}")
            
            # Categorize nodes
            categories = {}
            for node in nt.nodes:
                cat = node.bl_idname.split('Node')[-1] if 'Node' in node.bl_idname else 'Other'
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(node.name)
            
            print("\nNodes by category:")
            for cat, nodes in sorted(categories.items()):
                print(f"  {cat}: {len(nodes)} nodes")
                for n in nodes[:5]:  # Show first 5
                    print(f"    - {n}")
                if len(nodes) > 5:
                    print(f"    ... and {len(nodes)-5} more")
            
            # Look for key nodes related to Gaussian splatting
            print("\nKey nodes for Gaussian rendering:")
            keywords = ['scale', 'rotation', 'position', 'color', 'opacity', 'sh', 'spherical', 'gaussian', 'splat', 'camera', 'view', 'project']
            for node in nt.nodes:
                name_lower = node.name.lower()
                if any(kw in name_lower for kw in keywords):
                    print(f"  * {node.name} ({node.bl_idname})")
    
    # Check for custom shaders
    print("\n## Shader Analysis:")
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            print(f"\nMaterial: {mat.name}")
            for node in mat.node_tree.nodes:
                print(f"  - {node.name} ({node.bl_idname})")
                
                # Look for OSL or custom scripts
                if 'Script' in node.bl_idname:
                    print(f"    [CUSTOM SHADER DETECTED]")
                    if hasattr(node, 'script') and node.script:
                        print(f"    Script: {node.script.name}")

    # Export full node tree as JSON for detailed analysis
    output_data = {}
    for nt in bpy.data.node_groups:
        output_data[nt.name] = extract_node_tree(nt)
    
    with open("C:/Projects/GS_Blender/node_tree_dump.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print("\n\nFull node tree exported to: node_tree_dump.json")

if __name__ == "__main__":
    main()
