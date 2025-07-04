'''
replace_verts_preserve_shapekeys.py

Blender Python script to replace vertex positions of a mesh with those from an OBJ file,
while preserving existing Shape Keys.

Usage (run from terminal):
    blender /path/to/your.blend --background --python replace_verts_preserve_shapekeys.py
    
    
    emotalk/blender_ver_3_6/blender _test/shapes/feng/may_rigged.blend --background --python _test/shapes/codes/replace_verts_preserve_shapekeys.py
'''

import bpy
import os
import sys

# —— USER SETTINGS —— #
OBJ_FILE       = "_test/shapes/feng/face_mp468_feng_0.obj"   # Path to your .obj file
TARGET_NAME    = "face_mp468_90"                     # Name of the mesh object in the .blend
SAVE_OVERWRITE = True                          # Overwrite the original .blend if True
# SAVE_OVERWRITE = False                        # Otherwise, saves to *_replaced.blend
# —— END SETTINGS —— #


def read_obj_vertices(path):
    """
    Read vertex positions from an OBJ file.
    Returns a list of (x, y, z) tuples.
    """
    verts = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()[1:4]
                    verts.append(tuple(float(c) for c in parts))
    except Exception as e:
        print(f"Error reading OBJ file: {e}")
        sys.exit(1)
    return verts


def main():
    # 1. Read new vertex positions
    verts = read_obj_vertices(OBJ_FILE)

    # 2. Get target object
    if TARGET_NAME not in bpy.data.objects:
        print(f"Error: Object '{TARGET_NAME}' not found in .blend.")
        sys.exit(1)
    obj = bpy.data.objects[TARGET_NAME]
    mesh = obj.data

    # 3. Ensure vertex count matches
    vert_count = len(verts)
    # If shape keys exist, use Basis key for count; else mesh.vertices
    sk = mesh.shape_keys
    if sk and 'Basis' in sk.key_blocks:
        basis_block = sk.key_blocks['Basis'].data
        if len(basis_block) != vert_count:
            print(f"Error: Vertex count mismatch (OBJ: {vert_count}, Basis: {len(basis_block)})")
            sys.exit(1)
    else:
        if len(mesh.vertices) != vert_count:
            print(f"Error: Vertex count mismatch (OBJ: {vert_count}, Mesh: {len(mesh.vertices)})")
            sys.exit(1)

    # 4. Preserve shape key offsets (if any)
    offsets = {}
    if sk and len(sk.key_blocks) > 1:
        print("Preserving existing Shape Keys...")
        basis_coords = [v.co.copy() for v in sk.key_blocks['Basis'].data]
        # Compute offsets for each non-Basis key
        for key_block in sk.key_blocks:
            if key_block.name == 'Basis':
                continue
            offsets[key_block.name] = [key_block.data[i].co - basis_coords[i]
                                       for i in range(len(basis_coords))]

    # 5. Apply new positions
    if sk and 'Basis' in sk.key_blocks:
        # Write into Basis Shape Key
        basis_kb = sk.key_blocks['Basis'].data
        for i, coord in enumerate(verts):
            basis_kb[i].co = coord
        # Reapply offsets
        for key_name, delta_list in offsets.items():
            kb = sk.key_blocks[key_name].data
            for i, delta in enumerate(delta_list):
                kb[i].co = basis_kb[i].co + delta
    else:
        # No shape keys: direct mesh edit
        for i, coord in enumerate(verts):
            mesh.vertices[i].co = coord

    # 6. Update and save
    mesh.update()
    if SAVE_OVERWRITE:
        bpy.ops.wm.save_mainfile()
        print("Saved and overwritten the original .blend.")
    else:
        base, ext = os.path.splitext(bpy.data.filepath)
        new_path = base + "_replaced" + ext
        bpy.ops.wm.save_as_mainfile(filepath=new_path)
        print(f"Saved to: {new_path}")


if __name__ == '__main__':
    main()
