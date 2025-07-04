How to get a blend model of a new head

step1: get `mean_plus_id.npy` in gene.py (by uncomment the code in `emogene/gene.py`)

step2: use `parse_face.py` to transform the `.npy` to `.obj`

step3: use `refine_face.py` to delete the unneeded keypoints

```bash
// blender /path/to/your.blend --background --python replace_verts.py


emotalk/blender_ver_3_6/blender _test/shapes/feng/may_rigged.blend --background --python _test/shapes/codes/replace_verts.py

```