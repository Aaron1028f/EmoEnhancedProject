import bpy
import os

# —— 使用者設定區 —— #
# OBJ_FILE      = "/full/path/to/model.obj"    # 你的 .obj 完整路徑
# TARGET_NAME   = "MyMesh"                     # Blend 裡 Mesh 物件的名稱
# SAVE_OVERWRITE = True                        # 是否覆寫原 .blend

OBJ_FILE      = "_test/shapes/feng/face_mp468_feng_0.obj"    # 你的 .obj 完整路徑
TARGET_NAME   = "face_mp468_90"                     # Blend 裡 Mesh 物件的名稱
SAVE_OVERWRITE = True                        # 是否覆寫原 .blend

# 1. 解析 .obj 裡的頂點
verts = []
with open(OBJ_FILE, 'r') as f:
    for line in f:
        if line.startswith('v '):
            # v x y z
            _, x, y, z = line.split(None, 3)
            verts.append((float(x), float(y), float(z)))

# 2. 取得 Blend 中的目標物件
if TARGET_NAME not in bpy.data.objects:
    raise KeyError(f"找不到名為 '{TARGET_NAME}' 的物件")
obj = bpy.data.objects[TARGET_NAME]
mesh = obj.data

# 3. 確認頂點數吻合
if len(mesh.vertices) != len(verts):
    raise ValueError(f"頂點數不符：Blend 中有 {len(mesh.vertices)}，.obj 中有 {len(verts)}")

# 4. 寫回座標
for i, v in enumerate(mesh.vertices):
    v.co = verts[i]

# 5. 更新並儲存
mesh.update()
if SAVE_OVERWRITE:
    # 直接覆寫當前開啟的 .blend
    bpy.ops.wm.save_mainfile()
else:
    # 或另存新檔：
    base, ext = os.path.splitext(bpy.data.filepath)
    bpy.ops.wm.save_as_mainfile(filepath=base + "_replaced" + ext)
