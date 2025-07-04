import bpy
import os
import numpy as np
import sys

filename = str(sys.argv[-1])
root_dir = str(sys.argv[-2])

bs52_level = float(sys.argv[-3])
lm468_bs_np_path = str(sys.argv[-4])
output_video = bool(sys.argv[-5])
output_video = False

model_bsList = ["browDownLeft",
                "browDownRight",
                "browInnerUp",
                "browOuterUpLeft",
                "browOuterUpRight",
                "cheekPuff",
                "cheekSquintLeft",
                "cheekSquintRight",
                "eyeBlinkLeft",
                "eyeBlinkRight",
                "eyeLookDownLeft",
                "eyeLookDownRight",
                "eyeLookInLeft",
                "eyeLookInRight",
                "eyeLookOutLeft",
                "eyeLookOutRight",
                "eyeLookUpLeft",
                "eyeLookUpRight",
                "eyeSquintLeft",
                "eyeSquintRight",
                "eyeWideLeft",
                "eyeWideRight",
                "jawForward",
                "jawLeft",
                "jawOpen",
                "jawRight",
                "mouthClose",
                "mouthDimpleLeft",
                "mouthDimpleRight",
                "mouthFrownLeft",
                "mouthFrownRight",
                "mouthFunnel",
                "mouthLeft",
                "mouthLowerDownLeft",
                "mouthLowerDownRight",
                "mouthPressLeft",
                "mouthPressRight",
                "mouthPucker",
                "mouthRight",
                "mouthRollLower",
                "mouthRollUpper",
                "mouthShrugLower",
                "mouthShrugUpper",
                "mouthSmileLeft",
                "mouthSmileRight",
                "mouthStretchLeft",
                "mouthStretchRight",
                "mouthUpperUpLeft",
                "mouthUpperUpRight",
                "noseSneerLeft",
                "noseSneerRight",
                "tongueOut"]

# ================== Load the blend file ==================
# when use original face obj

# obj = bpy.data.objects["face"]
# bs_face = bpy.data.objects["testing_1_out_tri"]
# bs_face.hide_render = True

# ================== Load the blend file ==================
# use lm468_bs
temp = bpy.data.objects["face"]
temp.hide_render = True
# obj = bpy.data.objects["face_mp468_2"]
# obj = bpy.data.objects["face_mp468_90"]
obj = bpy.data.objects["face_mp468_90"]

# ================== Set up the scene ==================
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'MATCAP'
bpy.context.scene.display.render_aa = 'FXAA'
bpy.context.scene.render.resolution_x = int(512)
bpy.context.scene.render.resolution_y = int(768)
bpy.context.scene.render.fps = 30
bpy.context.scene.render.image_settings.file_format = 'PNG'

cam = bpy.data.objects['Camera']
cam.scale = [2, 2, 2]   
bpy.context.scene.camera = cam

output_dir = root_dir + filename
blendshape_path = root_dir + filename + '.npy'

bs = np.load(blendshape_path) # bs has shape (num_frames, 52)

# 獲取網格頂點數量
depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = obj.evaluated_get(depsgraph)
mesh = obj_eval.data
num_vertices = len(mesh.vertices)

# # set the number of frames for generating the lm468bs
# frame_num = bs.shape[0] * 5 // 6
# frame_counter = 0
frame_num = bs.shape[0]

lm468bs = np.zeros([frame_num, num_vertices, 3], dtype=np.float32)

for i in range(frame_num):
    curr_bs = bs[i]
    for j in range(52):
        # if obj.data.shape_keys.key_blocks[model_bsList[j]] == "mouthClose":
        #     obj.data.shape_keys.key_blocks[model_bsList[j]].value = curr_bs[j]
        # else:
        obj.data.shape_keys.key_blocks[model_bsList[j]].value = bs52_level * curr_bs[j]
        # if obj.data.shape_keys.key_blocks[model_bsList[j]].value >= 1:
        #     obj.data.shape_keys.key_blocks[model_bsList[j]].value = 1
            
    # ===========================================================================
    # 獲取當前 frame 的變形後頂點座標
    # if (i + 1) % 6 == 0: # make it 25fps instead of 30fps
    #     pass
    # elif frame_counter < frame_num:
    #     obj_eval = obj.evaluated_get(depsgraph)  # 獲取計算後的物件
    #     mesh = obj_eval.data  # 獲取計算後的網格數據

    #     for vertex_idx, vertex in enumerate(mesh.vertices):
    #         position = vertex.co  # 當前頂點的位置 (Vector)
    #         lm468bs[frame_counter, vertex_idx] = [position.x, position.y, position.z]
    #     frame_counter += 1
    # else: 
    #     pass
    
    obj_eval = obj.evaluated_get(depsgraph)  # 獲取計算後的物件
    mesh = obj_eval.data  # 獲取計算後的網格數據

    for vertex_idx, vertex in enumerate(mesh.vertices):
        position = vertex.co  # 當前頂點的位置 (Vector)
        lm468bs[i, vertex_idx] = [position.x, position.y, position.z]

    # ===========================================================================
    # # render video if needed
    if output_video:
        bpy.context.scene.render.filepath = os.path.join(output_dir, '{}.png'.format(i))
        bpy.ops.render.render(write_still=True)
    else:
        # update the scene
        bpy.context.view_layer.update()

# save the lm468bs
out = np.save(lm468_bs_np_path, lm468bs)
# print(out)
# print(lm468bs.shape)
# print('='*50)


