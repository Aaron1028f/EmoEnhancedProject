# transform .npy to .obj

import numpy as np

def save_obj(vertices, faces=None, filename="output.obj"):
    """
    將 3D 頂點數據和面數據（可選）保存為 .obj 文件格式。

    :param vertices: numpy.ndarray，形狀為 (N, 3)，包含每個頂點的 (x, y, z) 坐標。
    :param faces: numpy.ndarray（可選），形狀為 (M, 3)，包含每個面的頂點索引。
    :param filename: str，輸出的 .obj 文件名稱。
    """
    with open(filename, 'w') as file:
        # 寫入頂點數據，每行使用 `v x y z` 格式
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # 如果有面信息，則寫入面數據，每行使用 `f i j k` 格式
        if faces is not None:
            for face in faces:
                # .obj 文件中的索引從 1 開始
                # file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
                file.write(f"f {face[0]} {face[1]} {face[2]}\n")

# # 示例使用
# # 假設有頂點數據和面數據
# vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# faces = np.array([[0, 1, 2]])  # 假設一個三角形面
# save_obj(vertices, faces, "output.obj")




if __name__ == "__main__":
    
    face = np.load("_test/shapes/feng/mean_plus_id_feng.npy")

    # face = np.load("GeneFacePlusPlus-main/_test/shapes/mean_plus_id.npy")

    print(face.shape)

    face = face / 10

    save_obj(face, face, "_test/shapes/feng/face_mp468_feng_0.obj")
