import sys
sys.path.append('./')
import torch
import numpy as np
from emogene.tools.face3d_helper_bs import Face3DHelper
from emogene.tools.face_landmarker_bs import index_lm68_from_lm478
face3d_helper = Face3DHelper(keypoint_mode='mediapipe', use_gpu=True)
mean_face = face3d_helper.key_mean_shape[index_lm68_from_lm478].squeeze().reshape([1, -1, 3]) # [1, 68, 3]

def calculate_mouth_openness(lm3d, return_distance_only=False):
    """
    This is just simple solution.
    Return a list of frames which have mouth openness less than MOUTH_OPEN_THRESHOLD.
    Args:
        lm3d (_type_): _description_
    """
    MOUTH_OPEN_THRESHOLD = 0.04  # threshold for mouth openness
    
    lm3d = lm3d.reshape([-1, 68, 3])  # [T, 68, 3]

    face_real_lm = lm3d/10 + mean_face # [T, 68, 3]
    # mean_face_tensor = torch.from_numpy(mean_face).to(lm3d.device)
    # face_real_lm = lm3d/10 + mean_face_tensor # [T, 68, 3]
    upper_lip = face_real_lm[:, 62, :]  # [T, 3]
    lower_lip = face_real_lm[:, 66, :]  # [T, 3]
    mouth_openness = torch.norm(upper_lip - lower_lip, dim=-1)  # [T]
    
    if return_distance_only:
        return mouth_openness
    
    # print frames of mouth openness less than MOUTH_OPEN_THRESHOLD
    print('-' * 50)
    print(f"Number of frames with mouth openness less than {MOUTH_OPEN_THRESHOLD}: {torch.sum(mouth_openness < MOUTH_OPEN_THRESHOLD).item()}")
    print(f"Frames with mouth openness less than {MOUTH_OPEN_THRESHOLD}: {torch.where(mouth_openness < MOUTH_OPEN_THRESHOLD)[0].tolist()}")
    # for i in torch.where(mouth_openness < MOUTH_OPEN_THRESHOLD)[0]:
    #     print(f"Frame {i.item()}: Mouth openness = {mouth_openness[i].item():.3f}")
    print('-' * 50)

    return torch.where(mouth_openness < MOUTH_OPEN_THRESHOLD)[0]

# def calculate_mouth_openness_dynamic(lm3d):
#     """
#     Dynamically finds frames where the mouth closes rapidly.
#     Only flags frames that are both "closed" and reached that state "quickly".

#     Args:
#         lm3d (torch.Tensor): The landmark tensor of shape [T, 204] or [T, 68, 3].

#     Returns:
#         torch.Tensor: A tensor of frame indices that need substitution.
#     """
#     # --- 1. 設定閾值 ---
#     # 位置閾值：定義什麼程度算是「閉合」。可以比靜態閾值稍高一些。
#     OPENNESS_THRESHOLD = 0.05
#     # 速度閾值：定義多快的速度算是「急遽閉合」。這是一個負數。
#     VELOCITY_THRESHOLD = -0.02

#     # --- 2. 計算嘴巴張開度 ---
#     lm3d = lm3d.reshape([-1, 68, 3])  # [T, 68, 3]
#     face_real_lm = lm3d / 10 + mean_face.to(lm3d.device) # [T, 68, 3]
#     upper_lip = face_real_lm[:, 62, :]  # 上內唇中心點
#     lower_lip = face_real_lm[:, 66, :]  # 下內唇中心點
#     mouth_openness = torch.norm(upper_lip - lower_lip, dim=-1)  # [T]

#     # --- 3. 計算閉合速度 ---
#     # velocity[t] = openness[t] - openness[t-1]
#     # 我們在前面補一個0，使得維度與 mouth_openness 一致
#     velocity = torch.cat([torch.tensor([0.0]).to(lm3d.device), mouth_openness[1:] - mouth_openness[:-1]])

#     # --- 4. 應用雙重條件找出目標幀 ---
#     # 條件1: 張開度小於位置閾值
#     closed_frames = mouth_openness < OPENNESS_THRESHOLD
#     # 條件2: 閉合速度快於速度閾值
#     fast_closing_frames = velocity < VELOCITY_THRESHOLD
    
#     # 最終目標是兩個條件都滿足的幀
#     target_frames_indices = torch.where(closed_frames & fast_closing_frames)[0]

#     print('-' * 50)
#     print("Dynamic Mouth Closure Detection:")
#     print(f"Found {len(target_frames_indices)} frames with rapid mouth closure.")
#     print(f"Frames: {target_frames_indices.tolist()}")
#     print('-' * 50)

#     return target_frames_indices


def calculate_mouth_openness_dynamic(lm3d):
    """
    Dynamically finds frames where the mouth closes rapidly AND is followed by a rapid opening.
    This helps to distinguish speech-related closures from end-of-utterance closures.

    Args:
        lm3d (torch.Tensor): The landmark tensor of shape [T, 204] or [T, 68, 3].

    Returns:
        torch.Tensor: A tensor of frame indices that need substitution.
    """
    # --- 1. 設定閾值 ---
    # 位置閾值：定義什麼程度算是「閉合」。
    OPENNESS_THRESHOLD = 0.05
    # 速度閾值 (閉合)：定義多快的速度算是「急遽閉合」。這是一個負數。
    CLOSING_VELOCITY_THRESHOLD = -0.02
    # 速度閾值 (張開)：定義多快的速度算是「急遽張開」。這是一個正數。
    OPENING_VELOCITY_THRESHOLD = 0.03
    # 向前搜尋窗口：在閉合後，要往前看多少幀來尋找張開動作。
    LOOKAHEAD_WINDOW = 10

    # --- 2. 計算嘴巴張開度 ---
    T = lm3d.shape[0]
    lm3d = lm3d.reshape([-1, 68, 3])  # [T, 68, 3]
    face_real_lm = lm3d / 10 + mean_face.to(lm3d.device) # [T, 68, 3]
    upper_lip = face_real_lm[:, 62, :]  # 上內唇中心點
    lower_lip = face_real_lm[:, 66, :]  # 下內唇中心點
    mouth_openness = torch.norm(upper_lip - lower_lip, dim=-1)  # [T]

    # --- 3. 計算閉合/張開速度 ---
    # velocity[t] = openness[t] - openness[t-1]
    velocity = torch.cat([torch.tensor([0.0]).to(lm3d.device), mouth_openness[1:] - mouth_openness[:-1]])

    # --- 4. 找出候選的「急遽閉合」幀 ---
    # 條件1: 張開度小於位置閾值
    closed_frames = mouth_openness < OPENNESS_THRESHOLD
    # 條件2: 閉合速度快於速度閾值
    fast_closing_frames = velocity < CLOSING_VELOCITY_THRESHOLD
    
    # 找出同時滿足兩個條件的候選幀
    candidate_indices = torch.where(closed_frames & fast_closing_frames)[0]
    
    if len(candidate_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=lm3d.device)

    # --- 5. 驗證候選幀：檢查後續是否有「急遽張開」的動作 ---
    final_target_indices = []
    for idx in candidate_indices:
        # 定義搜尋的窗口範圍
        start_lookahead = idx + 1
        end_lookahead = min(start_lookahead + LOOKAHEAD_WINDOW, T)
        
        if start_lookahead >= end_lookahead:
            continue

        # 在窗口內檢查是否有任何一幀的速度超過了「張開速度閾值」
        if torch.any(velocity[start_lookahead:end_lookahead] > OPENING_VELOCITY_THRESHOLD):
            final_target_indices.append(idx)

    target_frames_indices = torch.tensor(final_target_indices, dtype=torch.long, device=lm3d.device)

    print('-' * 50)
    print("Dynamic Mouth Closure Detection (v2 - with lookahead):")
    print(f"Found {len(target_frames_indices)} frames with rapid mouth closure followed by an opening.")
    print(f"Frames: {target_frames_indices.tolist()}")
    print('-' * 50)

    return target_frames_indices