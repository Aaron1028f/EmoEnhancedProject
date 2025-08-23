import torch
import numpy as np
from numpy.linalg import solve

from utils.commons.tensor_utils import convert_to_tensor

# import filters
from scipy.signal import savgol_filter

import sys
sys.path.append('./')
from emogene.tools.one_euro_filter import OneEuroFilter 

# ========================================
def find_k_nearest_neighbors_cosine(feats, feat_database, K=10):
    """使用餘弦相似度尋找 KNN"""
    feats = convert_to_tensor(feats)
    feat_database = convert_to_tensor(feat_database)

    # 歸一化，將向量轉換為單位向量
    feats_norm = torch.nn.functional.normalize(feats, p=2, dim=1)
    feat_database_norm = torch.nn.functional.normalize(feat_database, p=2, dim=1)

    # 計算餘弦相似度（等同於歸一化後的內積）
    # 值越大表示越相似
    similarity_mat = torch.matmul(feats_norm, feat_database_norm.t())

    # topk 需要找最大的值，所以 largest=True
    ind = similarity_mat.topk(K, dim=1, largest=True).indices
    return ind

# ========================================
def find_k_nearest_neighbors(feats, feat_database, K=10):
    """
    KNN (K-nearest neighbor), return the index of k-nearest neighbors in the feat_database
    args:
        feats: [N_sample_in_batch, C]
        feats_database: [N_sample_in_dataset, C]
        K: the number of topK nearest neighbors
    return:
        ind: [N_sample_in_batch, K=10] the index of K nearest neighbors in the database, CPU tensor
    """
    feats = convert_to_tensor(feats)
    feat_database = convert_to_tensor(feat_database)
    # Training
    feat_base_norm = (feat_database ** 2).sum(-1) # [N_sample_in_database,]
    # start computing KNN ...
    feats_norm = (feats ** 2).sum(-1) # [N_sample_in_batch,]
    # calculate distance via : (x-y)^2  = x^2 + y^2 - 2xy
    distance_mat = (feats_norm.view(-1, 1) + feat_base_norm.view(1, -1) - 2 * feats @ feat_database.t()) # [N_sample_in_batch, N_sample_in_database]
    # get the index of k nearest neighbors
    ind = distance_mat.topk(K, dim=1, largest=False).indices
    return ind

def solve_LLE_projection_batch(feat, feat_base):
    """
    Find LLE projection weights given feat base and target feat
    Project a batch of feat vector into a linear combination of feat_base
    TODO: perform this process in a mini-batch.
    =======================================
    We need to solve the following function
    ```
        min|| feat - \sum_0^k{w_i} * feat_base_i ||, s.t. \sum_0^k{w_i}=1
    ```
    equals to:
        ft = w1*f1 + w2*f2 + ... + wk*fk, s.t. w1+w2+...+wk=1
           = (1-w2-...-wk)*f1 + w2*f2 + ... + wk*fk
     ft-f1 = w2*(f2-f1) + w3*(f3-f1) + ... + wk*(fk-f1)
     ft-f1 = (f2-f1, f3-f1, ..., fk-f1) dot (w2, w3, ..., wk).T
        B  = A dot w_,  here, B: [ndim,]  A: [ndim, k-1], w_: [k-1,]
    Finally,
       ft' = (1-w2-..wk, w2, ..., wk) dot (f1, f2, ..., fk)
    =======================================    
    args:
        feat: [N_sample_in_batch, C], the feat to be preocessed
        feat_base: [N_sample_in_batch, K, C], the base vectors to represent the feat
    return:
        weights: [N, K], the linear weights of K base vectors, sums to 1
        fear_fuse: [N, C], the processed feat
    """
    feat = convert_to_tensor(feat)
    feat_base = convert_to_tensor(feat_base)
    N, K, C = feat_base.shape
    if K == 1:
        weights = torch.ones([N, 1])
        feat_fuse = feat_base[:, 0, ]
        errors = None
    else:
        weights = torch.zeros(N, K, device=feat.device)
        B = feat - feat_base[:, 0, :]   # [N, C]
        A = (feat_base[:, 1:, :] - feat_base[:, 0:1, :]).transpose(1,2)   # [N, C, K-1]
        AT = A.transpose(1,2) # [N, K-1, C]
        # solve the AX=B with Least square method
        # where X [N, K-1] is the weights[1:] we want to learn
        # AT*A*X=AT*B ==> X = inv(ATA)*AT*B
        ATA = torch.bmm(AT, A) # [N, K-1, K-1]
        inv_ATA = torch.inverse(ATA) # [N, K-1, K-1]
        X = torch.bmm(torch.bmm(inv_ATA, AT), B.unsqueeze(2)).squeeze() # [N, K-1] 
        weights[:, 1:] = X
        weights[:, 0] = torch.ones_like(weights[:, 0]) - X.sum(dim=1) 
        feat_fuse = torch.bmm(weights.unsqueeze(1), feat_base).squeeze(1) # [N,1,K] @ [N,K,C] ==> [N,1,C] ==> [N, C]
        errors = (torch.bmm(A,X.unsqueeze(-1)).squeeze() - B).abs().mean(dim=-1) # [N,]
    return feat_fuse, errors, weights

def compute_LLE_projection(feats, feat_database, K=10):
    """
    Project the feat into a linear combination of K base vectors in feat_database
    args:
        feat: [N_sample_in_batch, C], the feat to be processed
        feat_database: [N_sample_in_batch, C], all feat datapoints in dataset
        K: int, number of K neighbors 
    return:
        weights: [N_sample_K, ] 
    """
    # ======================================
    index_of_K_neighbors_in_database = find_k_nearest_neighbors(feats, feat_database, K) # [N_sample_in_batch, K=10]
    
    # index_of_K_neighbors_in_database = find_k_nearest_neighbors_cosine(feats, feat_database, K) # [N_sample_in_batch, K=10]
    # ======================================
    feat_base = feat_database[index_of_K_neighbors_in_database]
    # print("performing LLE projection ...")
    feat_fuse, errors, weights = solve_LLE_projection_batch(feats, feat_base)
    # print("LLE projection Done.")
    return feat_fuse, errors, weights

# ============== bs_ver_modified ==============
# brow: 17~26
# eye: 36~47 
# nose: 27~35
# mouth: 48~67
# yaw: 0~16


# 68 個關鍵點的索引分組
FACIAL_LANDMARK_REGIONS = {
    'mouth': list(range(48, 68)),      # 嘴巴 (20 points)
    'right_eyebrow': list(range(17, 22)), # 右眉毛 (5 points)
    'left_eyebrow': list(range(22, 27)),  # 左眉毛 (5 points)
    'right_eye': list(range(36, 42)),     # 右眼 (6 points)
    'left_eye': list(range(42, 48)),      # 左眼 (6 points)
    'nose': list(range(27, 36)),         # 鼻子 (9 points)
    'jaw': list(range(0, 17))            # 下巴輪廓 (17 points)
}

# 為了簡化，可以先粗略地分為「嘴巴」和「其他」
FACIAL_LANDMARK_REGIONS_SIMPLE = {
    'mouth': list(range(48, 68)),
    'upper_face': list(range(0, 48)) # 包含除了嘴巴以外的所有部分
}

# 測試，每個區域都分開
FAICIAL_LANDMARK_REGIONS_TEST = {
    'mouth': list(range(48, 68)),      # 嘴巴 (20 points)
    # 'right_eyebrow': list(range(17, 22)), # 右眉毛 (5 points)
    # 'left_eyebrow': list(range(22, 27)),  # 左眉毛 (5 points)
    # 'right_eye': list(range(36, 42)),     # 右眼 (6 points)
    # 'left_eye': list(range(42, 48)),      # 左眼 (6 points)
    'upper_face': list(range(17, 27)) + list(range(36, 48)), # 包含眉毛和眼睛 (22 points)
    'nose': list(range(27, 36)),         # 鼻子 (9 points)
    'jaw': list(range(0, 17))            # 下巴輪廓 (17 points)
}

def smooth_features_seq_one_euro(feats_seq, freq=25, min_cutoff=0.4, beta=0.7, d_cutoff=1.0):
    """
    使用 One Euro Filter 平滑特徵的時間序列。

    args:
        feats_seq: [T, C] 的張量，T 是總幀數，C 是特徵維度。
        freq: 資料的採樣頻率 (Hz)。
        min_cutoff: 最小截止頻率。值越低，靜止時的平滑效果越強。
        beta: 截止頻率的變化速度。值越高，對快速運動的反應越靈敏。
        d_cutoff: 導數的截止頻率，通常保持為 1.0。
    return:
        smoothed_feats_seq: [T, C] 平滑後的特徵張量。
    """
    # 將輸入張量轉換為 numpy array
    feats_seq_np = feats_seq.cpu().numpy()
    T, C = feats_seq_np.shape
    
    # 準備用於存放平滑後數據的 array
    smoothed_feats_seq_np = np.zeros_like(feats_seq_np)
    
    # 處理第一幀
    t0 = 0.0
    x0 = feats_seq_np[0]
    smoothed_feats_seq_np[0] = x0
    
    # 為每個特徵維度初始化一個 OneEuroFilter
    # 注意：根據提供的 one_euro_filter.py，初始化需要 t0 和 x0
    filters = [
        OneEuroFilter(t0, x0[c], min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        for c in range(C)
    ]
    
    # 逐幀應用濾波器
    t_step = 1.0 / freq
    for t in range(1, T):
        current_t = t0 + t * t_step
        current_x = feats_seq_np[t]
        
        for c in range(C):
            smoothed_feats_seq_np[t, c] = filters[c](current_t, current_x[c])
            
    # 轉回 torch tensor
    smoothed_feats_seq = torch.from_numpy(smoothed_feats_seq_np).to(feats_seq.device)
    
    return smoothed_feats_seq

def smooth_features_seq_savgol(feats_seq, window_length=5, polyorder=2):
    """
    使用 Savitzky-Golay 濾波器平滑特徵的時間序列。

    args:
        feats_seq: [T, C] 的張量，T 是總幀數，C 是特徵維度。
        window_length: 濾波器的窗口大小，必須是正奇數。值越大，平滑效果越強。
        polyorder: 擬合多項式的階數。必須小於 window_length。值越小，平滑效果越強。
    return:
        smoothed_feats_seq: [T, C] 平滑後的特徵張量。
    """
    # 確保 window_length 是奇數
    if window_length % 2 == 0:
        window_length += 1
        
    # savgol_filter 需要在 numpy array 上操作
    feats_seq_np = feats_seq.cpu().numpy()
    
    # 沿著時間軸 (axis=0) 進行濾波
    smoothed_feats_seq_np = savgol_filter(feats_seq_np, window_length, polyorder, axis=0)
    
    # 轉回 torch tensor
    smoothed_feats_seq = torch.from_numpy(smoothed_feats_seq_np).to(feats_seq.device)
    
    return smoothed_feats_seq


def compute_LLE_projection_by_parts(feats, feat_database, K=10, gene_feat=None, regions=FACIAL_LANDMARK_REGIONS_SIMPLE):
    """
    對臉部不同區域分別進行 LLE 投影。
    
    args:
        feats: [N, 204], 完整的 68*3 特徵
        feat_database: [M, 204], 完整的特徵庫
        K: KNN 的鄰居數量
        regions: 一個字典，定義了區域名稱和對應的 68 點索引
    return:
        final_feat_fuse_raw: [N, 204], 組合後的 LLE 投影結果
    """
    # regions = FAICIAL_LANDMARK_REGIONS_TEST
    
    
    if regions is None:
        # 如果未提供區域劃分，則退回原始的整體 LLE
        return compute_LLE_projection(feats, feat_database, K)

    N, C = feats.shape
    final_feat_fuse_raw = torch.zeros_like(feats)
    final_errors = {}
    final_weights = {}

    # 將 68*3 的索引轉換為對應的 C 維度索引
    def get_dim_indices(landmark_indices):
        dim_indices = []
        for idx in landmark_indices:
            dim_indices.extend([idx*3, idx*3 + 1, idx*3 + 2])
        return sorted(dim_indices)

    for name, landmark_indices in regions.items():
        print(f"Performing LLE for region: {name}")
        dim_indices = get_dim_indices(landmark_indices)
        
        # 根據區域索引，切分輸入特徵和特徵庫
        feats_part = feats[:, dim_indices]
        feat_database_part = feat_database[:, dim_indices]
        
        # 對該區域獨立進行 LLE 投影
        feat_fuse_part, errors_part, weights_part = compute_LLE_projection(
            feats_part, feat_database_part, K
        )
        
        # 將計算結果放回完整特徵的對應位置
        final_feat_fuse_raw[:, dim_indices] = feat_fuse_part
        final_errors[name] = errors_part
        final_weights[name] = weights_part
    
    # smooth the final fused features (use savgol_filter, good engough for May, but not good engough for Feng)
    # smoothed_feat_fuse = smooth_features_seq_savgol(
    #     final_feat_fuse_raw, 
    #     window_length=9, 
    #     polyorder=2
    # )
    
    # # <experiment>
    # # testing using geneface feat when geneface mouth is small (not open)
    # if gene_feat is not None:
    #     from emogene.tools.mouth_openness_calculation import calculate_mouth_openness, calculate_mouth_openness_dynamic
    #     # small_mouth_frames = calculate_mouth_openness(gene_feat)
    #     small_mouth_frames = calculate_mouth_openness_dynamic(gene_feat)


    #     # substitute the small mouth frames with the original gene_feat
    #     gene_feat_T683 = gene_feat.reshape([-1, 68, 3])  # [T, 68, 3]
    #     final_feat_fuse_raw_T683 = final_feat_fuse_raw.reshape([-1, 68, 3])  # [T, 68, 3]
    #     final_feat_fuse_raw_T683[small_mouth_frames, 48:68, :] = gene_feat_T683[small_mouth_frames, 48:68, :]
    #     final_feat_fuse_raw = final_feat_fuse_raw_T683.reshape([-1, 204])
    #     # print(f"Substituted {len(small_mouth_frames)} frames with gene_feat where mouth is small.")
    #     # print(f"The substituted frames are: {small_mouth_frames.tolist()}")
    # # </experiment>
    
    # <experiment>
    if gene_feat is not None:
        final_feat_fuse_raw = apply_smooth_blending_for_closure(
            final_feat_fuse_raw, gene_feat, window_size=5
        )
    # </experiment>
    
    # use one_euro_filter
    smoothed_feat_fuse = smooth_features_seq_one_euro(
        final_feat_fuse_raw, 
        freq=25, 
        min_cutoff=0.6, # (last setting is 0.4)
        # beta=0.7, (default)
        beta=1.2, # (last setting is 0.9)
        d_cutoff=1.0
    )
    
    # <experiment>
    # substitute the small mouth frames with the original gene_feat manually AFTER smoothing
    smoothed_feat_fuse = apply_final_mouth_closure(smoothed_feat_fuse, gene_feat)
    # </experiment>    
    

    return smoothed_feat_fuse, final_errors, final_weights

def apply_final_mouth_closure(smoothed_feat, gene_feat, final_threshold=0.04, blend_window_size=4):
    """
    在全局平滑後，對仍未完全閉合的嘴部進行最終的、柔和的修正。
    使用窗口化混合來平滑過渡，避免突變。

    Args:
        smoothed_feat (torch.Tensor): [T, 204], 經過全局平滑後的特徵。
        gene_feat (torch.Tensor): [T, 204], 原始的 GeneFace++ 特徵，作為閉嘴的目標。
        final_threshold (float): 最終的閉合判斷閾值。
        blend_window_size (int): 在目標幀前後進行混合的半徑。總窗口大小為 2*size+1。

    Returns:
        torch.Tensor: [T, 204], 經過最終修正的特徵。
    """
    from emogene.tools.mouth_openness_calculation import calculate_mouth_openness, calculate_mouth_openness_dynamic
    
    # 1. 找出閉合失敗的幀
    mouth_openness = calculate_mouth_openness(smoothed_feat, return_distance_only=True)
    originally_targeted_frames = calculate_mouth_openness_dynamic(gene_feat)
    
    # 確保我們只修正那些本來就想閉嘴但沒做好的幀
    failed_mask = (mouth_openness > final_threshold)
    target_mask = torch.zeros_like(failed_mask)
    target_mask[originally_targeted_frames] = True
    final_target_indices = torch.where(failed_mask & target_mask)[0]
    
    if len(final_target_indices) == 0:
        return smoothed_feat

    print(f"Final closure: Found {len(final_target_indices)} frames that failed to close properly. Applying soft blending fix.")
    print(f"The frames are: {final_target_indices.tolist()}")
    print('-'*60)

    # 2. 準備數據
    corrected_feat = smoothed_feat.clone()
    corrected_feat_T683 = corrected_feat.reshape([-1, 68, 3])
    gene_feat_T683 = gene_feat.reshape([-1, 68, 3])
    # mouth_indices = list(range(48, 68))
    mouth_indices = list(range(0, 68))
    T = smoothed_feat.shape[0]

    # 3. 創建一個應用於所有幀的基礎權重數組，初始為0
    blend_weights = torch.zeros(T, device=smoothed_feat.device)

    # 4. 為每個目標窗口生成混合權重
    # 使用三角窗口 (線性漸變) 作為權重
    window = torch.bartlett_window(blend_window_size * 2 + 1, periodic=False, device=smoothed_feat.device)

    for frame_idx in final_target_indices:
        start = max(0, frame_idx - blend_window_size)
        end = min(T, frame_idx + blend_window_size + 1)
        
        win_start = max(0, blend_window_size - frame_idx)
        win_end = min(len(window), T - frame_idx + blend_window_size)

        # 將窗口權重應用到全局權重數組上
        # 使用 torch.max 確保重疊的窗口能取最強的修正權重
        current_weights = blend_weights[start:end]
        new_weights = window[win_start:win_end]
        blend_weights[start:end] = torch.max(current_weights, new_weights)

    # 5. 根據最終計算出的權重，一次性應用混合
    # 擴展權重以便進行廣播操作
    w = blend_weights.view(T, 1, 1) # [T, 1, 1]
    
    # 僅對嘴部區域進行加權平均
    corrected_feat_T683[:, mouth_indices, :] = \
        (1 - w) * corrected_feat_T683[:, mouth_indices, :] + \
        w * gene_feat_T683[:, mouth_indices, :]

    return corrected_feat_T683.reshape([-1, 204])


# def apply_final_mouth_closure(smoothed_feat, gene_feat, final_threshold=0.04, local_smooth_strength=0.8):
#     """
#     在全局平滑後，對仍未完全閉合的嘴部進行最終的、外科手術式的修正。

#     Args:
#         smoothed_feat (torch.Tensor): [T, 204], 經過全局平滑後的特徵。
#         gene_feat (torch.Tensor): [T, 204], 原始的 GeneFace++ 特徵，作為閉嘴的目標。
#         final_threshold (float): 最終的閉合判斷閾值。高於此值被認為是“閉合失敗”。
#         local_smooth_strength (float): 對鄰居幀進行局部平滑的強度 (0到1之間)。

#     Returns:
#         torch.Tensor: [T, 204], 經過最終修正的特徵。
#     """
#     from emogene.tools.mouth_openness_calculation import calculate_mouth_openness
    
#     # 1. 找出閉合失敗的幀
#     # 注意：這裡我們需要一個簡單的 mouth_openness 計算函式，它只計算距離而不做判斷
#     # 假設 calculate_mouth_openness 可以返回一個距離數組
#     mouth_openness = calculate_mouth_openness(smoothed_feat, return_distance_only=True)
#     failed_frames = torch.where(mouth_openness > final_threshold)[0]
    
#     # 找出原始動態檢測出的需要閉合的幀，只在這些幀的子集裡尋找失敗者
#     from emogene.tools.mouth_openness_calculation import calculate_mouth_openness_dynamic
#     originally_targeted_frames = calculate_mouth_openness_dynamic(gene_feat)
    
#     # 取交集，確保我們只修正那些本來就想閉嘴但沒做好的幀
#     final_target_frames = [f for f in failed_frames if f in originally_targeted_frames]
    
#     if not final_target_frames:
#         return smoothed_feat

#     print(f"Final closure: Found {len(final_target_frames)} frames that failed to close properly. Applying surgical fix.")
#     print(f"The frames are: {final_target_frames}")
#     print('-'*60)

#     # 2. 準備數據
#     corrected_feat = smoothed_feat.clone()
#     corrected_feat_T683 = corrected_feat.reshape([-1, 68, 3])
#     gene_feat_T683 = gene_feat.reshape([-1, 68, 3])
    
#     # <exp>
#     mouth_indices = list(range(48, 68))
#     # mouth_indices = list(range(0, 68))
#     # </exp>

#     # 3. 進行硬性替換和局部平滑
#     for frame_idx in final_target_frames:
#         # --- 硬性替換目標幀 ---
#         corrected_feat_T683[frame_idx, mouth_indices, :] = gene_feat_T683[frame_idx, mouth_indices, :]

#         # --- 局部平滑鄰居幀 ---
#         # 平滑前一幀
#         prev_idx = frame_idx - 1
#         if prev_idx >= 0 and prev_idx not in final_target_frames: # 確保鄰居本身不是目標
#             # 將鄰居幀的嘴型，向被替換的目標幀的嘴型靠近一點
#             corrected_feat_T683[prev_idx, mouth_indices, :] = \
#                 (1 - local_smooth_strength) * corrected_feat_T683[prev_idx, mouth_indices, :] + \
#                 local_smooth_strength * corrected_feat_T683[frame_idx, mouth_indices, :]

#         # 平滑後一幀
#         next_idx = frame_idx + 1
#         if next_idx < len(smoothed_feat) and next_idx not in final_target_frames:
#             corrected_feat_T683[next_idx, mouth_indices, :] = \
#                 (1 - local_smooth_strength) * corrected_feat_T683[next_idx, mouth_indices, :] + \
#                 local_smooth_strength * corrected_feat_T683[frame_idx, mouth_indices, :]

#     return corrected_feat_T683.reshape([-1, 204])


def apply_smooth_blending_for_closure(emogene_feat, gene_feat, window_size=5):
    """
    在需要閉嘴的幀周圍，將 Emogene 特徵平滑地混合到 GeneFace++ 特徵。

    Args:
        emogene_feat (torch.Tensor): [T, 204], 經過 LLE 處理的 Emogene 特徵。
        gene_feat (torch.Tensor): [T, 204], 原始的 GeneFace++ 特徵。
        window_size (int): 在目標幀前後進行混合的幀數。

    Returns:
        torch.Tensor: [T, 204], 經過平滑混合處理後的特徵。
    """
    from emogene.tools.mouth_openness_calculation import calculate_mouth_openness_dynamic
    
    # 1. 找出需要強制閉嘴的核心幀
    target_frames = calculate_mouth_openness_dynamic(gene_feat)
    if len(target_frames) == 0:
        return emogene_feat

    # 2. 準備數據
    blended_feat = emogene_feat.clone()
    emogene_feat_T683 = emogene_feat.reshape([-1, 68, 3])
    gene_feat_T683 = gene_feat.reshape([-1, 68, 3])
    blended_feat_T683 = blended_feat.reshape([-1, 68, 3])

    # 3. 創建混合權重 (例如，線性漸變)
    # 權重為1時，完全使用 gene_feat
    # weights = torch.linspace(0, 1, steps=window_size + 1)
    
    # 3. 創建混合權重 (使用非線性權重，例如平方)
    # 這會讓權重在接近1時變化更劇烈，形成更明確的閉合
    weights = torch.linspace(0, 1, steps=window_size + 1)**2    

    # 4. 對每個目標幀應用混合窗口
    for frame_idx in target_frames:
        # 處理中心幀 (100% gene_feat)
        blended_feat_T683[frame_idx, 48:68, :] = gene_feat_T683[frame_idx, 48:68, :]

        # 向前後擴展混合區域
        for i in range(1, window_size + 1):
            # --- 向前混合 (漸出) ---
            prev_idx = frame_idx - i
            if prev_idx >= 0:
                # 權重從1向0遞減
                w = weights[-i-1] 
                blended_feat_T683[prev_idx, 48:68, :] = \
                    (1 - w) * emogene_feat_T683[prev_idx, 48:68, :] + \
                    w * gene_feat_T683[prev_idx, 48:68, :]

            # --- 向後混合 (漸入) ---
            next_idx = frame_idx + i
            if next_idx < len(emogene_feat):
                # 權重從1向0遞減
                w = weights[-i-1]
                blended_feat_T683[next_idx, 48:68, :] = \
                    (1 - w) * emogene_feat_T683[next_idx, 48:68, :] + \
                    w * gene_feat_T683[next_idx, 48:68, :]

    return blended_feat_T683.reshape([-1, 204])


# ============== bs_ver_modified ==============

if __name__ == '__main__':
    audio_feats = torch.randn(1000, 64).numpy()
    feat_database = torch.randn(10000, 64).numpy()
    Knear = 10
    # LLE_percent =1.
    ind = find_k_nearest_neighbors(audio_feats, feat_database, K=Knear)
    weights, feat_fuse = compute_LLE_projection(audio_feats, feat_database, K=10)
    # audio_feats = audio_feats * (1-LLE_percent) + feat_fuse * LLE_percent
    print(" ")