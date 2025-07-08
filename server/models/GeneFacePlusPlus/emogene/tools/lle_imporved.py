import torch
import numpy as np
from numpy.linalg import solve

from utils.commons.tensor_utils import convert_to_tensor

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

def compute_LLE_projection_by_parts(feats, feat_database, K=10, regions=FACIAL_LANDMARK_REGIONS_SIMPLE):
    """
    對臉部不同區域分別進行 LLE 投影。
    
    args:
        feats: [N, 204], 完整的 68*3 特徵
        feat_database: [M, 204], 完整的特徵庫
        K: KNN 的鄰居數量
        regions: 一個字典，定義了區域名稱和對應的 68 點索引
    return:
        final_feat_fuse: [N, 204], 組合後的 LLE 投影結果
    """
    if regions is None:
        # 如果未提供區域劃分，則退回原始的整體 LLE
        return compute_LLE_projection(feats, feat_database, K)

    N, C = feats.shape
    final_feat_fuse = torch.zeros_like(feats)
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
        final_feat_fuse[:, dim_indices] = feat_fuse_part
        final_errors[name] = errors_part
        final_weights[name] = weights_part

    return final_feat_fuse, final_errors, final_weights

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