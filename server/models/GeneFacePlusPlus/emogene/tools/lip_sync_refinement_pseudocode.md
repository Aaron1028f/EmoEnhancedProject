1. `calculate_mouth_openness_dynamic` 的偽代碼
此演算法旨在從原始的臉部特徵序列中，識別出代表爆破音（如 /b/, /p/）的關鍵幀。其核心思想是，這些音素不僅僅是嘴唇閉合，而是伴隨著一個「快速閉合」後緊接著一個「快速張開」的過程。

```psuedocode
ALGORITHM 1: Dynamic Mouth Closure Detection

FUNCTION calculate_mouth_openness_dynamic(landmarks_sequence):
    // --- 1. 初始化參數 ---
    PosThreshold ← 0.05                // 定義嘴部“閉合”的張開度閾值
    NegVelThreshold ← -0.02            // 定義“快速閉合”的速度閾值 (負值)
    PosVelThreshold ← 0.03             // 定義“快速張開”的速度閾值 (正值)
    LookaheadWindow ← 10               // 閉合後向前搜尋快速張開的幀數範圍

    // --- 2. 計算嘴部張開度及其變化速度 ---
    // 遍歷序列，計算每幀上下唇關鍵點的歐氏距離
    openness_series ← calculate_lip_distance(landmarks_sequence)
    // 計算張開度的逐幀變化率 (速度)
    velocity_series ← difference(openness_series)

    // --- 3. 找出所有“快速閉合”的候選幀 ---
    candidate_frames ← empty_list
    FOR t FROM 1 TO length(landmarks_sequence) - 1:
        // 條件：當前幀既是“閉合”狀態，又是“快速閉合”的
        IF openness_series[t] < PosThreshold AND velocity_series[t] < NegVelThreshold:
            add t to candidate_frames
        END IF
    END FOR

    // --- 4. 驗證候選幀，確保其後跟隨一個“快速張開”動作 ---
    final_target_frames ← empty_list
    FOR each frame_index in candidate_frames:
        // 定義向前看的搜尋窗口
        window_start ← frame_index + 1
        window_end ← min(frame_index + LookaheadWindow, length(landmarks_sequence))

        // 在窗口內檢查是否存在至少一幀是“快速張開”的
        IF exists t in [window_start, window_end) such that velocity_series[t] > PosVelThreshold:
            add frame_index to final_target_frames
        END IF
    END FOR

    RETURN final_target_frames
```
2. `apply_smooth_blending_for_closure` 的偽代碼
此演算法作為預處理步驟，在全局平滑之前，將 LLE 輸出的特徵在檢測到的閉合幀周圍，平滑地引導向一個完美的閉合狀態。
```psuedocode
ALGORITHM 2: Preventive Closure Guidance via Smooth Blending

FUNCTION apply_smooth_blending_for_closure(LLE_features, Gene_features):
    // --- 1. 初始化參數 ---
    WindowSize ← 5                     // 在目標幀前後進行混合的半徑

    // --- 2. 獲取需要強制閉合的目標幀 ---
    // 使用算法1來識別關鍵幀
    target_frames ← calculate_mouth_openness_dynamic(Gene_features)
    
    blended_features ← copy(LLE_features)

    // --- 3. 創建混合權重 ---
    // 權重從0平滑過渡到1，w(d)表示距離中心d幀時的權重
    // w(0)=1 (完全使用Gene_features), w(WindowSize)=0 (完全使用LLE_features)
    weights ← create_blending_weights(from=0, to=1, steps=WindowSize+1)

    // --- 4. 對每個目標幀的鄰域應用混合 ---
    FOR each frame_idx in target_frames:
        // 在中心幀，直接替換為完美的閉合嘴型
        blended_features.mouth[frame_idx] ← Gene_features.mouth[frame_idx]

        // 在中心幀周圍的窗口內進行加權混合
        FOR d FROM 1 TO WindowSize:
            w ← weights[d] // 獲取對應距離的權重

            // 處理中心幀之前的鄰居
            IF frame_idx - d >= 0:
                prev_idx ← frame_idx - d
                blended_features.mouth[prev_idx] ← (1-w) * LLE_features.mouth[prev_idx] + w * Gene_features.mouth[prev_idx]
            END IF

            // 處理中心幀之後的鄰居
            IF frame_idx + d < length(LLE_features):
                next_idx ← frame_idx + d
                blended_features.mouth[next_idx] ← (1-w) * LLE_features.mouth[next_idx] + w * Gene_features.mouth[next_idx]
            END IF
        END FOR
    END FOR

    RETURN blended_features
```

3. `apply_final_mouth_closure` 的偽代碼
此演算法作為後處理步驟，在全局平滑之後，對極少數仍未完全閉合的嘴型進行最終的、柔和的修正，以保證物理真實感。
```psuedocode
ALGORITHM 3: Post-hoc Surgical Closure Correction

FUNCTION apply_final_mouth_closure(Smoothed_features, Gene_features):
    // --- 1. 初始化參數 ---
    FinalThreshold ← 0.04              // 最終判斷嘴部是否閉合的張開度閾值
    BlendWindowSize ← 4                // 柔和混合窗口的半徑

    // --- 2. 找出閉合失敗的幀 ---
    // 獲取原始希望閉合的幀
    originally_targeted_frames ← calculate_mouth_openness_dynamic(Gene_features)
    // 計算平滑後序列的嘴部張開度
    current_openness ← calculate_lip_distance(Smoothed_features)
    // 找出那些“本應閉合”但“實際未閉合”的幀
    failed_frames ← frames where (current_openness > FinalThreshold) AND (frame is in originally_targeted_frames)

    IF is_empty(failed_frames):
        RETURN Smoothed_features
    END IF

    // --- 3. 創建全局混合權重序列 ---
    T ← length(Smoothed_features)
    global_blend_weights ← array of zeros with length T
    
    // 創建一個三角旗幟形狀的權重窗口 (例如 Bartlett window)
    // 中心為1，兩邊線性衰減至0
    window_shape ← create_triangular_window(size = 2 * BlendWindowSize + 1)

    // --- 4. 將所有失敗幀的影響疊加到全局權重上 ---
    FOR each frame_idx in failed_frames:
        // 定義窗口應用的範圍
        start ← max(0, frame_idx - BlendWindowSize)
        end ← min(T, frame_idx + BlendWindowSize + 1)
        
        // 將旗幟窗口放置在對應位置，並與已有權重取最大值
        // 這確保了重疊區域取最強的修正力度
        FOR t FROM start TO end-1:
            w_shape_idx ← t - (frame_idx - BlendWindowSize)
            global_blend_weights[t] ← max(global_blend_weights[t], window_shape[w_shape_idx])
        END FOR
    END FOR

    // --- 5. 根據全局權重一次性應用修正 ---
    corrected_features ← copy(Smoothed_features)
    FOR t FROM 0 TO T-1:
        w ← global_blend_weights[t]
        // 僅對嘴部區域進行加權平均
        corrected_features.mouth[t] ← (1-w) * Smoothed_features.mouth[t] + w * Gene_features.mouth[t]
    END FOR

    RETURN corrected_features
```
---
## explain the difference between the two mouth_closure functions before and after One-Euro Filter
好的，這就為您詳細解釋在 One-Euro Filter 前後，您所使用的兩個 `mouth_closure` 函式在平滑演算法上的核心差異。

總體來說，這兩個函式雖然目標都是閉合嘴部，但它們的**設計哲學、作用時機、演算法策略和影響範圍**截然不同。它們分別扮演著「預防性引導」和「全局精修」的角色，是一個分工明確的組合。

---

### 1. `apply_smooth_blending_for_closure` (濾波前)

這個函式可以稱之為「**局部引導式混合 (Local-Guided Blending)**」。

*   **作用對象**: 原始、未經平滑、可能帶有高頻抖動的 LLE 輸出 (`final_feat_fuse_raw`)。
*   **核心演算法**:
    1.  **迭代式處理 (Iterative Processing)**：它通過一個 `for` 迴圈，**逐一處理**每一個被 `calculate_mouth_openness_dynamic` 檢測出的「核心目標幀」。
    2.  **局部窗口混合 (Local Window Blending)**：對於每一個目標幀，它以該幀為中心，創建一個獨立的、固定大小的 (`window_size=5`) 影響窗口。
    3.  **獨立計算與應用 (Independent Calculation)**：在迴圈的每一次迭代中，它都會根據當前目標幀的位置，獨立地計算其周圍鄰居幀的混合權重，並**直接修改**特徵序列。如果兩個目標幀的窗口有重疊，後處理的窗口會**覆蓋**先處理的窗口的結果。
    4.  **權重函數**: 使用 `torch.linspace(...)**2`，這是一個簡單的非線性插值，用於在窗口內創建從 0 到 1 的漸變權重。

*   **演算法特點**:
    *   **策略**: 預防性的。它在全局平滑之前，主動將有問題的區域「拉」到正確的軌道上。
    *   **範圍**: 局部的、離散的。它的每一次操作都只影響一個小鄰域，並且是逐個窗口獨立應用的。
    *   **比喻**: 像是在一條坑坑窪窪的土路上，在鋪設柏油（One-Euro Filter）之前，先把幾個最大的坑（閉嘴失敗）及其周邊用沙土填平，為後續的全局施工打好基礎。

### 2. `apply_final_mouth_closure` (濾波後)

這個函式可以稱之為「**全局疊加式混合 (Global-Aggregated Blending)**」。

*   **作用對象**: 已經過 One-Euro Filter 處理的、整體非常平滑但可能在細節處（如嘴唇）仍有微小偏差的特徵序列 (`smoothed_feat_fuse`)。
*   **核心演算法**:
    1.  **全局權重構建 (Global Weight Construction)**：它首先創建一個與整個時間序列等長、初始值全為零的全局混合權重數組 `blend_weights`。
    2.  **權重疊加 (Weight Aggregation)**：它遍歷所有「閉合失敗」的幀，在每一幀的位置上「放置」一個三角旗幟形狀的權重窗口 (`torch.bartlett_window`)。關鍵在於，當窗口發生重疊時，它使用 `torch.max` 來合併權重。這意味著在重疊區域，會取**最強的修正力度**，而不是互相覆蓋。
    3.  **一次性應用 (One-shot Application)**：在構建完最終的、合併了所有窗口影響的 `global_blend_weights` 之後，它才**一次性地**將這個權重曲線應用於整個平滑後的特徵序列，完成最終的混合修正。
    4.  **權重函數**: 使用 `torch.bartlett_window`，這是一個標準的三角窗函數，能提供中心最強、向兩側線性衰減的平滑過渡。

*   **演算法特點**:
    *   **策略**: 修正性的、品質保證的。它在全局平滑後，對結果進行最終檢查和精修。
    *   **範圍**: 全局的、整合的。它先整合所有需要修正點的資訊，形成一個統一的「修正藍圖」（`global_blend_weights`），然後一次性施工。
    *   **比喻**: 像是在已經鋪好柏油的路上進行最後的精修。檢查員標出了所有仍然輕微不平整的地方，然後一個專業的施工隊根據這些標記，用一個平滑的刮刀，一次性地將所有不平整處用一層薄薄的材料完美地抹平，確保最終路面的絕對平滑。

### 核心差異總結

| 特性 | `apply_smooth_blending_for_closure` (濾波前) | `apply_final_mouth_closure` (濾波後) |
| :--- | :--- | :--- |
| **處理方式** | **迭代式 (Iterative)**，在迴圈中逐個修正 | **整合式 (Aggregative)**，先整合所有資訊再修正 |
| **影響範圍** | **局部 (Local)**，每次操作影響一個小窗口 | **全局 (Global)**，先構建全局權重再應用 |
| **重疊處理** | **覆蓋 (Overwrite)**，後處理的窗口會覆蓋前者 | **疊加取最大 (Aggregate by Max)**，重疊區取最強修正 |
| **演算法目標** | **預防與引導**，為後續平滑做準備 | **修正與精修**，保證最終結果的物理真實感 |
| **演算法本質** | 一系列的**獨立局部插值** | 一個**全局加權平均**，其權重由多個窗口疊加而成 |

這兩種截然不同的演算法設計，恰恰說明了您對問題的深刻理解：在平滑前需要的是**針對性的預處理**，而在平滑後則需要一種**不破壞已有平滑性的、更為柔和的全局修正方案**。