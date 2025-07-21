import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def run_comparison():
    """
    加載 May 和 Feng 模型的數據，並進行量化與視覺化比較。
    """
    # --- 1. 加載數據 ---
    base_path = './' # 假設此腳本與 .npy 檔案在同一目錄下
    
    try:
        # 加載 May 的數據
        may_std_np = np.load(os.path.join(base_path, 'May_idexp_lm3d_std.npy'))
        may_lower_np = np.load(os.path.join(base_path, 'May_lower.npy'))
        may_upper_np = np.load(os.path.join(base_path, 'May_upper.npy'))

        # 加載 Feng 的數據
        feng_std_np = np.load(os.path.join(base_path, 'Feng_idexp_lm3d_std.npy'))
        feng_lower_np = np.load(os.path.join(base_path, 'Feng_lower.npy'))
        feng_upper_np = np.load(os.path.join(base_path, 'Feng_upper.npy'))
    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案 {e.filename}。")
        print("請確保此腳本與您的 .npy 檔案位於同一個目錄中，或者修改 'base_path'。")
        return

    # 將 numpy array 轉換為 torch tensor
    may_std = torch.from_numpy(may_std_np)
    may_lower = torch.from_numpy(may_lower_np)
    may_upper = torch.from_numpy(may_upper_np)
    
    feng_std = torch.from_numpy(feng_std_np)
    feng_lower = torch.from_numpy(feng_lower_np)
    feng_upper = torch.from_numpy(feng_upper_np)

    # --- 2. 方法一：計算「動態範圍總分」 ---
    print("="*40)
    print("方法一：整體量化比較")
    print("="*40)

    # 計算動態範圍 (Dynamic Range)
    may_range = may_upper - may_lower
    feng_range = feng_upper - feng_lower

    # 計算動態範圍總分 (使用平均值)
    may_range_score = may_range.mean().item()
    feng_range_score = feng_range.mean().item()

    # 計算標準差總分 (衡量原始數據的變化量)
    may_std_score = may_std.mean().item()
    feng_std_score = feng_std.mean().item()

    print(f"原始數據標準差 (STD Score):")
    print(f"  - May : {may_std_score:.4f}")
    print(f"  - Feng: {feng_std_score:.4f}")
    print(f"  - 差異 (May/Feng): {may_std_score / feng_std_score:.2f}x\n")

    print(f"正規化後動態範圍 (Dynamic Range Score):")
    print(f"  - May : {may_range_score:.4f}")
    print(f"  - Feng: {feng_range_score:.4f}")
    print(f"  - 差異 (May/Feng): {may_range_score / feng_range_score:.2f}x")

    # --- 3. 方法二：分區域量化比較 ---
    print("\n" + "="*40)
    print("方法二：分區域量化比較 (動態範圍)")
    print("="*40)
    
    # dlib 68點定義
    FACIAL_REGIONS = {
        '輪廓 (Jaw)': list(range(0, 17)),
        '眉毛 (Eyebrows)': list(range(17, 27)),
        '鼻子 (Nose)': list(range(27, 36)),
        '眼睛 (Eyes)': list(range(36, 48)),
        '嘴巴 (Mouth)': list(range(48, 68)),
    }

    regional_scores = {'May': [], 'Feng': []}
    region_labels = list(FACIAL_REGIONS.keys())

    for name, indices in FACIAL_REGIONS.items():
        may_region_score = may_range[indices].mean().item()
        feng_region_score = feng_range[indices].mean().item()
        regional_scores['May'].append(may_region_score)
        regional_scores['Feng'].append(feng_region_score)
        
        print(f"區域: {name}")
        print(f"  - May : {may_region_score:.4f}")
        print(f"  - Feng: {feng_region_score:.4f}")
        print(f"  - 差異 (May/Feng): {may_region_score / feng_region_score:.2f}x")

    # # # --- 4. 方法三：視覺化比較 ---
    # print("\n" + "="*40)
    # print("方法三：視覺化比較")
    # print("="*40)

    # # A. 長條圖
    # # plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 設置字體以支持中文
    # plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

    # x = np.arange(len(region_labels))
    # width = 0.35

    # fig, ax = plt.subplots(figsize=(12, 7))
    # rects1 = ax.bar(x - width/2, regional_scores['May'], width, label='May')
    # rects2 = ax.bar(x + width/2, regional_scores['Feng'], width, label='Feng')

    # # ax.set_ylabel('平均動態範圍 (Normalized)', fontsize=12)
    # ax.set_ylabel('Normalized dynamic range', fontsize=12)
    # # ax.set_title('May 與 Feng 模型在不同臉部區域的動態範圍比較', fontsize=16)
    # ax.set_title('Comparison of Dynamic Range in Different Facial Regions for May and Feng Models', fontsize=16)
    # ax.set_xticks(x)
    # ax.set_xticklabels(region_labels, rotation=45, ha="right")
    # ax.legend()
    # ax.bar_label(rects1, padding=3, fmt='%.2f')
    # ax.bar_label(rects2, padding=3, fmt='%.2f')
    # fig.tight_layout()
    
    # bar_chart_path = "dynamic_range_comparison.png"
    # plt.savefig(bar_chart_path)
    # print(f"長條比較圖已保存至: {bar_chart_path}")

    # # B. 分佈直方圖
    # plt.figure(figsize=(10, 6))
    # plt.hist(may_range.flatten().cpu().numpy(), bins=50, alpha=0.7, label='May', density=True)
    # plt.hist(feng_range.flatten().cpu().numpy(), bins=50, alpha=0.7, label='Feng', density=True)
    # # plt.xlabel("正規化後動態範圍值", fontsize=12)
    # plt.xlabel("Normalized Dynamic Range", fontsize=12)
    # # plt.ylabel("機率密度", fontsize=12)
    # plt.ylabel("Probability Density", fontsize=12)
    # # plt.title("May 與 Feng 動態範圍分佈直方圖", fontsize=16)
    # plt.title("Distribution Histogram of Dynamic Range for May and Feng Models", fontsize=16)
    # plt.legend()
    # plt.grid(axis='y', alpha=0.5)
    
    # hist_path = "range_distribution_comparison.png"
    # plt.savefig(hist_path)
    # print(f"分佈直方圖已保存至: {hist_path}")


if __name__ == '__main__':
    run_comparison()