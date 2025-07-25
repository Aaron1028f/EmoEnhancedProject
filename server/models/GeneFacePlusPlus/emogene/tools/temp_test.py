# find index of inner lips in lm68 corresponding to lm478

index_lm68_from_lm478 = [127,234,93,132,58,136,150,176,152,400,379,365,288,361,323,454,356,70,63,105,66,107,336,296,334,293,300,168,197,5,4,75,97,2,326,305,
                         33,160,158,133,153,144,362,385,387,263,373,380,61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178]
index_innerlip_from_lm478 = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

# 68 個關鍵點的索引分組
FACIAL_LANDMARK_REGIONS = {
    'mouth': list(range(48, 68)),      # 嘴巴 (20 points) (48~59: outer, 60~67: inner)
    'right_eyebrow': list(range(17, 22)), # 右眉毛 (5 points)
    'left_eyebrow': list(range(22, 27)),  # 左眉毛 (5 points)
    'right_eye': list(range(36, 42)),     # 右眼 (6 points)
    'left_eye': list(range(42, 48)),      # 左眼 (6 points)
    'nose': list(range(27, 36)),         # 鼻子 (9 points)
    'jaw': list(range(0, 17))            # 下巴輪廓 (17 points)
}

#=======================================
# inner lips
#    61 62 63
# 60          64
#    67 66 65
#=======================================

# find the index both in lips_index and index_lm68_from_lm478

for index_lm68, index_lm478 in enumerate(index_lm68_from_lm478):
    if index_lm478 in index_innerlip_from_lm478:
        # print(f"index_lm68: {index_lm68}, index_lm478: {index_lm478}")
        if index_lm478 in index_innerlip_from_lm478:
            print(f"  inner lip index: {index_lm68} in lm68 corresponds to {index_lm478} in lm478")
        else:
            continue
            # print(f"  not an inner lip index: {index_lm68} in lm68 corresponds to {index_lm478} in lm478")
            
# output:
#   inner lip index: 60 in lm68 corresponds to 78 in lm478
#   inner lip index: 61 in lm68 corresponds to 81 in lm478
#   inner lip index: 62 in lm68 corresponds to 13 in lm478
#   inner lip index: 63 in lm68 corresponds to 311 in lm478
#   inner lip index: 64 in lm68 corresponds to 308 in lm478
#   inner lip index: 65 in lm68 corresponds to 402 in lm478
#   inner lip index: 66 in lm68 corresponds to 14 in lm478
#   inner lip index: 67 in lm68 corresponds to 178 in lm478


# therefore, the center of inner lips
# lm68 ->  lm478
# 62   ->  13
# 66   ->  14

# dimension of y in 2D(cano_lm2d), is dimension of z in 3D(cano_lm3d)
