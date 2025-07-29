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

# for index_lm68, index_lm478 in enumerate(index_lm68_from_lm478):
#     if index_lm478 in index_innerlip_from_lm478:
#         # print(f"index_lm68: {index_lm68}, index_lm478: {index_lm478}")
#         if index_lm478 in index_innerlip_from_lm478:
#             print(f"  inner lip index: {index_lm68} in lm68 corresponds to {index_lm478} in lm478")
#         else:
#             continue
#             # print(f"  not an inner lip index: {index_lm68} in lm68 corresponds to {index_lm478} in lm478")
            
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

# ==============================================

for index_lm68, index_lm478 in enumerate(index_lm68_from_lm478):
    print(f"index_lm68: {index_lm68}, index_lm478: {index_lm478}")


# output
# jaw
# index_lm68: 0, index_lm478: 127
# index_lm68: 1, index_lm478: 234
# index_lm68: 2, index_lm478: 93
# index_lm68: 3, index_lm478: 132
# index_lm68: 4, index_lm478: 58
# index_lm68: 5, index_lm478: 136
# index_lm68: 6, index_lm478: 150
# index_lm68: 7, index_lm478: 176
# index_lm68: 8, index_lm478: 152
# index_lm68: 9, index_lm478: 400
# index_lm68: 10, index_lm478: 379
# index_lm68: 11, index_lm478: 365
# index_lm68: 12, index_lm478: 288
# index_lm68: 13, index_lm478: 361
# index_lm68: 14, index_lm478: 323
# index_lm68: 15, index_lm478: 454
# index_lm68: 16, index_lm478: 356

# right eyebrow
# index_lm68: 17, index_lm478: 70
# index_lm68: 18, index_lm478: 63
# index_lm68: 19, index_lm478: 105
# index_lm68: 20, index_lm478: 66
# index_lm68: 21, index_lm478: 107

# left eyebrow
# index_lm68: 22, index_lm478: 336
# index_lm68: 23, index_lm478: 296
# index_lm68: 24, index_lm478: 334
# index_lm68: 25, index_lm478: 293
# index_lm68: 26, index_lm478: 300


# nose
# index_lm68: 27, index_lm478: 168
# index_lm68: 28, index_lm478: 197
# index_lm68: 29, index_lm478: 5
# index_lm68: 30, index_lm478: 4
# index_lm68: 31, index_lm478: 75
# index_lm68: 32, index_lm478: 97
# index_lm68: 33, index_lm478: 2
# index_lm68: 34, index_lm478: 326
# index_lm68: 35, index_lm478: 305


# right eye
# index_lm68: 36, index_lm478: 33
# index_lm68: 37, index_lm478: 160
# index_lm68: 38, index_lm478: 158
# index_lm68: 39, index_lm478: 133
# index_lm68: 40, index_lm478: 153
# index_lm68: 41, index_lm478: 144

# left eye
# index_lm68: 42, index_lm478: 362
# index_lm68: 43, index_lm478: 385
# index_lm68: 44, index_lm478: 387
# index_lm68: 45, index_lm478: 263
# index_lm68: 46, index_lm478: 373
# index_lm68: 47, index_lm478: 380


# mouth ()
# index_lm68: 48, index_lm478: 61
# index_lm68: 49, index_lm478: 40
# index_lm68: 50, index_lm478: 37
# index_lm68: 51, index_lm478: 0
# index_lm68: 52, index_lm478: 267
# index_lm68: 53, index_lm478: 270
# index_lm68: 54, index_lm478: 291
# index_lm68: 55, index_lm478: 321
# index_lm68: 56, index_lm478: 314
# index_lm68: 57, index_lm478: 17
# index_lm68: 58, index_lm478: 84
# index_lm68: 59, index_lm478: 91
# index_lm68: 60, index_lm478: 78
# index_lm68: 61, index_lm478: 81
# index_lm68: 62, index_lm478: 13
# index_lm68: 63, index_lm478: 311
# index_lm68: 64, index_lm478: 308
# index_lm68: 65, index_lm478: 402
# index_lm68: 66, index_lm478: 14
# index_lm68: 67, index_lm478: 178
