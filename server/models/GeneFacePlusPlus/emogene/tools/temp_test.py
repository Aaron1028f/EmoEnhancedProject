import numpy as np
from one_euro_filter import OneEuroFilter

t0 = 0  # 初始时间
x0 = 0  # 初始值
filter = OneEuroFilter(t0, x0)

t = 1  # 当前时间
x = 1  # 当前信号值
smoothed_value = filter(t, x)


print(f"Smoothed value at time {t}: {smoothed_value}")
