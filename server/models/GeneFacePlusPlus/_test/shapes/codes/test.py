import os
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

import eos
import np2obj

model = loadmat("deep_3drecon/BFM/BFM_model_front.mat")

print(model.keys())
# print all the shapes of keys in the model
for key in model.keys():
    if(isinstance(model[key], np.ndarray)):
        print(key, model[key].shape)



mean_shape = torch.from_numpy(model['meanshape'].transpose()).float()
mean_shape = mean_shape.reshape([-1, 3])
mean_shape = mean_shape - torch.mean(mean_shape, dim=0, keepdims=True)
mean_shape = mean_shape / 10

tri_mask = model['tri']

np2obj.save_obj(mean_shape.numpy(), tri_mask, "_test/shapes/testing_1_out_tri.obj")

print(mean_shape.shape)






# ==============================================================================
# model = loadmat("deep_3drecon/BFM/BFM_model_front.mat")

# mean_shape_np = model['meanshape'].transpose()
# mean_shape_np = mean_shape_np.reshape([-1, 3])

# np2obj.save_obj(mean_shape_np, None, "output.obj")


# import scipy.io as sio
# data = sio.loadmat("deep_3drecon/BFM/BFM_model_front.mat")
# for i in range(0, 10):
#     print(data['tri'][i])