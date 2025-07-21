# python emogene/experiment/visualize_2dlandmark/vis_cano_lm_ds.py

import os
import sys
sys.path.append('./')

import numpy as np
import torch
import cv2
from utils.commons.tensor_utils import move_to_cuda, convert_to_np, convert_to_tensor

from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
import copy

def vis_cano_lm3d_to_imgs(cano_lm3d, cano_lm3d_clamped=None, hw=512, color_label='red'):
    # ============== bs_ver_modified ==============
    color = (255, 0, 0) # default red
    if color_label == 'red':
        color = (255, 0, 0)
    elif color_label == 'green':
        color = (0, 255, 0)
    elif color_label == 'blue':
        color = (0, 0, 255)        
    # load img(.png) background
    img_bg = load_img_to_512_hwc_array('emogene/experiment/visualize_2dlandmark/data/may_cano_lm3d_img.png')
    if img_bg is not None:
        img_bg = cv2.resize(img_bg, (hw, hw), interpolation=cv2.INTER_LINEAR)
        img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
        img_bg = img_bg.astype(np.uint8)
    else:
        img_bg = np.ones([hw, hw, 3], dtype=np.uint8) * 255
    # ============== bs_ver_modified ==============
    # preprocess cano_lm3d_clamped
    color_clamped = (0, 0, 255) # red for clamped
    if cano_lm3d_clamped is not None:
        cano_lm3d_clamped_ = cano_lm3d_clamped[:1, ].repeat([len(cano_lm3d),1,1])
        cano_lm3d_clamped_[:, 17:27] = cano_lm3d[:, 17:27] # brow
        cano_lm3d_clamped_[:, 36:48] = cano_lm3d[:, 36:48] # eye
        cano_lm3d_clamped_[:, 27:36] = cano_lm3d[:, 27:36] # nose
        cano_lm3d_clamped_[:, 48:68] = cano_lm3d[:, 48:68] # mouth
        cano_lm3d_clamped_[:, 0:17] = cano_lm3d[:, :17] # yaw
        
        cano_lm3d_clamped = cano_lm3d_clamped_
        cano_lm3d_clamped = convert_to_np(cano_lm3d_clamped)
        cano_lm3d_clamped = (cano_lm3d_clamped * hw/2 + hw/2).astype(int)
    
    
    # ============== bs_ver_modified ==============
    cano_lm3d_ = cano_lm3d[:1, ].repeat([len(cano_lm3d),1,1])
    cano_lm3d_[:, 17:27] = cano_lm3d[:, 17:27] # brow
    cano_lm3d_[:, 36:48] = cano_lm3d[:, 36:48] # eye
    cano_lm3d_[:, 27:36] = cano_lm3d[:, 27:36] # nose
    cano_lm3d_[:, 48:68] = cano_lm3d[:, 48:68] # mouth
    cano_lm3d_[:, 0:17] = cano_lm3d[:, :17] # yaw
    
    cano_lm3d = cano_lm3d_

    cano_lm3d = convert_to_np(cano_lm3d)

    WH = hw
    cano_lm3d = (cano_lm3d * WH/2 + WH/2).astype(int)
    frame_lst = []
    for i_img in range(len(cano_lm3d)):
        # lm2d = cano_lm3d[i_img ,:, 1:] # [68, 2]
        lm2d = cano_lm3d[i_img ,:, :2] # [68, 2]
        # img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
        img = copy.deepcopy(img_bg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # flip back
        img = cv2.flip(img, 0)
        
        if cano_lm3d_clamped is not None:
            lm2d_clamped = cano_lm3d_clamped[i_img ,:, :2]
        
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            # color = (255,0,0)
            img = cv2.circle(img, center=(x,y), radius=1, color=color, thickness=-1)
            if cano_lm3d_clamped is not None:
                x_clamped, y_clamped = lm2d_clamped[i]
                img = cv2.circle(img, center=(x_clamped,y_clamped), radius=1, color=color_clamped, thickness=-1)
                # img = cv2.putText(img, f"{i}", org=(x_clamped,y_clamped), fontFace=font, fontScale=0.3, color=(0,255,0))
                if x!= x_clamped or y != y_clamped:
                    print(f'original {i}: ({x}, {y})')
                    print(f"clamped {i}: ({x_clamped}, {y_clamped})")
            font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.flip(img, 0)
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            y = WH - y
            # img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=(255,0,0))
            img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=color)
            
        frame_lst.append(img)
    return frame_lst


def vis_cano_lm3d_to_imgs_for_exp(cano_lm3d, cano_lm3d_clamped=None, hw=512, color_label='red'):
    # ============== bs_ver_modified ==============
    color = (255, 0, 0) # default red
    if color_label == 'red':
        color = (255, 0, 0)
    elif color_label == 'green':
        color = (0, 255, 0)
    elif color_label == 'blue':
        color = (0, 0, 255)        
    # ============== bs_ver_modified ==============
    # color = ( 0, 0,255) # default red
    color = (241, 238, 187) # light blue
    color_clamped = (0, 255, 0) # green for clamped
    
    WH = hw
    # preprocess cano_lm3d_clamped
    if cano_lm3d_clamped is not None:
        cano_lm3d_clamped_ = cano_lm3d_clamped[:1, ].repeat([len(cano_lm3d),1,1])
        cano_lm3d_clamped_[:, 17:27] = cano_lm3d[:, 17:27] # brow
        cano_lm3d_clamped_[:, 36:48] = cano_lm3d[:, 36:48] # eye
        cano_lm3d_clamped_[:, 27:36] = cano_lm3d[:, 27:36] # nose
        cano_lm3d_clamped_[:, 48:68] = cano_lm3d[:, 48:68] # mouth
        cano_lm3d_clamped_[:, 0:17] = cano_lm3d[:, :17] # yaw
        
        cano_lm3d_clamped = cano_lm3d_clamped_
        cano_lm3d_clamped = convert_to_np(cano_lm3d_clamped)
        
        cano_lm3d_clamped = (cano_lm3d_clamped * WH/2 + WH/2).astype(int)

    # preprocess cano_lm3d
    cano_lm3d_ = cano_lm3d[:1, ].repeat([len(cano_lm3d),1,1])
    cano_lm3d_[:, 17:27] = cano_lm3d[:, 17:27] # brow
    cano_lm3d_[:, 36:48] = cano_lm3d[:, 36:48] # eye
    cano_lm3d_[:, 27:36] = cano_lm3d[:, 27:36] # nose
    cano_lm3d_[:, 48:68] = cano_lm3d[:, 48:68] # mouth
    cano_lm3d_[:, 0:17] = cano_lm3d[:, :17] # yaw
    
    cano_lm3d = cano_lm3d_

    cano_lm3d = convert_to_np(cano_lm3d)

    # WH = hw
    cano_lm3d = (cano_lm3d * WH/2 + WH/2).astype(int)
    # frame_lst = []
    # =====================================================
    
    img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
    
    for i_img in range(len(cano_lm3d)):
        # lm2d = cano_lm3d[i_img ,:, 1:] # [68, 2]
        lm2d = cano_lm3d[i_img ,:, :2] # [68, 2]
        # img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
        if cano_lm3d_clamped is not None:
            lm2d_clamped = cano_lm3d_clamped[i_img ,:, :2]
        
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            # color = (255,0,0)
            img = cv2.circle(img, center=(x,y), radius=1, color=color, thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            if cano_lm3d_clamped is not None:
                x_clamped, y_clamped = lm2d_clamped[i]
                img = cv2.circle(img, center=(x_clamped,y_clamped), radius=1, color=color_clamped, thickness=-1)
                # img = cv2.putText(img, f"{i}", org=(x_clamped,y_clamped), fontFace=font, fontScale=0.3, color=(0,255,0))
            
        # img = cv2.flip(img, 0)
        # for i in range(len(lm2d)):
        #     x, y = lm2d[i]
        #     y = WH - y
        #     # img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=(255,0,0))
        #     img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=color)
            
        # frame_lst.append(img)
    # return frame_lst
    
    img = cv2.flip(img, 0)
    return img


def vis_area_of_all_lm3d_in_ds():
    # load dataset and helper
    # from tasks.radnerfs.dataset_utils import RADNeRFDataset, get_boundary_mask, dilate_boundary_mask, get_lf_boundary_mask
    # dataset_cls = RADNeRFDataset
    # dataset = dataset_cls('trainval', training=False)
    
    from emogene.tools.face3d_helper_bs import Face3DHelper
    face3d_helper = Face3DHelper(keypoint_mode='mediapipe', use_gpu=True)
    
    # get idexp_lm3d
    # load numpy file
    id_ds_numpy = np.load('emogene/experiment/visualize_2dlandmark/data/feng_id_ds.npy')
    exp_ds_numpy = np.load('emogene/experiment/visualize_2dlandmark/data/feng_exp_ds.npy')
    
    id_ds = torch.from_numpy(id_ds_numpy).float().cuda()
    exp_ds = torch.from_numpy(exp_ds_numpy).float().cuda()

    idexp_lm3d_ds = face3d_helper.reconstruct_idexp_lm3d(id_ds, exp_ds)
    idexp_lm3d = idexp_lm3d_ds.clone()
    
    idexp_lm3d_mean = idexp_lm3d_ds.mean(dim=0, keepdim=True)
    idexp_lm3d_std = idexp_lm3d_ds.std(dim=0, keepdim=True)
    
    from utils.commons.hparams import hparams, set_hparams
    if hparams.get("normalize_cond", True):
        idexp_lm3d_ds_normalized = (idexp_lm3d_ds - idexp_lm3d_mean) / idexp_lm3d_std
    else:
        idexp_lm3d_ds_normalized = idexp_lm3d_ds
        
    # lower = torch.quantile(idexp_lm3d_ds_normalized, q=0.03, dim=0)
    # upper = torch.quantile(idexp_lm3d_ds_normalized, q=0.97, dim=0)
    lower = torch.quantile(idexp_lm3d_ds_normalized, q=0.20, dim=0)
    upper = torch.quantile(idexp_lm3d_ds_normalized, q=0.80, dim=0)    

    # idexp_lm3d_ds_lle = idexp_lm3d_ds[:, index_lm68_from_lm478].reshape([-1, 68*3])
    # idexp_lm3d = idexp_lm3d.reshape([-1, 68, 3])
    
    # print('idexp_lm3d_ds_normalized.shape:', idexp_lm3d_ds_normalized.shape)
    # print('lower:', lower.shape, lower)
    # print('upper:', upper.shape, upper)
    # print('idexp_lm3d_mean.shape:', idexp_lm3d_mean.shape)
    # print('idexp_lm3d_std.shape:', idexp_lm3d_std.shape)
    # print('idexp_lm3d_ds:', idexp_lm3d_ds.shape)
    
    from emogene.tools.face_landmarker_bs import index_lm68_from_lm478, index_lm131_from_lm478
    
    idexp_lm3d = idexp_lm3d[:, index_lm68_from_lm478]
    idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm68_from_lm478]
    idexp_lm3d_std = idexp_lm3d_std[:, index_lm68_from_lm478]
    lower = lower[index_lm68_from_lm478]
    upper = upper[index_lm68_from_lm478]
    
    print('idexp_lm3d.shape:', idexp_lm3d.shape)
    print('idexp_lm3d_mean.shape:', idexp_lm3d_mean.shape)
    print('idexp_lm3d_std.shape:', idexp_lm3d_std.shape)
    print('lower.shape:', lower.shape)
    print('upper.shape:', upper.shape)
    
    
    idexp_lm3d = idexp_lm3d.reshape([-1, 68, 3])
    idexp_lm3d_normalized = (idexp_lm3d - idexp_lm3d_mean) / idexp_lm3d_std

    # 計算原始的cano_lm3d (不經過clamp，實際推論出來的cano_lm3d，但還未經過修正)
    cano_lm3d = (idexp_lm3d_mean + idexp_lm3d_std * idexp_lm3d_normalized) / 10 + face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0) # 轉換成座標，方便投影
    print('cano_lm3d.shape:', cano_lm3d.shape)

    idexp_lm3d_normalized = ((cano_lm3d - face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)) * 10 - idexp_lm3d_mean) / idexp_lm3d_std # 恢復成變化量
    idexp_lm3d_normalized_clamped = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)

    # 計算經過clamp後的cano_lm3d
    cano_lm3d_clamped = (idexp_lm3d_mean + idexp_lm3d_std * idexp_lm3d_normalized_clamped) / 10 + face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0) # 轉換成座標，方便投影
    print('cano_lm3d_clamped.shape:', cano_lm3d_clamped.shape)

    # ===================================================================================
    # draw the cano_lm3d to image
    cano_lm3d_img = vis_cano_lm3d_to_imgs_for_exp(cano_lm3d, cano_lm3d_clamped=cano_lm3d_clamped, hw=512, color_label='red')
    
    cano_lm3d_img = cv2.resize(cano_lm3d_img, (512, 512))
    
    # save the image
    filename = 'feng_cano_lm3d_img_clamped'
    save_path = f'emogene/experiment/visualize_2dlandmark/data/{filename}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cano_lm3d_img)
    print(f'Saved cano_lm3d image to {save_path}')
    

if __name__ == '__main__':
    vis_area_of_all_lm3d_in_ds()
