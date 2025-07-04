# import pysnooper
# @pysnooper.snoop(output='prepare_batch_debug.log')    
# ====================================
import os
import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import uuid
import traceback
# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_np, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.utils.mp_feature_extractors.face_landmarker import index_lm68_from_lm478, index_lm131_from_lm478
# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from utils.commons.pitch_utils import f0_to_coarse
# Dataset Related
from tasks.radnerfs.dataset_utils import RADNeRFDataset, get_boundary_mask, dilate_boundary_mask, get_lf_boundary_mask
# Method Related
from modules.audio2motion.vae import VAEModel, PitchContourVAEModel
from modules.postnet.lle import compute_LLE_projection, find_k_nearest_neighbors
from modules.radnerfs.utils import get_audio_features, get_rays, get_bg_coords, convert_poses, nerf_matrix_to_ngp
from modules.radnerfs.radnerf import RADNeRF
from modules.radnerfs.radnerf_sr import RADNeRFwithSR
from modules.radnerfs.radnerf_torso import RADNeRFTorso
from modules.radnerfs.radnerf_torso_sr import RADNeRFTorsowithSR


face3d_helper = None
def vis_cano_lm3d_to_imgs(cano_lm3d, hw=512):
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
        img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
        
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            color = (255,0,0)
            img = cv2.circle(img, center=(x,y), radius=3, color=color, thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.flip(img, 0)
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            y = WH - y
            img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=(255,0,0))
        frame_lst.append(img)
    return frame_lst

def inject_blink_to_lm68(lm68, opened_eye_area_percent=0.6, closed_eye_area_percent=0.15):
    # [T, 68, 2]
    # lm68[:,36:48] = lm68[0:1,36:48].repeat([len(lm68), 1, 1])
    opened_eye_lm68 = copy.deepcopy(lm68)
    eye_area_percent = 0.6 * torch.ones([len(lm68), 1], dtype=opened_eye_lm68.dtype, device=opened_eye_lm68.device)

    eye_open_scale = (opened_eye_lm68[:, 41, 1] - opened_eye_lm68[:, 37, 1]) + (opened_eye_lm68[:, 40, 1] - opened_eye_lm68[:, 38, 1]) + (opened_eye_lm68[:, 47, 1] - opened_eye_lm68[:, 43, 1]) + (opened_eye_lm68[:, 46, 1] - opened_eye_lm68[:, 44, 1])
    eye_open_scale = eye_open_scale.abs()
    idx_largest_eye = eye_open_scale.argmax()
    # lm68[:, list(range(17,27))] = lm68[idx_largest_eye:idx_largest_eye+1, list(range(17,27))].repeat([len(lm68),1,1])
    # lm68[:, list(range(36,48))] = lm68[idx_largest_eye:idx_largest_eye+1, list(range(36,48))].repeat([len(lm68),1,1])
    lm68[:,[37,38,43,44],1] = lm68[:,[37,38,43,44],1] + 0.03
    lm68[:,[41,40,47,46],1] = lm68[:,[41,40,47,46],1] - 0.03
    closed_eye_lm68 = copy.deepcopy(lm68)
    closed_eye_lm68[:,37] = closed_eye_lm68[:,41] = closed_eye_lm68[:,36] * 0.67 + closed_eye_lm68[:,39] * 0.33
    closed_eye_lm68[:,38] = closed_eye_lm68[:,40] = closed_eye_lm68[:,36] * 0.33 + closed_eye_lm68[:,39] * 0.67
    closed_eye_lm68[:,43] = closed_eye_lm68[:,47] = closed_eye_lm68[:,42] * 0.67 + closed_eye_lm68[:,45] * 0.33
    closed_eye_lm68[:,44] = closed_eye_lm68[:,46] = closed_eye_lm68[:,42] * 0.33 + closed_eye_lm68[:,45] * 0.67
    
    T = len(lm68)
    period = 100
    # blink_factor_lst = np.array([0.4, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.4]) # * 0.9
    # blink_factor_lst = np.array([0.4, 0.7, 0.8, 1.0, 0.8, 0.6, 0.4]) # * 0.9
    blink_factor_lst = np.array([0.1, 0.5, 0.7, 1.0, 0.7, 0.5, 0.1]) # * 0.9
    dur = len(blink_factor_lst)
    for i in range(T):
        if (i + 25) % period == 0:
            for j in range(dur):
                idx = i+j
                if idx > T - 1: # prevent out of index error
                    break
                blink_factor = blink_factor_lst[j]
                lm68[idx, 36:48] = lm68[idx, 36:48] * (1-blink_factor) + closed_eye_lm68[idx, 36:48] * blink_factor
                eye_area_percent[idx] = opened_eye_area_percent * (1-blink_factor) + closed_eye_area_percent * blink_factor
    return lm68, eye_area_percent


class GeneFace2Infer:
    def __init__(self, audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.audio2secc_dir = audio2secc_dir # the directory of the audio2secc model
        self.postnet_dir = postnet_dir # the directory of the postnet model
        self.head_model_dir = head_model_dir # the directory of the head model
        self.torso_model_dir = torso_model_dir # the directory of the torso model
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir) # load the audio2secc model
        self.postnet_model = self.load_postnet(postnet_dir)
        self.secc2video_model = self.load_secc2video(head_model_dir, torso_model_dir) # load the secc2video model
        
        
        self.audio2secc_model.to(device).eval()
        # device is the device that the model is loaded on
        # device is 'cuda' if torch.cuda.is_available() is True
        # eval() is used to set the model to evaluation mode
        # evaluation mode is used to set the model to inference mode
        if self.postnet_model is not None:
            self.postnet_model.to(device).eval()
        self.secc2video_model.to(device).eval()
        self.seg_model = MediapipeSegmenter() # load the mediapipe segmenter
        # the mediapipe segmenter is used to segment the face
        # the mediapipe segmenter is used to get the face mask

        self.secc_renderer = SECC_Renderer(512)
        # SECC_Renderer is used to render the SECC
        # SECC means the speech-to-expression-to-camera-control
        self.face3d_helper = Face3DHelper(keypoint_mode='mediapipe', use_gpu=True)
        # Face3DHelper is used to help the 3D face reconstruction
        
        hparams['infer_smooth_camera_path_kernel_size'] = 7
        # hparams is a dict that contains the hyperparameters of the model
        # hparams['infer_smooth_camera_path_kernel_size'] is the kernel size of the smooth camera path
        # hparams is defined in the hparams.py file in the utils/commons folder
        # hparams is used to set the hyperparameters of the model

    def load_audio2secc(self, audio2secc_dir):
        set_hparams(f"{os.path.dirname(audio2secc_dir) if os.path.isfile(audio2secc_dir) else audio2secc_dir}/config.yaml")
        # set_hparams is used to load the hyperparameters of the audio2secc model
        # the hyperparameters are stored in the config.yaml file
        self.audio2secc_hparams = copy.deepcopy(hparams) # copy the hyperparameters
        # hparams is a dict that contains the hyperparameters of the audio2secc model
        
        # copy.deepcopy is used to copy the hyperparameters
        # the difference between copy and deepcopy is that deepcopy will copy the content of the object, 
        # while copy will only copy the reference of the object
        # object copy is used to avoid the modification of the original object
        if hparams["motion_type"] == 'id_exp':
            self.in_out_dim = 80 + 64
        # if the motion type is id_exp, the in_out_dim is 80 + 64
        # 80 is the dimension of the identity, 64 is the dimension of the expression
        # in out dim is the dimension of the input and output of the audio2secc model
        # in_out_dim is calculated based on the motion type
        # if the motion type is id_exp, the in_out_dim is 80 + 64
        # if the motion type is exp, the in_out_dim is 64    
        
        elif hparams["motion_type"] == 'exp':
            self.in_out_dim = 64
        audio_in_dim = 1024
        # audio_in_dim is the dimension of the audio input
        # audio_in_dim is 1024
        # audio_in_dim is used to calculate the dimension of the audio input
        
        # the method of get() means to get the value of the key in the dict
        # if the key is not in the dict, the default value is returned
        # if the key is in the dict, the value of the key is returned
        
        # the default value of the dict hparams is the value in the config.yaml file
        if hparams.get("use_pitch", False) is True:
            self.model = PitchContourVAEModel(hparams, in_out_dim=self.in_out_dim, audio_in_dim=audio_in_dim)
            # PitchContourVAEModel and VAEModel differs in the input dimension
            # PitchContourVAEModel has an additional input dimension of pitch
            # VAEModel does not have the input dimension of pitch
            
        else:
            self.model = VAEModel(in_out_dim=self.in_out_dim, audio_in_dim=audio_in_dim)
        load_ckpt(self.model, f"{audio2secc_dir}", model_name='model', strict=True)
        # load_ckpt is used to load the checkpoint of the model
        # where is load_ckpt defined?
        # load_ckpt is defined in the ckpt_utils.py file
        # in the directory of the utils/commons folder
        
        # checkpoint of a model means the model is saved at a certain point
        # the checkpoint is saved as a file
        # the checkpoint contains the weights of the model
        # the checkpoint is used to save the model
        # the checkpoint is used to load the model
        # the checkpoint is used to resume the training of the model
        # the checkpoint is used to save the training progress of the model
        return self.model

    def load_postnet(self, postnet_dir):
        # load the postnet model
        # the postnet model is used to refine the predicted motion
        # postnet means post-processing network
        
        if postnet_dir == '':
            return None
        from modules.postnet.models import PitchContourCNNPostNet
        set_hparams(f"{os.path.dirname(postnet_dir) if os.path.isfile(postnet_dir) else postnet_dir}/config.yaml")
        self.postnet_hparams = copy.deepcopy(hparams)
        in_out_dim = 68*3
        pitch_dim = 128
        self.model = PitchContourCNNPostNet(in_out_dim=in_out_dim, pitch_dim=pitch_dim)
        load_ckpt(self.model, f"{postnet_dir}", steps=20000, model_name='model', strict=True)
        return self.model

    def load_secc2video(self, head_model_dir, torso_model_dir):

        if torso_model_dir != '':
            set_hparams(f"{os.path.dirname(torso_model_dir) if os.path.isfile(torso_model_dir) else torso_model_dir}/config.yaml")
            self.secc2video_hparams = copy.deepcopy(hparams)
            if hparams.get("with_sr"):
                model = RADNeRFTorsowithSR(hparams)
            else:
                model = RADNeRFTorso(hparams)
            load_ckpt(model, f"{torso_model_dir}", model_name='model', strict=True)
        else:
            set_hparams(f"{os.path.dirname(head_model_dir) if os.path.isfile(head_model_dir) else head_model_dir}/config.yaml")
            self.secc2video_hparams = copy.deepcopy(hparams)
            if hparams.get("with_sr"):
                model = RADNeRFwithSR(hparams)
            else:
                model = RADNeRF(hparams)
            load_ckpt(model, f"{head_model_dir}", model_name='model', strict=True)
        self.dataset_cls = RADNeRFDataset # the dataset only provides head pose 
        # RADNeRFDataset is the dataset class of the RADNeRF model
        # RADNeRFDataset is used to load the dataset of the RADNeRF model
        # RADNeRFDataset is used to prepare the dataset of the RADNeRF model
        
        # radnerf is a model that generates the video from the audio
        # RAD-NeRF（Real-time Audio-spatial Decomposed Neural Radiance Fields）    
            
        self.dataset = self.dataset_cls('trainval', training=False)
        eye_area_percents = torch.tensor(self.dataset.eye_area_percents)
        self.closed_eye_area_percent = torch.quantile(eye_area_percents, q=0.03).item()
        self.opened_eye_area_percent = torch.quantile(eye_area_percents, q=0.97).item()
        try:
            model = torch.compile(model)
        except:
            traceback.print_exc()
        return model

    def infer_once(self, inp):
        self.inp = inp
        # inp is a dict that contains the input information

        samples = self.prepare_batch_from_inp(inp)
        # samples is a dict that contains the condition feature of NeRF
        # samples is used to prepare the batch from the input
        # samples contains the information of the input
        
        out_name = self.forward_system(samples, inp)
        return out_name
    def prepare_batch_from_inp(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a dict that contains the condition feature of NeRF
        """
        sample = {} # sample is a dict that contains the condition feature of NeRF
        # batch
        
        # Process Audio
        self.save_wav16k(inp['drv_audio_name']) # save the audio file in the 16k format 
        hubert = self.get_hubert(self.wav16k_name) # get the hubert from the 16k audio file
        # hubert is the mel spectrogram of the audio file
        # mel spectrogram is a representation of the audio file
        # mel spectrogram is used to represent the audio file in the frequency domain
        # hubert has the data type of numpy.ndarray
        # hubert has the shape of [T, 80]
        # T is the number of frames, 80 is the dimension of the mel spectrogram
        # a matrix to reresent hubert is like:
        # [[1, 2, 3, ..., 80],
        #  [1, 2, 3, ..., 80],
        #  [1, 2, 3, ..., 80],
        #  ...]
        # shape[0] means the number of frames or the number of rows
        
        t_x = hubert.shape[0] # t_x is the number of frames
        x_mask = torch.ones([1, t_x]).float() # mask for audio frames
        # x_mask is a mask for the audio frames
        # x_mask is a matrix that contains ones
        # x_mask has the data type of torch.Tensor
        # x_mask has the shape of [1, T]
        # T is the number of frames
        # x_mask is used to mask the audio frames
        # mask the audio frames means to filter the audio frames
        # mask the audio frames means to select the audio frames
        
        y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames
        # y_mask is a mask for the motion/image frames
        # y_mask is a matrix that contains ones
        # y_mask has the data type of torch.Tensor
        # y_mask has the shape of [1, T//2]
        # T is the number of frames
        # y_mask is used to mask the motion/image frames
        # mask the motion/image frames means to filter the motion/image frames
        
        f0 = self.get_f0(self.wav16k_name) # get the f0 from the 16k audio file
        # f0 is the fundamental frequency of the audio file
        # f0 has the data type of numpy.ndarray
        # f0 has the shape of [T, 1]
        # T is the number of frames
        # 1 is the dimension of the fundamental frequency
        # a matrix to represent f0 is like:
        # [[1],
        #  [1],
        #  [1],
        #  ...]
        # shape[0] means the number of frames or the number of rows
        
        if f0.shape[0] > len(hubert): # if the number of frames of f0 is greater than the number of frames of hubert
            # len(hubert) means the number of frames of hubert
            f0 = f0[:len(hubert)]
            # f0 is the first len(hubert) frames of f0
            # f0 have the shpae of [len(hubert), 1] here
            
        else:
            num_to_pad = len(hubert) - len(f0) # the number of frames to pad
            # num_to_pad is the difference between the number of frames of hubert and the number of frames of f0
            # num_to_pad is the number of frames to pad
            f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))
            # pad the f0 with zeros
            # pad the f0 with zeros means to add zeros to the f0
            # pad the f0 with zeros means to increase the number of frames of f0
            # pad the f0 with zeros means to make the number of frames of f0 equal to the number of frames of hubert
            # pad the f0 with zeros means to make the shape of f0 equal to the shape of hubert
            # np.pad() is a function to pad the array
            # np.pad() is a function in the numpy library
            # the way to use np.pad() is np.pad(array, pad_width, mode='constant', constant_values=0)
            # pad_width=((0,num_to_pad), (0,0)) means to pad the array with zeros
            # (0,num_to_pad) means to pad the array with zeros at the beginning and the end
            # (0,0) means to pad the array with zeros at the beginning and the end
            # pad the array with zeros at the beginning and the end means to add zeros to the array at the beginning and the end
            
        # Process Pose
        sample.update({
            'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
            'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
            'x_mask': x_mask.cuda(),
            'y_mask': y_mask.cuda(),
            })
        # .unsqueeze(0) means to add a dimension to the tensor
        # .unsqueeze(0) means to add a dimension to the tensor at the beginning
        # unsqueeze hubert means to add a dimension to the hubert
        # hubert before unsqueeze has the shape of [T, 80]
        # hubert after unsqueeze has the shape of [1, T, 80]
        # 1 is the dimension of the batch size
        # T is the number of frames
        # 80 is the dimension of the mel spectrogram
        # .cuda() means to move the tensor to the cuda device
        # .cuda() means to move the tensor to the GPU
        # .cuda() is used to move the tensor to the GPU
        # .reshape([1,-1]) means to reshape the tensor
        # .reshape([1,-1]) means to reshape the tensor to have 1 row and -1 columns
        # -1 means the number of columns is inferred from the number of elements in the tensor
        
        sample['audio'] = sample['hubert']
        sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()
        # sample['blink'] is a tensor that contains the blink information
        # sample['blink'] is a tensor that contains zeros
        # sample['blink'] has the data type of torch.Tensor
        # sample['blink'] has the shape of [1, T, 1]
        # 1 is the dimension of the batch size
        # T is the number of frames
        # 1 is the dimension of the blink information
        # sample['blink'] represent as a matrix
        # [[0],
        #  [0],
        #  [0],
        #  ...]
        # shape[0] means the number of frames or the number of rows

        sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
        # sample['eye_amp'] is a tensor that contains the eye amplitude
        # sample['eye_amp'] is a tensor that contains ones
        # sample['eye_amp'] has the data type of torch.Tensor
        # sample['eye_amp'] has the shape of [1, 1]
        # 1 is the dimension of the batch size
        # 1 is the dimension of the eye amplitude
        # *1.0 means to multiply the tensor by 1.0
        # *1.0 means to multiply the tensor by 1.0 to keep the tensor unchanged
        
        sample['mouth_amp'] = torch.ones([1, 1]).cuda() * float(inp['mouth_amp'])
        # sample['id'] = torch.tensor(self.dataset.ds_dict['id'][0:t_x//2]).cuda()
        sample['id'] = torch.tensor(self.dataset.ds_dict['id'][0:1]).cuda().repeat([t_x//2, 1])
        # sample['id'] is a tensor that contains the identity information
        # sample['id'] is a tensor that contains the identity information of the person
        # self.dataset.ds_dict['id'][0:1]).cuda().repeat([t_x//2, 1]) is the identity information of the person
        # self.dataset.ds_dict['id'][0:1] is the identity information of the person
        # self.dataset.dict is a dict that contains the identity information of the person
        # .cuda() means to move the tensor to the cuda device
        # .repeat([t_x//2, 1]) means to repeat the tensor
        # .repeat([t_x//2, 1]) means to repeat the tensor t_x//2 times along the first dimension
        # .repeat([t_x//2, 1]) means to repeat the tensor 1 time along the second dimension
        # .repeat([t_x//2, 1]) means to repeat the tensor along the first dimension and the second dimension
        
        pose_lst = [] # pose_lst is a list that contains the pose information
        euler_lst = [] # euler_lst is a list that contains the euler information
        trans_lst = [] # trans_lst is a list that contains the translation information
        # translation information is the information of the translation of the camera
        rays_o_lst = [] # rays_o_lst is a list that contains the rays_o information
        # rays_o information is the information of the origin of the rays
        rays_d_lst = [] # rays_d_lst is a list that contains the rays_d information
        # rays_d information is the information of the direction of the rays
        

        if '-' in inp['drv_pose']: # if the driver pose is a range
            start_idx, end_idx = inp['drv_pose'].split("-") # split the driver pose by '-'
            # inp is a dict that contains the input information
            # dict.split() is a method to split the dict
            # dict.split() returns a list of strings after splitting the dict
            # start_idx is the start index of the driver pose
            # end_idx is the end index of the driver pose
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            ds_ngp_pose_lst = [self.dataset.poses[i].unsqueeze(0) for i in range(start_idx, end_idx)]
            # ds_ngp_pose_lst is a list that contains the ngp pose information
            # ngp means the next generation of the person
            ds_euler_lst = [torch.tensor(self.dataset.ds_dict['euler'][i]) for i in range(start_idx, end_idx)]
            ds_trans_lst = [torch.tensor(self.dataset.ds_dict['trans'][i]) for i in range(start_idx, end_idx)]

        for i in range(t_x//2):
            if inp['drv_pose'] == 'static':
                ngp_pose = self.dataset.poses[0].unsqueeze(0)
                euler = torch.tensor(self.dataset.ds_dict['euler'][0])
                trans = torch.tensor(self.dataset.ds_dict['trans'][0])
            elif '-' in inp['drv_pose']:
                mirror_idx = mirror_index(i, len(ds_ngp_pose_lst))
                ngp_pose = ds_ngp_pose_lst[mirror_idx]
                euler = ds_euler_lst[mirror_idx]
                trans = ds_trans_lst[mirror_idx]
            else:
                ngp_pose = self.dataset.poses[i].unsqueeze(0)
                euler = torch.tensor(self.dataset.ds_dict['euler'][i])
                trans = torch.tensor(self.dataset.ds_dict['trans'][i])

            rays = get_rays(ngp_pose.cuda(), self.dataset.intrinsics, self.dataset.H, self.dataset.W, N=-1)
            rays_o_lst.append(rays['rays_o'].cuda())
            rays_d_lst.append(rays['rays_d'].cuda())
            pose = convert_poses(ngp_pose).cuda()
            pose_lst.append(pose)
            euler_lst.append(euler)
            trans_lst.append(trans)
        sample['rays_o'] = rays_o_lst
        sample['rays_d'] = rays_d_lst
        sample['poses'] = pose_lst
        sample['euler'] = torch.stack(euler_lst).cuda()
        sample['trans'] = torch.stack(trans_lst).cuda()
        sample['bg_img'] = self.dataset.bg_img.reshape([1,-1,3]).cuda()
        sample['bg_coords'] = self.dataset.bg_coords.cuda()
        return sample

    @torch.no_grad()
    def get_hubert(self, wav16k_name):
        from data_gen.utils.process_audio.extract_hubert import get_hubert_from_16k_wav
        hubert = get_hubert_from_16k_wav(wav16k_name).detach().numpy()
        len_mel = hubert.shape[0]
        x_multiply = 8
        hubert = hubert[:len(hubert)//8*8]
        # if len_mel % x_multiply == 0:
        #     num_to_pad = 0
        # else:
        #     num_to_pad = x_multiply - len_mel % x_multiply
        # hubert = np.pad(hubert, pad_width=((0,num_to_pad), (0,0)))
        return hubert

    def get_f0(self, wav16k_name):
        from data_gen.utils.process_audio.extract_mel_f0 import extract_mel_from_fname, extract_f0_from_wav_and_mel
        wav, mel = extract_mel_from_fname(self.wav16k_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        f0 = f0.reshape([-1,1])
        return f0
    
    @torch.no_grad()
    def forward_audio2secc(self, batch, inp=None):
        # forward the audio-to-motion
        ret = {}
        pred = self.audio2secc_model.forward(batch, ret=ret,train=False, temperature=inp['temperature'])
        
        # pred is the predicted motion
        # pred has the data type of torch.Tensor
        # pred has a shape of [B, T, 144]
        # B is the batch size, T is the number of frames, 144 is the dimension of the motion
        
        # ret is a dict that contains the predicted motion
        # ret has a shape of [B, T, 144]
        # B is the batch size, T is the number of frames, 144 is the dimension of the motion
        
        # the difference between pred and ret is that pred is the predicted motion, while ret is the predicted motion and some other information
        # some other information includes the predicted motion, the predicted id, the predicted expression, etc.
        
        # get the predicted motion
        if pred.shape[-1] == 144: # tensor.shape[-1] means the last dimension of the tensor
            id = ret['pred'][0][:,:80] 
            # ret['pred'] is a dict that contains the predicted motion
            # ret['pred'] has the data type of torch.Tensor
            # ret['pred'] has a shape of [B, T, 144]
            # represent ret['pred'] as a matrix is like:
            # [[1, 2, 3, ..., 144],
            #  [1, 2, 3, ..., 144],
            #  [1, 2, 3, ..., 144],
            #  ...]
            # shape[0] means the number of frames or the number of rows
            # shape[1] means the number of columns
            # shape[2] means the dimension of the motion
            
            exp = ret['pred'][0][:,80:]
        else:
            id = batch['id']
            exp = ret['pred'][0]
        
        # test print
        # print("====================================================")
        # print("pred shape: ", pred.shape)
        # print("pred", pred)
        # print("====================================================")
        # print("ret['pred'] shape: ", ret['pred'].shape)
        # print("ret['pred']", ret['pred'])
        # print("====================================================")
        # print("id shape: ", id.shape)
        # print("id", id)
        # print("====================================================")
        # print("exp shape: ", exp.shape)
        # print("exp", exp)
        # print("====================================================")
        
        # ====================================================
        # pred shape:  torch.Size([1, 296, 64]) 296 means the number of frames, 64 means the dimension of the motion
        # pred tensor([[[-0.6625, -0.2739, -0.5991,  ..., -0.0323, -0.0255,  0.0196],
        #          [-0.8144, -0.3875, -0.7192,  ..., -0.0404, -0.0281,  0.0265],
        #          [-0.8860, -0.4139, -0.7199,  ..., -0.0364, -0.0250,  0.0259],
        #          ...,
        #          [-0.5183, -0.2565, -0.5164,  ..., -0.0408, -0.0231,  0.0244],
        #          [-0.5325, -0.2631, -0.6023,  ..., -0.0394, -0.0348,  0.0232],
        #          [-0.3949, -0.2112, -0.4662,  ..., -0.0295, -0.0316,  0.0205]]],
        #        device='cuda:0')
        # ====================================================
        # ret['pred'] shape:  torch.Size([1, 296, 64])
        # ret['pred'] tensor([[[-0.6625, -0.2739, -0.5991,  ..., -0.0323, -0.0255,  0.0196],
        #          [-0.8144, -0.3875, -0.7192,  ..., -0.0404, -0.0281,  0.0265],
        #          [-0.8860, -0.4139, -0.7199,  ..., -0.0364, -0.0250,  0.0259],
        #          ...,
        #          [-0.5183, -0.2565, -0.5164,  ..., -0.0408, -0.0231,  0.0244],
        #          [-0.5325, -0.2631, -0.6023,  ..., -0.0394, -0.0348,  0.0232],
        #          [-0.3949, -0.2112, -0.4662,  ..., -0.0295, -0.0316,  0.0205]]],
        #        device='cuda:0')
        
        
        # ====================================================
        # id shape:  torch.Size([296, 80])
        # id tensor([[-0.2102, -0.3624, -0.5302,  ...,  0.6760,  0.1232,  0.5615],
        #         [-0.2102, -0.3624, -0.5302,  ...,  0.6760,  0.1232,  0.5615],
        #         [-0.2102, -0.3624, -0.5302,  ...,  0.6760,  0.1232,  0.5615],
        #         ...,
        #         [-0.2102, -0.3624, -0.5302,  ...,  0.6760,  0.1232,  0.5615],
        #         [-0.2102, -0.3624, -0.5302,  ...,  0.6760,  0.1232,  0.5615],
        #         [-0.2102, -0.3624, -0.5302,  ...,  0.6760,  0.1232,  0.5615]],
        #        device='cuda:0')
        # ====================================================
        # exp shape:  torch.Size([296, 64])
        # exp tensor([[-0.6625, -0.2739, -0.5991,  ..., -0.0323, -0.0255,  0.0196],
        #         [-0.8144, -0.3875, -0.7192,  ..., -0.0404, -0.0281,  0.0265],
        #         [-0.8860, -0.4139, -0.7199,  ..., -0.0364, -0.0250,  0.0259],
        #         ...,
        #         [-0.5183, -0.2565, -0.5164,  ..., -0.0408, -0.0231,  0.0244],
        #         [-0.5325, -0.2631, -0.6023,  ..., -0.0394, -0.0348,  0.0232],
        #         [-0.3949, -0.2112, -0.4662,  ..., -0.0295, -0.0316,  0.0205]],
        #        device='cuda:0')
        # ====================================================
        
    
        # id means the identity of the person, which is a 80-dim vector
        # exp means the expression of the person, which is a 64-dim vector
        # ret is a dict that contains the predicted motion
        # secc is the predicted motion, which is a 144-dim vector
        # secc means the speech-to-expression-to-camera-control, which is a 144-dim vector
        # pred is the predicted motion, which is a 144-dim vector

        # exp = smooth_features_xd(exp, kernel_size=3)
        batch['exp'] = exp

        # render the SECC given the id,exp.
        # note that SECC is only used for visualization
        zero_eulers = torch.zeros([id.shape[0], 3]).to(id.device)
        zero_trans = torch.zeros([id.shape[0], 3]).to(exp.device)
        
        # test print
        # print("====================================================")
        # print("zero_eulers shape: ", zero_eulers.shape)
        # print("zero_eulers", zero_eulers)
        # print("====================================================")
        # print("zero trans shape: ", zero_trans.shape)
        # print("zero_trans ", zero_trans)
        # print("====================================================")
        
        # ====================================================
        # zero trans shape:  torch.Size([296, 3])
        # zero_trans  tensor([[0., 0., 0.],
        #                     ====================================================
        # zero trans shape:  torch.Size([296, 3])
        # zero_trans  tensor([[0., 0., 0.],
        
        # zero_eulers is a 3-dim vector, which is [0,0,0]
        # zero_trans is a 3-dim vector, which is [0,0,0]
        
        # zero_eulers is the euler angle of the camera, which is a 3-dim vector
        # zero_trans is the translation of the camera, which is a 3-dim vector
        
        # render the SECC
        if inp['debug']:
            with torch.no_grad():
                chunk_size = 50 # chunck size means the number of frames in a chunk
                drv_secc_color_lst = [] # drv_secc_color_lst is a list that contains the rendered SECC
                num_iters = len(id)//chunk_size if len(id)%chunk_size == 0 else len(id)//chunk_size+1
                # num_iters is the number of iterations
                # num_iters is the number of frames divided by the chunk size
                # len(id) is the number of frames
                
                
                # 296//50 = 5 
                # so there run 5 iterations here
                for i in tqdm.trange(num_iters, desc="rendering secc"): # tqdm.trange is used to create a progress bar
                    torch.cuda.empty_cache() # empty the cache
                    face_mask, drv_secc_color = self.secc_renderer(id[i*chunk_size:(i+1)*chunk_size], exp[i*chunk_size:(i+1)*chunk_size], zero_eulers[i*chunk_size:(i+1)*chunk_size], zero_trans[i*chunk_size:(i+1)*chunk_size])
                    # face_mask is the mask of the face
                    # drv_secc_color is the rendered SECC
                    # face_mask has the data type of torch.Tensor
                    # drv_secc_color has the data type of torch.Tensor
                    # face_mask has the shape of [B, T, H, W]
                    # drv_secc_color has the shape of [B, T, H, W, 3]
                    # B is the batch size, T is the number of frames, H is the height of the image, W is the width of the image
                    # 3 is the dimension of the color image
                    drv_secc_color_lst.append(drv_secc_color.cpu())
                    # append the drv_secc_color to the drv_secc_color_lst
                    
                    
                    
            drv_secc_colors = torch.cat(drv_secc_color_lst, dim=0)
            _, src_secc_color = self.secc_renderer(id[0:1], exp[0:1], zero_eulers[0:1], zero_trans[0:1])
            # drv_secc_colors is the rendered SECC
            # drv_secc_colors has the data type of torch.Tensor
            # drv_secc_colors has the shape of [B, T, H, W, 3]
            # B is the batch size, T is the number of frames, H is the height of the image, W is the width of the image
            # src_secc_color is the rendered SECC
            # src_secc means the source SECC
            # drv_secc means the driver SECC
            # cano_secc means the canonical SECC
            # cononical means the standard or the normal
            _, cano_secc_color = self.secc_renderer(id[0:1]*0, exp[0:1]*0, zero_eulers[0:1], zero_trans[0:1])
            batch['drv_secc'] = drv_secc_colors.cuda()
            batch['src_secc'] = src_secc_color.cuda()
            batch['cano_secc'] = cano_secc_color.cuda()

        # get idexp_lm3d
        id_ds = torch.from_numpy(self.dataset.ds_dict['id']).float().cuda()
        exp_ds = torch.from_numpy(self.dataset.ds_dict['exp']).float().cuda()
        # id_ds is the identity of the person
        # exp_ds is the expression of the person
        
        idexp_lm3d_ds = self.face3d_helper.reconstruct_idexp_lm3d(id_ds, exp_ds)
        # idexp_lm3d_ds is the reconstructed idexp_lm3d
        # idexp means the identity and the expression
        # ds means the dataset
        
        # test print
        # print("====================================================")
        # print("idexp_lm3d_ds shape: ", idexp_lm3d_ds.shape)
        # print("idexp_lm3d_ds", idexp_lm3d_ds)
        # print("====================================================")
        # ====================================================
        
        
        # idexp_lm3d_ds shape:  torch.Size([6073, 468, 3])
        # 6073 is the number of vertices (time????) this is from the dataset
        # 468 is the number of frames (468 keypoints)
        # 3 is the dimension of the vertices (x, y, z)
        
        
        # idexp_lm3d_ds tensor([[[ 6.3554e-02,  9.4643e-02, -9.9931e-02],
        #          [ 2.8271e-02,  1.5831e-01, -4.7812e-01],
        #          [ 4.6547e-02,  1.8106e-01, -2.4417e-01],
        #          ...,
        #          [ 1.3497e-02, -2.1638e-01, -2.0108e-01],
        #          [ 1.7164e-02, -1.5568e-01,  1.3663e-01],
        #          [ 2.1344e-02, -1.5367e-01,  1.4077e-01]],

        #         [[ 6.6701e-02,  1.4066e-01, -1.0002e-01],
        #          [ 2.5007e-02,  1.6064e-01, -4.6687e-01],
        #          [ 4.6522e-02,  2.0462e-01, -2.3817e-01],
        #          ...,
        #          [ 7.4772e-03, -2.2352e-01, -1.9713e-01],
        #          [ 1.2038e-02, -1.6846e-01,  1.4172e-01],
        #          [ 1.7085e-02, -1.6401e-01,  1.4455e-01]],

        #         [[ 6.7414e-02,  1.6843e-01, -1.0111e-01],
        #          [ 2.1617e-02,  1.7016e-01, -4.5523e-01],
        #          [ 4.5371e-02,  2.2322e-01, -2.3122e-01],
        #          ...,
        #          [ 3.0959e-03, -2.2651e-01, -1.9275e-01],
        #          [ 8.4278e-03, -1.7282e-01,  1.4730e-01],
        #          [ 1.3790e-02, -1.6802e-01,  1.4974e-01]],

        #         ...,

        #         [[ 7.0893e-02,  7.8350e-02, -2.1736e-01],
        #          [ 5.1951e-03,  2.0557e-01, -4.4006e-01],
        #          [ 3.8828e-02,  1.9628e-01, -2.6556e-01],
        #          ...,
        #          [-1.6553e-02, -2.1003e-01, -1.9693e-01],
        #          [-1.1919e-02, -1.5648e-01,  1.5169e-01],
        #          [-5.7347e-03, -1.5486e-01,  1.5720e-01]],

        #         [[ 6.9242e-02,  8.0264e-02, -2.2330e-01],
        #          [ 4.0673e-03,  2.0553e-01, -4.3907e-01],
        #          [ 3.7566e-02,  1.9613e-01, -2.6938e-01],
        #          ...,
        #          [-1.7211e-02, -2.0916e-01, -1.9789e-01],
        #          [-1.2680e-02, -1.5436e-01,  1.5150e-01],
        #          [-6.3383e-03, -1.5323e-01,  1.5713e-01]],

        #         [[ 6.0420e-02,  7.7217e-02, -2.2183e-01],
        #          [-4.2177e-04,  2.0450e-01, -4.3958e-01],
        #          [ 3.1322e-02,  1.9402e-01, -2.6742e-01],
        #          ...,
        #          [-1.8336e-02, -2.0784e-01, -1.9843e-01],
        #          [-1.4090e-02, -1.5234e-01,  1.5207e-01],
        #          [-7.5138e-03, -1.5169e-01,  1.5785e-01]]], device='cuda:0')
        # ====================================================
        
        
        idexp_lm3d_mean = idexp_lm3d_ds.mean(dim=0, keepdim=True)
        # idexp_lm3d_mean is the mean of the idexp_lm3d
        # mean is the average value of the idexp_lm3d
        # this is used to normalize the idexp_lm3d
        # keepdim=True means to keep the dimension of the tensor
        idexp_lm3d_std = idexp_lm3d_ds.std(dim=0, keepdim=True)
        # idexp_lm3d_std is the standard deviation of the idexp_lm3d
        # torch.std() is a function to calculate the standard deviation of the tensor
        
        if hparams.get("normalize_cond", True):
            idexp_lm3d_ds_normalized = (idexp_lm3d_ds - idexp_lm3d_mean) / idexp_lm3d_std
            # normalize the idexp_lm3d
            # normalize_cond means to normalize the condition feature
            # normalize can make the condition feature have a mean of 0 and a standard deviation of 1
            # why normalize the condition feature?
            # because the condition feature is used to condition the NeRF model
            # the condition feature is used to provide the information of the person
        else:
            idexp_lm3d_ds_normalized = idexp_lm3d_ds
            
        lower = torch.quantile(idexp_lm3d_ds_normalized, q=0.03, dim=0)
        upper = torch.quantile(idexp_lm3d_ds_normalized, q=0.97, dim=0)
        # torch.quantile() is a function to calculate the quantile of the tensor
        # quantile means the value below which a certain percentage of the data falls
        
        # lower is the 3rd percentile of the idexp_lm3d
        # upper is the 97th percentile of the idexp_lm3d
        
        # lower has the shape of [468, 3]
        # upper has the shape of [468, 3]
        # 468 is the number of keypoint of the face
        # 3 is the dimension of the keypoint of the face
        # the dimension of the keypoint of the face is 3, which is the x, y, z coordinate of the keypoint
        # x is the horizontal coordinate of the keypoint
        # y is the vertical coordinate of the keypoint
        # z is the depth coordinate of the keypoint
        # depth coordinate means the distance from the camera to the keypoint

        LLE_percent = inp['lle_percent']
            
        keypoint_mode = self.secc2video_model.hparams.get("nerf_keypoint_mode", "lm68")
        
        # keypoint_mode = self.secc2video_model.hparams.get("nerf_keypoint_mode", "lm468")
        
        
        # keypoint_mode is the mode of the keypoint
        # keypoint_mode is the mode of the keypoint of the NeRF model
        # dict.get() is a method to get the value of the key in the dict
        # dict.get() has two parameters, the first parameter is the key, the second parameter is the default value
        # if the key is not in the dict, then return the default value
        # if the key is in the dict, then return the value of the key
        
        # test print
        # print("====================================================")
        # print("keypoint_mode", keypoint_mode)
        # print("====================================================")
        # keypoint_mode lm68
        
        # =====================================================================================================
        # postnet model is not used here
        if self.postnet_model is not None:
            idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm68_from_lm478]
            idexp_lm3d_std = idexp_lm3d_std[:, index_lm68_from_lm478]
            lower = lower[index_lm68_from_lm478]
            upper = upper[index_lm68_from_lm478]

            f0 = self.audio2secc_model.downsampler(batch['f0'].unsqueeze(-1)).squeeze(-1)
            pitch = self.audio2secc_model.pitch_embed(f0_to_coarse(f0))
            b,t,*_ = pitch.shape
            raw_pred_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(id, exp).reshape([t, -1, 3])
            raw_pred_lm3d = raw_pred_lm3d[:, index_lm68_from_lm478].reshape([1,t,68*3])
            
            idexp_lm3d = self.postnet_model(raw_pred_lm3d, pitch[:, :exp.shape[0]*2, :]).reshape([t,68*3])

            idexp_lm3d_ds_lle = idexp_lm3d_ds[:, index_lm68_from_lm478].reshape([-1, 68*3])
            feat_fuse, _, _ = compute_LLE_projection(feats=idexp_lm3d[:, :68*3], feat_database=idexp_lm3d_ds_lle[:, :68*3], K=10)
            # feat_fuse = smooth_features_xd(feat_fuse, kernel_size=3)
            idexp_lm3d[:, :68*3] = LLE_percent * feat_fuse + (1-LLE_percent) * idexp_lm3d[:,:68*3]
            idexp_lm3d = idexp_lm3d.reshape([t, 68, 3])
            idexp_lm3d_normalized = (idexp_lm3d - idexp_lm3d_mean) / idexp_lm3d_std
            idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)
        # =====================================================================================================
        else:
            #　如果要疊有表情的 lm68 就需要在這邊做處理
            idexp_lm3d = self.face3d_helper.reconstruct_idexp_lm3d(id, exp)
            
            
            
            # test print
            # print("====================================================")
            # print("idexp_lm3d shape: ", idexp_lm3d.shape)
            # print("idexp_lm3d", idexp_lm3d)
            # print("====================================================")
            # ====================================================
            # idexp_lm3d shape:  torch.Size([296, 468, 3])
            # idexp_lm3d tensor([[[ 1.3137e-02,  1.8173e-01, -2.1091e-01],
            #          [-1.9534e-02,  1.8525e-01, -4.4876e-01],
            #          [-5.7675e-04,  2.3181e-01, -2.8790e-01],
            #          ...,
            #          [-1.6201e-02, -2.2581e-01, -1.9396e-01],
            #          [-8.4291e-03, -1.7772e-01,  1.5168e-01],
            #          [-1.3177e-03, -1.7242e-01,  1.5368e-01]],

            #         [[ 7.3765e-03,  1.9598e-01, -2.4478e-01],
            #          [-2.2736e-02,  1.9214e-01, -4.3519e-01],
            #          [-4.8533e-03,  2.4138e-01, -2.9759e-01],
            #          ...,
            #          [-1.7356e-02, -2.2807e-01, -1.8986e-01],
            #          [-6.7041e-03, -1.8041e-01,  1.5190e-01],
            #          [ 2.5152e-04, -1.7388e-01,  1.5329e-01]],

            #         [[ 6.9289e-03,  1.9042e-01, -2.6781e-01],
            #          [-2.1124e-02,  1.9054e-01, -4.3554e-01],
            #          [-4.0215e-03,  2.3949e-01, -3.0157e-01],
            #          ...,
            #          [-1.5905e-02, -2.2791e-01, -1.9238e-01],
            #          [-5.3149e-03, -1.8023e-01,  1.4957e-01],
            #          [ 2.1884e-03, -1.7337e-01,  1.5042e-01]],

            #         ...,

            #         [[ 4.1743e-03,  1.2756e-01, -2.3630e-01],
            #          [-2.0098e-02,  1.8841e-01, -4.4073e-01],
            #          [-4.0078e-03,  2.1338e-01, -2.6787e-01],
            #          ...,
            #          [-1.3927e-02, -2.2692e-01, -1.8910e-01],
            #          [-4.8045e-03, -1.7795e-01,  1.5306e-01],
            #          [ 1.8897e-03, -1.7175e-01,  1.5439e-01]],

            #         [[ 6.2336e-03,  1.2417e-01, -2.3731e-01],
            #          [-1.8384e-02,  1.9502e-01, -4.3667e-01],
            #          [-1.7208e-03,  2.1602e-01, -2.6759e-01],
            #          ...,
            #          [-1.2820e-02, -2.2846e-01, -1.8238e-01],
            #          [-3.4929e-03, -1.7912e-01,  1.6003e-01],
            #          [ 2.5559e-03, -1.7413e-01,  1.6214e-01]],

            #         [[ 3.2827e-03,  1.2444e-01, -2.1859e-01],
            #          [-2.1148e-02,  1.8789e-01, -4.4953e-01],
            #          [-4.7389e-03,  2.1248e-01, -2.6418e-01],
            #          ...,
            #          [-1.4558e-02, -2.2717e-01, -1.8962e-01],
            #          [-7.6087e-03, -1.7937e-01,  1.5646e-01],
            #          [-9.2136e-04, -1.7459e-01,  1.5829e-01]]], device='cuda:0')
            # ====================================================

            # =====================================================================================================
            # create emotional_lm3d here
            # emotional_lm3d will have the same shape as idexp_lm3d
            # shape: (frame_num, 68, 3)
            
            # and will be concatenated to the idexp_lm3d

            # code here (pseudo code)
            # get the emotional_lm3d from the emotional model
            
            
            # emotional_lm3d = emotional_model(id, exp)
            
            
            
            
            # =====================================================================================================
            # runs lm68 here
            if keypoint_mode == 'lm68':
                idexp_lm3d = idexp_lm3d[:, index_lm68_from_lm478]
                # index_lm68_from_lm478 is the index of the lm68 from the lm478
                # index_lm68_from_lm478 is a list that contains the index of the lm68 from the lm478
                
                # lm478 is the 478 keypoint of the face
                # lm68 is the 68 keypoint of the face
                # lm131 is the 131 keypoint of the face
                
                idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm68_from_lm478] 
                idexp_lm3d_std = idexp_lm3d_std[:, index_lm68_from_lm478]
                
                lower = lower[index_lm68_from_lm478]
                upper = upper[index_lm68_from_lm478]
                
                # test print
                # print("====================================================")
                # print("idexp_lm3d shape: ", idexp_lm3d.shape)
                # print("idexp_lm3d", idexp_lm3d)
                # print("====================================================")
                # ====================================================
                # idexp_lm3d shape:  torch.Size([296, 68, 3])
                # idexp_lm3d tensor([[[ 0.0998, -0.0313,  0.0252],
                #          [ 0.0998, -0.0313,  0.0252],
                #          [ 0.0998, -0.0313,  0.0252],
                #          ...,
                #          [ 0.1482, -0.5972, -0.4158],
                #          [ 0.0498, -0.6626, -0.3912],
                #          [-0.0482, -0.5735, -0.4367]],

                #         [[ 0.0973, -0.0259,  0.0259],
                #          [ 0.0973, -0.0259,  0.0259],
                #          [ 0.0973, -0.0259,  0.0259],
                #          ...,
                #          [ 0.1630, -0.7042, -0.5250],
                #          [ 0.0398, -0.7849, -0.4879],
                #          [-0.0839, -0.6789, -0.5548]],

                #         [[ 0.0961, -0.0291,  0.0227],
                #          [ 0.0961, -0.0291,  0.0227],
                #          [ 0.0961, -0.0291,  0.0227],
                #          ...,
                #          [ 0.1625, -0.7147, -0.5643],
                #          [ 0.0340, -0.7973, -0.5250],
                #          [-0.0932, -0.6887, -0.6003]],

                #         ...,

                #         [[ 0.1011, -0.0265,  0.0285],
                #          [ 0.1011, -0.0265,  0.0285],
                #          [ 0.1011, -0.0265,  0.0285],
                #          ...,
                #          [ 0.1180, -0.3764, -0.3691],
                #          [ 0.0224, -0.4199, -0.3501],
                #          [-0.0621, -0.3587, -0.3954]],

                #         [[ 0.1023, -0.0258,  0.0336],
                #          [ 0.1023, -0.0258,  0.0336],
                #          [ 0.1023, -0.0258,  0.0336],
                #          ...,
                #          [ 0.1239, -0.4469, -0.4064],
                #          [ 0.0240, -0.5006, -0.3847],
                #          [-0.0650, -0.4313, -0.4347]],

                #         [[ 0.1013, -0.0341,  0.0292],
                #          [ 0.1013, -0.0341,  0.0292],
                #          [ 0.1013, -0.0341,  0.0292],
                #          ...,
                #          [ 0.1008, -0.3502, -0.3132],
                #          [ 0.0223, -0.3872, -0.3029],
                #          [-0.0445, -0.3323, -0.3396]]], device='cuda:0')
                # ====================================================                

            elif keypoint_mode == 'lm131':
                idexp_lm3d = idexp_lm3d[:, index_lm131_from_lm478]
                idexp_lm3d_mean = idexp_lm3d_mean[:, index_lm131_from_lm478]
                idexp_lm3d_std = idexp_lm3d_std[:, index_lm131_from_lm478]
                lower = lower[index_lm131_from_lm478]
                upper = upper[index_lm131_from_lm478]
                
                
            elif keypoint_mode == 'lm468':
                idexp_lm3d = idexp_lm3d
            else:
                raise NotImplementedError()
            
            
            # idexp_lm3d vs idexp_lm3d_ds
            # idexp_lm3d is the predicted idexp_lm3d
            # idexp_lm3d has the data type of torch.Tensor
            # idexp_lm3d has the shape of [B, 68, 3]
            # B is the batch size, 68 is the number of keypoint of the face, 3 is the dimension of the keypoint of the face
            # 68 is the number of keypoint of the face
            # 3 is the dimension of the keypoint of the face
            # x is the horizontal coordinate of the keypoint
            # y is the vertical coordinate of the keypoint
            # z is the depth coordinate of the keypoint
            # depth coordinate means the distance from the camera to the keypoint
            
            # indexp_lm3d_ds is the identity and expression of the person
            # indexp_lm3d_ds has the shape of [6073, 468, 3]
            # indexp_lm3d comes from the dataset
            
            
            
            
            
            
            # LLE projection
            # =====================================================================================================
            # =====================================================================================================
            idexp_lm3d = idexp_lm3d.reshape([-1, 68*3])
            # indexp_lm3d is the identity and expression of the person
            # indexp_lm3d has the shape of [B, 68*3]
            # B is the batch size, 68 is the number of keypoint of the face, 3 is the dimension of the keypoint of the face
            # 68 is the number of keypoint of the face
            # .reshape([-1, 68*3]) means to reshape the tensor to the shape of [B, 68*3]

            
            # test print
            # print("====================================================")
            # print("idexp_lm3d shape: ", idexp_lm3d.shape)
            # print("idexp_lm3d", idexp_lm3d)
            # print("====================================================")
            # ====================================================
            # idexp_lm3d shape:  torch.Size([296, 204])
            # idexp_lm3d tensor([[ 0.1036, -0.0445,  0.0341,  ..., -0.0556, -0.5942, -0.4698],
            #         [ 0.1020, -0.0422,  0.0366,  ..., -0.0951, -0.6845, -0.5892],
            #         [ 0.1008, -0.0449,  0.0331,  ..., -0.1053, -0.6883, -0.6300],
            #         ...,
            #         [ 0.1007, -0.0312,  0.0281,  ..., -0.0017, -0.3658, -0.3024],
            #         [ 0.1019, -0.0301,  0.0326,  ...,  0.0075, -0.4107, -0.3298],
            #         [ 0.1007, -0.0375,  0.0288,  ...,  0.0162, -0.3012, -0.2510]],
            #        device='cuda:0')
            # ====================================================   
                  
            idexp_lm3d_ds_lle = idexp_lm3d_ds[:, index_lm68_from_lm478].reshape([-1, 68*3])
            # idexp_lm3d_ds_lle is the reconstructed idexp_lm3d
            # idexp_lm3d_ds_lle has the shape of [B, 68*3]
            
            feat_fuse, _, _ = compute_LLE_projection(feats=idexp_lm3d[:, :68*3], feat_database=idexp_lm3d_ds_lle[:, :68*3], K=10)
            # feat_fuse = smooth_features_xd(feat_fuse, kernel_size=3)
            
            idexp_lm3d[:, :68*3] = LLE_percent * feat_fuse + (1-LLE_percent) * idexp_lm3d[:,:68*3]
            idexp_lm3d = idexp_lm3d.reshape([-1, 68, 3])
            
            
            
            idexp_lm3d_normalized = (idexp_lm3d - idexp_lm3d_mean) / idexp_lm3d_std
            # idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)
            
            # =====================================================================================================
            # =====================================================================================================

        # cano_lm3d 是最後會可以被視覺化的 68 個 landmark
        # 這邊有怎麼來的，所以可以反推 cano_lm3d 是從 idexp_lm3d 來的
        # 可以推測出 idexp_lm3d 是最後生成臉部的資料型態
        # 這邊的 idexp_lm3d 是經過 LLE projection 的
        
        # (idexp_lm3d_mean + idexp_lm3d_std * idexp_lm3d_normalized) 其實就是 idexp_lm3d，只是經過了標準化再還原
        cano_lm3d = (idexp_lm3d_mean + idexp_lm3d_std * idexp_lm3d_normalized) / 10 + self.face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)
        eye_area_percent = self.opened_eye_area_percent * torch.ones([len(cano_lm3d), 1], dtype=cano_lm3d.dtype, device=cano_lm3d.device)
        
        # test print
        # print("====================================================")
        # print("cano_lm3d shape: ", cano_lm3d.shape)
        # print("cano_lm3d", cano_lm3d)
        # print("====================================================")
        # cano_lm3d shape:  torch.Size([296, 68, 3])
        # 應該是每個 frame 的 68 個特徵點的 x, y, z 座標，座標的y軸是垂直向上的
        # 每個座標值都介在 -1~1 之間
        
        if inp['blink_mode'] == 'period':
            cano_lm3d, eye_area_percent = inject_blink_to_lm68(cano_lm3d, self.opened_eye_area_percent, self.closed_eye_area_percent)
            print("Injected blink to idexp_lm3d by directly editting.")
            
            
        batch['eye_area_percent'] = eye_area_percent
        idexp_lm3d_normalized = ((cano_lm3d - self.face3d_helper.key_mean_shape[index_lm68_from_lm478].unsqueeze(0)) * 10 - idexp_lm3d_mean) / idexp_lm3d_std
        idexp_lm3d_normalized = torch.clamp(idexp_lm3d_normalized, min=lower, max=upper)
        batch['cano_lm3d'] = cano_lm3d

        assert keypoint_mode == 'lm68'
        # assert means to check if the condition is True
        # if the condition is True, then continue to run the code
        # if the condition is False, then raise an error
        
        idexp_lm3d_normalized_ = idexp_lm3d_normalized[0:1, :].repeat([len(exp),1,1]).clone()
        idexp_lm3d_normalized_[:, 17:27] = idexp_lm3d_normalized[:, 17:27] # brow
        idexp_lm3d_normalized_[:, 36:48] = idexp_lm3d_normalized[:, 36:48] # eye
        idexp_lm3d_normalized_[:, 27:36] = idexp_lm3d_normalized[:, 27:36] # nose
        idexp_lm3d_normalized_[:, 48:68] = idexp_lm3d_normalized[:, 48:68] # mouth
        idexp_lm3d_normalized_[:, 0:17] = idexp_lm3d_normalized[:, :17] # yaw

        idexp_lm3d_normalized = idexp_lm3d_normalized_

        cond_win = idexp_lm3d_normalized.reshape([len(exp), 1, -1])
        cond_wins = [get_audio_features(cond_win, att_mode=2, index=idx) for idx in range(len(cond_win))]
        batch['cond_wins'] = cond_wins

        # face boundark mask, for cond mask
        smo_euler = smooth_features_xd(batch['euler'])
        smo_trans = smooth_features_xd(batch['trans'])
        lm2d = self.face3d_helper.reconstruct_lm2d_nerf(id, exp, smo_euler, smo_trans)
        
        lm68 = lm2d[:, index_lm68_from_lm478, :] # lm468 convert to lm68
        batch['lm68'] = lm68.reshape([lm68.shape[0], 68*2]) # 用來丟進模型的 68 個特徵點
        # cond mask means the condition mask
        # cond mask is used to condition the NeRF model
        # 在NeRF（Neural Radiance Fields）模型中，condition mask（條件遮罩）的作用主要是用來控制或限制模型
        # 在不同情境下如何生成或渲染3D場景，尤其是在處理有多個條件（如視角、光照或物體屬性等）的情況下。
        
        # test print
        # print("====================================================")
        # print("lm68 shape: ", lm68.shape)
        # print("lm68", lm68)
        # print("====================================================")
        # lm68 shape:  torch.Size([296, 68, 2])
        # 應該是每個 frame 的 68 個特徵點的 x, y 座標
        # 每個座標值都介在 0~1 之間

        return batch

    @torch.no_grad()
    # todo: fp16 infer for faster inference. Now 192/128/64 are all 45fps
    def forward_secc2video(self, batch, inp=None):
        num_frames = len(batch['poses'])
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        cond_inp = batch['cond_wins']
        bg_coords = batch['bg_coords']
        bg_color = batch['bg_img']
        poses = batch['poses']
        lm68s = batch['lm68']
        eye_area_percent = batch['eye_area_percent']

        pred_rgb_lst = []

        # forward renderer  
        if inp['low_memory_usage']:
            # save memory, when one image is rendered, write it into video
            try:
                os.makedirs(os.path.dirname(inp['out_name']), exist_ok=True)
            except: pass
            import imageio
            tmp_out_name = inp['out_name'].replace(".mp4", ".tmp.mp4")
            writer = imageio.get_writer(tmp_out_name, fps = 25, format='FFMPEG', codec='h264')

            with torch.cuda.amp.autocast(enabled=True):
                # forward neural renderer
                for i in tqdm.trange(num_frames, desc="GeneFace++ is rendering... "):
                    model_out = self.secc2video_model.render(rays_o[i], rays_d[i], cond_inp[i], bg_coords, poses[i], index=i, staged=False, bg_color=bg_color, lm68=lm68s[i], perturb=False, force_all_rays=False,
                                    T_thresh=inp['raymarching_end_threshold'], eye_area_percent=eye_area_percent[i],
                                    **hparams)
                    if self.secc2video_hparams.get('with_sr', False):
                        pred_rgb = model_out['sr_rgb_map'][0].cpu() # [c, h, w]
                    else:
                        pred_rgb = model_out['rgb_map'][0].reshape([512,512,3]).permute(2,0,1).cpu()
                    img = (pred_rgb.permute(1,2,0) * 255.).int().cpu().numpy().astype(np.uint8)
                    writer.append_data(img)
            writer.close()

        else:

            with torch.cuda.amp.autocast(enabled=True): 
                # torch.cuda maeans to use the GPU to run the code
                # torch.amp means to use the automatic mixed precision to run the code
                # torch.amp.autocast() is a context manager to run the code with the automatic mixed precision
                
                # forward neural renderer
                for i in tqdm.trange(num_frames, desc="GeneFace++ is rendering... "):
                    model_out = self.secc2video_model.render(rays_o[i], rays_d[i], cond_inp[i], bg_coords, poses[i], index=i, staged=False, bg_color=bg_color, lm68=lm68s[i], perturb=False, force_all_rays=False,
                                    T_thresh=inp['raymarching_end_threshold'], eye_area_percent=eye_area_percent[i],
                                    **hparams)
                    if self.secc2video_hparams.get('with_sr', False):
                        pred_rgb = model_out['sr_rgb_map'][0].cpu() # [c, h, w]
                    else:
                        pred_rgb = model_out['rgb_map'][0].reshape([512,512,3]).permute(2,0,1).cpu()

                    pred_rgb_lst.append(pred_rgb)
            pred_rgbs = torch.stack(pred_rgb_lst).cpu()
            pred_rgbs = pred_rgbs * 2 - 1 # to -1~1 scale
            # -1~1 scale means the pixel value is in the range of -1 to 1
            # the benefit of the -1~1 scale is that the pixel value is centered at 0
            # the pixel value is centered at 0 means that the pixel value is centered at the origin
            # pred_rgbs has the shape of [B, C, H, W]

            if inp['debug']:
                # prepare for output
                drv_secc_colors = batch['drv_secc']
                secc_img = torch.cat([torch.nn.functional.interpolate(drv_secc_colors[i:i+1], (512,512)) for i in range(num_frames)]).cpu()
                cano_lm3d_frame_lst = vis_cano_lm3d_to_imgs(batch['cano_lm3d'], hw=512)
                cano_lm3d_frames = convert_to_tensor(np.stack(cano_lm3d_frame_lst)).permute(0, 3, 1, 2) / 127.5 - 1
                # np.stack means to stack the list of the numpy array
                # np.stack has two parameters, the first parameter is the list of the numpy array, the second parameter is the axis
                # 
                # tensor.permute() is a method to permute the dimensions of the tensor
                # tensor.permute() has the same effect as tensor.transpose()
                # tensor.permute() has one parameter, which is the order of the dimensions of the tensor
                # tensor.permute() can change the order of the dimensions of the tensor
                imgs = torch.cat([pred_rgbs, cano_lm3d_frames, secc_img], dim=3) # [B, C, H, W]
                
                # torch.cat() is a function to concatenate the tensor
                # torch.cat() has three parameters, the first parameter is the list of the tensor, the second parameter is the axis
                # torch.cat() can concatenate the tensor along the specified axis
                # for example, torch.cat([tensor1, tensor2], dim=0) will concatenate tensor1 and tensor2 along the 0th dimension
                # imgs has the shape of [B, C, H, W]
                # B is the batch size, C is the number of channels, H is the height, W is the width
                # imgs is the tensor that contains the images
                # imgs represent the color by the dimension of the tensor
                # the dimension is the channel of the image
                
            else:
                imgs = pred_rgbs
            imgs = imgs.clamp(-1,1)

            try:
                os.makedirs(os.path.dirname(inp['out_name']), exist_ok=True)
            except: pass
            import imageio
            tmp_out_name = inp['out_name'].replace(".mp4", ".tmp.mp4")
            out_imgs = ((imgs.permute(0, 2, 3, 1) + 1)/2 * 255).int().cpu().numpy().astype(np.uint8)
            writer = imageio.get_writer(tmp_out_name, fps = 25, format='FFMPEG', codec='h264')
            for i in tqdm.trange(len(out_imgs), desc=f"ImageIO is saving video using FFMPEG(h264) to {tmp_out_name}"):
                writer.append_data(out_imgs[i])
            writer.close()

        cmd = f"ffmpeg -i {tmp_out_name} -i {self.wav16k_name} -y -shortest -c:v libx264 -pix_fmt yuv420p -b:v 2000k -y -v quiet -shortest {inp['out_name']}"
        ret = os.system(cmd)
        if ret == 0:
            print(f"Saved at {inp['out_name']}")
            os.system(f"rm {self.wav16k_name}")
            os.system(f"rm {tmp_out_name}")
        else:
            raise ValueError(f"error running {cmd}, please check ffmpeg installation, especially check whether it supports libx264!")

    @torch.no_grad()
    def forward_system(self, batch, inp):
        self.forward_audio2secc(batch, inp)
        self.forward_secc2video(batch, inp)
        return inp['out_name']

    @classmethod
    def example_run(cls, inp=None):
        inp_tmp = {
            'drv_audio_name': 'data/raw/val_wavs/zozo.wav',
            'src_image_name': 'data/raw/val_imgs/Macron.png'
            }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp

        infer_instance = cls(inp['a2m_ckpt'], inp['postnet_ckpt'], inp['head_ckpt'], inp['torso_ckpt'])
        # cls method is used to create a new object of the class
        # the code will jump to the __init__ method of the class
        infer_instance.infer_once(inp)
        # infer_instance is an object of the class
        # infer_once is a method of the class
        # infer_once is used to run the model
        

    ##############
    # IO-related
    ##############
    def save_wav16k(self, audio_name):
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert audio_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = audio_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {audio_name} -f wav -ar 16000 -v quiet -y {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Extracted wav file (16khz) from {audio_name} to {wav16k_name}.")


if __name__ == '__main__':
    import argparse, glob, tqdm
    # argparse is a python library for parsing command-line arguments
    # glob is a python library for finding files/directories matching a specified pattern
    # tqdm is a python library for adding progress bars to loops
    # argparse.ArgumentParser() is a class that allows the user to specify what arguments are required and describes how the arguments will be parsed
    # argparse.ArgumentParser().add_argument() is a method that adds an argument to the parser
    # argparse.ArgumentParser().parse_args() is a method that parses the arguments from the command line
    # argparse.ArgumentParser().parse_args().a2m_ckpt is a variable that stores the path to the audio2motion checkpoint
    # argparse.ArgumentParser().parse_args().head_ckpt is a variable that stores the path to the head checkpoint
    # argparse.ArgumentParser().parse_args().postnet_ckpt is a variable that stores the path to the postnet checkpoint
    # argparse.ArgumentParser().parse_args().torso_ckpt is a variable that stores the path to the torso checkpoint
    # argparse.ArgumentParser().parse_args().drv_aud is a variable that stores the path to the audio source
    # argparse.ArgumentParser().parse_args().drv_pose is a variable that stores the pose of the driver
    # argparse.ArgumentParser().parse_args().blink_mode is a variable that stores the blink mode
    # argparse.ArgumentParser().parse_args().lle_percent is a variable that stores the percentage of LLE
    # LLE is a method that injects blink to the lm68
    # LLE.inject_blink_to_lm68() is a method that injects blink to the lm68
    # Locally Linear Embedding (LLE) is a method that reduces the dimensionality of the data
    # Locally Linear Embedding (LLE) is a method that preserves the local structure of the data
    
    
    # lm68 is a variable that stores the lm68
    
    # raymarching is a method that renders the video
    # ray marching means that the rays are marched through the scene to render the video
        
    
    # argparse.ArgumentParser().parse_args().temperature is a variable that stores the temperature
    # argparse.ArgumentParser().parse_args().mouth_amp is a variable that stores the mouth amplitude
    # argparse.ArgumentParser().parse_args().raymarching_end_threshold is a variable that stores the raymarching end threshold
    # argparse.ArgumentParser().parse_args().debug is a variable that stores whether to debug
    # argparse.ArgumentParser().parse_args().fast is a variable that stores whether to run fast
    # argparse.ArgumentParser().parse_args().out_name is a variable that stores the output name
    # argparse.ArgumentParser().parse_args().low_memory_usage is a variable that stores whether to use low memory usage
    # GeneFace2Infer.example_run() is a method that runs the GeneFace2Infer class
    # GeneFace2Infer.example_run().prepare_batch_from_inp() is a method that prepares the batch from the input
    # GeneFace2Infer.example_run().get_hubert() is a method that gets the hubert from the 16k wav
    # GeneFace2Infer.example_run().get_f0() is a method that gets the f0 from the 16k wav
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='checkpoints/audio2motion_vae')  # checkpoints/0727_audio2secc/audio2secc_withlm2d100_randomframe
    parser.add_argument("--head_ckpt", default='') 
    parser.add_argument("--postnet_ckpt", default='') 
    # parser.add_argument("--torso_ckpt", default='')
    parser.add_argument("--torso_ckpt", default='checkpoints/motion2video_nerf/may_torso')
    
    parser.add_argument("--drv_aud", default='data/raw/val_wavs/MacronSpeech.wav')
    parser.add_argument("--drv_pose", default='nearest', help="目前仅支持源视频的pose,包括从头开始和指定区间两种,暂时不支持in-the-wild的pose")
    parser.add_argument("--blink_mode", default='none') # none | period
    # parser.add_argument("--blink_mode", default='period') # none | period
    parser.add_argument("--lle_percent", default=0.2) # nearest | random
    parser.add_argument("--temperature", default=0.2) # nearest | random
    parser.add_argument("--mouth_amp", default=0.4) # nearest | random
    parser.add_argument("--raymarching_end_threshold", default=0.01, help="increase it to improve fps") # nearest | random
    parser.add_argument("--debug", action='store_true') 
    parser.add_argument("--fast", action='store_true') 
    
    # parser.add_argument("--out_name", default='tmp.mp4') 
    parser.add_argument("--out_name", default='tmp.mp4') 
    
    parser.add_argument("--low_memory_usage", action='store_true', help='write img to video upon generated, leads to slower fps, but use less memory')


    args = parser.parse_args()

    inp = {
            'a2m_ckpt': args.a2m_ckpt,
            'postnet_ckpt': args.postnet_ckpt,
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'drv_audio_name': args.drv_aud,
            'drv_pose': args.drv_pose,
            'blink_mode': args.blink_mode,
            'temperature': float(args.temperature),
            'mouth_amp': args.mouth_amp,
            'lle_percent': float(args.lle_percent),
            'debug': args.debug,
            'out_name': args.out_name,
            'raymarching_end_threshold': args.raymarching_end_threshold,
            'low_memory_usage': args.low_memory_usage,
            }
    if args.fast:
        inp['raymarching_end_threshold'] = 0.05
    GeneFace2Infer.example_run(inp)
