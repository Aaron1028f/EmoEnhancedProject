import torch
import numpy as np
import matplotlib.pyplot as plt
from emogene.tools.face_landmarker_bs import index_lm68_from_lm478, index_lm131_from_lm478

def analyze_lm_displacement_limit(idexp_lm3d_ds_lle, idexp_lm3d, idexp_lm3d_geneface, mean_face_exp):
    """
    Analyze the landmark displacement limit for GeneFace++.
    This function is used to check the limits of landmark displacements
    in the GeneFace++ model, particularly focusing on the mouth open distance
    and other key landmarks.
    """
    
    # mouth openness part
    # mouth_open_distance_before_lle() # this one should be called in face3d_helper_bs.py
    mouth_open_distance_after_lle(idexp_lm3d_ds_lle, idexp_lm3d, idexp_lm3d_geneface, mean_face_exp)
    
    # save the plot for mouth open distance
    plt.legend()
    plt.savefig('/home/aaron/project/server/models/GeneFacePlusPlus/emogene/experiment/lip_lm_limit/mouth_open_distance.png', dpi=300, bbox_inches='tight')
    plt.close()    
    
    # =============================================================================================
    # eyebrow displacement part
    eyebrow_displacement(idexp_lm3d_ds_lle, idexp_lm3d, idexp_lm3d_geneface, mean_face_exp)
    
    # eye openness part
    eye_open_distance()
    
def eye_open_distance():
    """
    Analyze the eye open distance limit for GeneFace++.
    This function is used to check the limits of eye openness in the GeneFace++ model.
    """
    
    
    
def eyebrow_displacement(idexp_lm3d_ds_lle, idexp_lm3d, idexp_lm3d_geneface, mean_face_exp):
    """
    Analyze the eyebrow displacement limit for GeneFace++.
    This function is used to check the limits of eyebrow displacements
    in the GeneFace++ model.
    """
    # Find Maximum eyebrow up and down displacement in dataset in comparison to the position in mean face
    LM68_RIGHT_EYEBROW_INDEX = 20  # index for right eyebrow in lm68
    # the eyebrow position in the mean face
    right_eyebrow_mean_pos = mean_face_exp[:, LM68_RIGHT_EYEBROW_INDEX, :]
    # max eyebrow up displacement in dataset
    max_right_eyebrow_up_ds = 0
    max_right_eyebrow_down_ds = 0
    # the eyebrow position in the dataset
    idexp_lm3d_ds_lle_exp = idexp_lm3d_ds_lle.reshape([-1, 68, 3])
    
    # clamp to avoid unusable values
    lower = torch.quantile(idexp_lm3d_ds_lle_exp, q=0.03, dim=0)
    upper = torch.quantile(idexp_lm3d_ds_lle_exp, q=0.97, dim=0)
    idexp_lm3d_ds_lle_exp = torch.clamp(idexp_lm3d_ds_lle_exp, min=lower, max=upper)
    
    right_eyebrow_ds_pos = idexp_lm3d_ds_lle_exp[:, LM68_RIGHT_EYEBROW_INDEX, :] / 10 + mean_face_exp[:, LM68_RIGHT_EYEBROW_INDEX, :]
    # compute the displacement
    displacement = torch.norm(right_eyebrow_ds_pos - right_eyebrow_mean_pos, dim=-1)    
    for i in range(idexp_lm3d_ds_lle_exp.shape[0]):
        # print(f'right_eyebrow_ds_pos shape: {right_eyebrow_ds_pos.shape}'): torch.Size([6073, 3])
        if displacement[i] > max_right_eyebrow_up_ds and idexp_lm3d_ds_lle_exp[i, LM68_RIGHT_EYEBROW_INDEX, 2] > 0:
            max_right_eyebrow_up_ds = displacement[i]
        elif displacement[i] > max_right_eyebrow_down_ds and idexp_lm3d_ds_lle_exp[i, LM68_RIGHT_EYEBROW_INDEX, 2] < 0:
            max_right_eyebrow_down_ds = displacement[i]
    max_right_eyebrow_down_ds = (-1)*max_right_eyebrow_down_ds
    # plot the max up and max down displacement in dataset
    plot_max_up = torch.tensor([max_right_eyebrow_up_ds]).reshape([-1, 1]).repeat([len(idexp_lm3d), 1])
    plot_max_down = torch.tensor([max_right_eyebrow_down_ds]).reshape([-1, 1]).repeat([len(idexp_lm3d), 1])
    plt.plot(plot_max_up.cpu().numpy(), color='yellow', linewidth=1.0, label='Max Up in dataset')
    plt.plot(plot_max_down.cpu().numpy(), color='yellow', linewidth=1.0, label='Max Down in dataset')
    
    # the eyebrow displacement in the mixed Emogene model
    idexp_lm3d_exp = idexp_lm3d / 10
    right_eyebrow_displacement_emogene = torch.zeros(idexp_lm3d_exp.shape[0])
    emogene_eyebrow_displacement = torch.norm(idexp_lm3d_exp[:, LM68_RIGHT_EYEBROW_INDEX, :], dim=-1)
    print(f'emogene_eyebrow_displacement shape: {emogene_eyebrow_displacement.shape}')  # [t]
    print(f'idexp_lm3d_exp shape: {idexp_lm3d_exp.shape}')  # torch.Size([t,N, 3]
    print(f'right_eyebrow_displacement_emogene shape: {right_eyebrow_displacement_emogene.shape}')  # torch.Size([t, 3])
    for i in range(idexp_lm3d_exp.shape[0]):
        if idexp_lm3d_exp[i, LM68_RIGHT_EYEBROW_INDEX, 2] > 0:
            right_eyebrow_displacement_emogene[i] = emogene_eyebrow_displacement[i]
        else:
            right_eyebrow_displacement_emogene[i] = -emogene_eyebrow_displacement[i]
    plt.plot(right_eyebrow_displacement_emogene.cpu().numpy(), color='green', linewidth=1.0, label='final Emogene')
    
    # the eyebrow displacement in the original GeneFace++ model
    idexp_lm3d_geneface_exp = idexp_lm3d_geneface / 10
    right_eyebrow_displacement_geneface = torch.zeros(idexp_lm3d_geneface_exp.shape[0])
    geneface_eyebrow_displacement = torch.norm(idexp_lm3d_geneface_exp[:, LM68_RIGHT_EYEBROW_INDEX, :], dim=-1)
    print(f'geneface_eyebrow_displacement shape: {geneface_eyebrow_displacement.shape}')  # [t]
    for i in range(idexp_lm3d_geneface_exp.shape[0]):
        if idexp_lm3d_geneface_exp[i, LM68_RIGHT_EYEBROW_INDEX, 2] > 0:
            right_eyebrow_displacement_geneface[i] = geneface_eyebrow_displacement[i]
        else:
            right_eyebrow_displacement_geneface[i] = -geneface_eyebrow_displacement[i]
            
    plt.plot(right_eyebrow_displacement_geneface.cpu().numpy(), color='black', linewidth=1.0, label='final GeneFace++') 
    plt.title('Eyebrow Displacement')
    plt.xlabel('Frame')
    plt.ylabel('Displacement')
    plt.grid()
    plt.legend()
    plt.savefig('/home/aaron/project/server/models/GeneFacePlusPlus/emogene/experiment/lip_lm_limit/eyebrow_displacement.png', dpi=300, bbox_inches='tight')
    plt.close()


def mouth_open_distance_before_lle(geneface_displacement, emotalk_displacement, mean_face):
    # preprocess
    geneface_displacement = geneface_displacement[:, index_lm68_from_lm478, :]
    emotalk_displacement = emotalk_displacement[:, index_lm68_from_lm478, :]
    mean_face = mean_face[:, index_lm68_from_lm478, :]
    # compute mouth open distance of geneface model
    face_real_lm_geneface = geneface_displacement + mean_face
    upper_lip_center_geneface = face_real_lm_geneface[:, 62, :]  # 62 is the index for upper inner lip center in lm68
    lower_lip_center_geneface = face_real_lm_geneface[:, 66, :]  # 66 is the index for lower inner lip center in lm68
    mouth_open_distance_geneface = torch.norm(upper_lip_center_geneface - lower_lip_center_geneface, dim=-1)
    plt.plot(mouth_open_distance_geneface.cpu().numpy(), color='blue', linewidth=1.0, label='raw GeneFace++')
    plt.title('Mouth Open Distance')
    plt.xlabel('Frame')
    plt.ylabel('Distance')
    plt.grid()
    
    # compute mouth open distance of mixed Emogene model
    face_emogene_displacement = geneface_displacement + emotalk_displacement
    
    face_real_lm_emogene = face_emogene_displacement + mean_face
    upper_lip_center_emogene = face_real_lm_emogene[:, 62, :]
    lower_lip_center_emogene = face_real_lm_emogene[:, 66, :]
    mouth_open_distance_emogene = torch.norm(upper_lip_center_emogene - lower_lip_center_emogene, dim=-1)
    plt.plot(mouth_open_distance_emogene.cpu().numpy(), color='red', linewidth=1.0, label='raw Emogene')
    

def mouth_open_distance_after_lle(idexp_lm3d_ds_lle, idexp_lm3d, idexp_lm3d_geneface, mean_face_exp):
    # draw the mouth largest and smallest open distance in the dataset
    idexp_lm3d_ds_lle_exp = idexp_lm3d_ds_lle.reshape([-1, 68, 3])
    face_real_lm_ds_exp = idexp_lm3d_ds_lle_exp/10 + mean_face_exp
    upper_lip_ds_exp = face_real_lm_ds_exp[:, 62, :]
    lower_lip_ds_exp = face_real_lm_ds_exp[:, 66, :]
    mouth_open_distance_ds_exp = torch.norm(upper_lip_ds_exp - lower_lip_ds_exp, dim=-1)
    # transform the largest number to a tensor and repeat it for the number of frames
    # the largest
    print(f'mouth_open_distance_ds_exp max: {mouth_open_distance_ds_exp.max()}')
    mouth_open_distance_ds_exp_max = torch.tensor(mouth_open_distance_ds_exp.max()).reshape([-1, 1]).repeat([len(idexp_lm3d), 1])
    # the smallest
    print(f'mouth_open_distance_ds_exp min: {mouth_open_distance_ds_exp.min()}')
    mouth_open_distance_ds_exp_min = torch.tensor(mouth_open_distance_ds_exp.min()).reshape([-1, 1]).repeat([len(idexp_lm3d), 1])
    plt.plot(mouth_open_distance_ds_exp_max.cpu().numpy(), color='yellow', linewidth=1.0, label='Max in dataset')
    plt.plot(mouth_open_distance_ds_exp_min.cpu().numpy(), color='yellow', linewidth=1.0, label='min in dataset')
    
    # draw the mouth open distance in the mixed Emogene model
    face_real_lm_exp = idexp_lm3d/10 + mean_face_exp
    upper_lip_exp = face_real_lm_exp[:, 62, :]
    lower_lip_exp = face_real_lm_exp[:, 66, :]
    mouth_open_distance_exp = torch.norm(upper_lip_exp - lower_lip_exp, dim=-1)
    plt.plot(mouth_open_distance_exp.cpu().numpy(), color='green', linewidth=1.0, label='final Emogene')   
    
    # draw the mouth open distance in the original GeneFace++ model
    face_real_lm_geneface_exp = idexp_lm3d_geneface/10 + mean_face_exp
    upper_lip_geneface_exp = face_real_lm_geneface_exp[:, 62, :]
    lower_lip_geneface_exp = face_real_lm_geneface_exp[:, 66, :]
    mouth_open_distance_geneface_exp = torch.norm(upper_lip_geneface_exp - lower_lip_geneface_exp, dim=-1)
    plt.plot(mouth_open_distance_geneface_exp.cpu().numpy(), color='black', linewidth=1.0, label='final GeneFace++')
