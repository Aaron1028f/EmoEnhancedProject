import sys
sys.path.append('./')

import argparse
import librosa
from scipy.signal import savgol_filter
import torch
import numpy as np
import os, subprocess, shlex

from emotalk.model import EmoTalk

@torch.no_grad()
def generate_bs52(model, args):    
    # =========================================================================================
    # use the following code to add blinking effect
    # eye1 = np.array([0.36537236, 0.950235724, 0.95593375, 0.916715622, 0.367256105, 0.119113259, 0.025357503])
    # eye2 = np.array([0.234776169, 0.909951985, 0.944758058, 0.777862132, 0.191071674, 0.235437036, 0.089163929])
    # eye3 = np.array([0.870040774, 0.949833691, 0.949418545, 0.695911646, 0.191071674, 0.072576277, 0.007108896])
    # eye4 = np.array([0.000307991, 0.556701422, 0.952656746, 0.942345619, 0.425857186, 0.148335218, 0.017659493])
    # =========================================================================================
    # prepare the model (run this only once)
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)    
    wav_path = args.wav_path
    file_name = wav_path.split('/')[-1].split('.')[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    audio = torch.FloatTensor(speech_array).unsqueeze(0).to(args.device)
    l = args.level # range[0, 1]
    p = args.person # range[0, 23]
    level = torch.tensor([l]).to(args.device)
    person = torch.tensor([p]).to(args.device)
    
    prediction = model.predict(audio, level, person)
    prediction = prediction.squeeze().detach().cpu().numpy()
    
    if args.post_processing:
        # smoothing
        output = np.zeros((prediction.shape[0], prediction.shape[1]))
        for i in range(prediction.shape[1]):
            output[:, i] = savgol_filter(prediction[:, i], 5, 2)
            
        # add blinking effect
        # output[:, 8] = 0
        # output[:, 9] = 0
        # i = random.randint(0, 60)
        # while i < output.shape[0] - 7:
        #     eye_num = random.randint(1, 4)
        #     if eye_num == 1:
        #         output[i:i + 7, 8] = eye1
        #         output[i:i + 7, 9] = eye1
        #     elif eye_num == 2:
        #         output[i:i + 7, 8] = eye2
        #         output[i:i + 7, 9] = eye2
        #     elif eye_num == 3:
        #         output[i:i + 7, 8] = eye3
        #         output[i:i + 7, 9] = eye3
        #     else:
        #         output[i:i + 7, 8] = eye4
        #         output[i:i + 7, 9] = eye4
        #     time1 = random.randint(60, 180)
        #     i = i + time1
        np.save(os.path.join(result_path, "{}.npy".format(file_name)), output)  # with postprocessing (smoothing and blinking)
    else:
        np.save(os.path.join(result_path, "{}.npy".format(file_name)), prediction)  # without post-processing
        
    return

def render_lm468(args):
    # get the landmarks output
    wav_name = args.wav_path.split('/')[-1].split('.')[0]
    image_path = os.path.join(args.result_path, wav_name)
    os.makedirs(image_path, exist_ok=True)
    image_temp = image_path + "/%d.png"
    output_path = os.path.join(args.result_path, wav_name + ".mp4")
    blender_path = args.blender_path
    
    # set the paths
    result_path = args.result_path
    python_path = args.python_render_path
    blend_path = args.blend_path
    
    bs52_level = args.bs52_level
    lm468_bs_np_path = args.lm468_bs_np_path
    output_video = args.output_video
    
    # join the paths
    cur_dir = os.getcwd()
    # os.path.join(cur_dir, result_path)
    # os.path.join(cur_dir, python_path)
    # os.path.join(cur_dir, blend_path)
    # os.path.join(cur_dir, lm468_bs_np_path)
    blender_path = cur_dir + "/" + blender_path
    result_path = cur_dir + "/" + result_path
    python_path = cur_dir + "/" + python_path
    # if blend_path is already an absolute path, we don't need to change it
    if not os.path.isabs(blend_path):
        # if blend_path is relative, we need to join it with the current directory
        os.path.join(cur_dir, blend_path)
    # blend_path = cur_dir + "/" + blend_path
    lm468_bs_np_path = cur_dir + "/" + lm468_bs_np_path
    

    # print("="*80)
    # print(f"blender_path: {blender_path}")
    # print(f"python_path: {python_path}")
    # print(f"blend_path: {blend_path}")
    # print(f"result_path: {result_path}")
    # print(f"wav_name: {wav_name}")
    # print(f"lm468_bs_np_path: {lm468_bs_np_path}")
    # print(f"output_video: {output_video}")
    # print(f"bs52_level: {bs52_level}")
    # print("="*80)

    # cmd = '{} -t 64 -b {} -P {} -- "{}" "{}" '.format(blender_path, blend_path, python_path, args.result_path, wav_name)
    cmd = f"{blender_path} --background {blend_path} --threads 64 --python {python_path} -- {output_video} {lm468_bs_np_path} {bs52_level} {result_path} {wav_name}"

    cmd = shlex.split(cmd)
    p = subprocess.Popen(
        cmd,
        cwd=cur_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False
    )
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print('[{}]'.format(line))
                        
    if p.returncode == 0:
        print('Subprogram success')
    else:
        print('Subprogram failed')
        
    if args.output_video:
        cmd = 'ffmpeg -r 30 -i "{}" -i "{}" -pix_fmt yuv420p -s 512x768 "{}" -y'.format(image_temp, args.wav_path, output_path)        
        subprocess.call(cmd, shell=True)

        cmd = 'rm -rf "{}"'.format(image_path)
        subprocess.call(cmd, shell=True)

def infer_emo_lm468(model=None, inp=None):
    # set the arguments (temp)
    inp_temp = {
        # used when loading the model
        "feature_dim": 832,
        "bs_dim": 52,
        "device": "cuda",
        "batch_size": 1,
        "max_seq_len": 5000,
        "period": 30,
        "model_path": "emotalk/pretrain_model/EmoTalk.pth",
        
        # used when inferencing
        "wav_path": "_test/emo_wav/angry2.wav",  # will be updated when inferencing
        "result_path": "emotalk/temp_result/",
        "num_workers": 0,
        "post_processing": True,
        "blender_path": "emotalk/blender_ver_3_6/blender",
        "python_render_path": "emogene/emo_render.py", # !!
        # "blend_path": "emotalk/render_testing_92.blend",
        "blend_path": "emotalk/feng_rigged.blend",
        
        "level": 1,
        "person": 1,
        "output_video": False,
        "bs52_level": 3,
        "lm468_bs_np_path": "emotalk/temp_result/lm468_bs_np.npy" # !!
    }

    # update the input arguments
    if inp is not None:
        inp_temp.update(inp)
    inp = inp_temp
    infer_args = argparse.Namespace(**inp)
    
    # generate blendshape
    generate_bs52(model, infer_args)
    
    # generate emo_lm468/ EmoTalk video
    render_lm468(infer_args)
    
    # transform the lm468_bs_np.npy to tensor
    emo_lm468_np = np.load(inp["lm468_bs_np_path"])
    emo_lm468_tensor = torch.from_numpy(emo_lm468_np).float().cuda()
    emo_lm468_tensor = emo_lm468_tensor * 10
    
    # return the lm468 predicted from EmoTalk
    return emo_lm468_tensor

@torch.no_grad()
def load_emotalk_model(model_path="emotalk/pretrain_model/EmoTalk.pth"):
    # set the arguments
    load_model_args = {
        "feature_dim": 832,
        "bs_dim": 52,
        "device": "cuda",
        "batch_size": 1,
        "max_seq_len": 5000,
        "period": 30,
        "model_path": model_path,
    }
    args = argparse.Namespace(**load_model_args)
    
    # load the model
    model = EmoTalk(args)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)), strict=False)
    model = model.to(args.device)
    model.eval()
    
    return model
    
if __name__ == '__main__':
    # used for debugging
    emo_model = load_emotalk_model()
    infer_emo_lm468()