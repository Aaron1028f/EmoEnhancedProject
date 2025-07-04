# geneface++ call functions used here
import sys
sys.path.append('./')

import librosa
import numpy as np
import argparse
from scipy.signal import savgol_filter
import torch
from emotalk.model import EmoTalk
import os, subprocess
import random
import shlex

@torch.no_grad()
def test(args):
    
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
    
    model = EmoTalk(args)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)), strict=False)
    model = model.to(args.device)
    model.eval()
    # =========================================================================================
    
    
    wav_path = args.wav_path
    file_name = wav_path.split('/')[-1].split('.')[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    audio = torch.FloatTensor(speech_array).unsqueeze(0).to(args.device)
    
    # =========================================================================================
    # infer the output blendshape coefficients
    
    # get the level and person arguments
    l = args.level # range[0, 1]
    p = args.person # range[0, 23]
    
    # level = torch.tensor([1]).to(args.device)
    # person = torch.tensor([0]).to(args.device)
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


def render_video(args):
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
    blend_path = cur_dir + "/" + blend_path
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
    # p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p = subprocess.Popen(
        cmd,
        cwd=cur_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False
    )
    
    # num_frames = 
    # frame_counter = 0
    
    
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print('[{}]'.format(line))
        # if line:
        #     if line[0:7] == 'b"Saved':
        #         pass
        #     elif line[0:6] == "b'Time":
        #         pass
        #     else:
        #         print('[{}]'.format(line))
                        
    if p.returncode == 0:
        print('Subprogram success')
    else:
        print('Subprogram failed')
        
        
    if args.output_video:
        cmd = 'ffmpeg -r 30 -i "{}" -i "{}" -pix_fmt yuv420p -s 512x768 "{}" -y'.format(image_temp, args.wav_path, output_path)        
        subprocess.call(cmd, shell=True)

        cmd = 'rm -rf "{}"'.format(image_path)
        subprocess.call(cmd, shell=True)

# =========================================================================================
def infer_by_func(level=1, person=0, bs52_level = 5, output_video=True, wav_name=''):
    """
    level: int, 0 or 1
    person: int, 0, 1, 2, 3
    bs52_level: int, 0 ~ n
    output_video: bool
    wav_name: str
    """
    # set the parameters
    params = {
        "wav_path": "./audio/angry2.wav",
        "bs_dim": 52,
        "feature_dim": 832,
        "period": 30,
        "device": "cuda",
        "model_path": "./pretrain_model/EmoTalk.pth",
        "result_path": "./result/",
        "max_seq_len": 5000,
        "num_workers": 0,
        "batch_size": 1,
        "post_processing": True,
        
        "blender_path": "./blender_ver_3_6/blender",
        "python_render_path": "./render_lm3d.py",
        "blend_path": "./render_testing_5.blend",
        # "blender_path": "/home/aaron/genefacepp_project/EmoTalk_release/blender_ver_3_6",
        # "python_render_path": "/home/aaron/genefacepp_project/EmoTalk_release/render_testing.py",
        # "blend_path": "/home/aaron/genefacepp_project/EmoTalk_release/render_testing_5.blend",
        
        "level": level,
        "person": person,
        "output_video": output_video,
        "bs52_level": bs52_level
    }
    if wav_name != '':  
        params["wav_path"] = './audio/' + wav_name
        
    # transform args to argparse.Namespace
    args = argparse.Namespace(**params)

    # run
    test(args)
    render_video(args)
    
    # get the result
    lm468_bs_np = np.load("lm468_temp.npy")
    return lm468_bs_np
    # return output_video

def main():
    # get the arguments
    input_wav_path = "_test/emo_wav/angry2.wav"
    
    # basic arguments
    parser = argparse.ArgumentParser(
        description='EmoTalk: Speech-driven Emotional Disentanglement for 3D Face Animation')
    parser.add_argument("--wav_path", type=str, default="_test/emo_wav/angry2.wav", help='path of the test data')
    
    parser.add_argument("--bs_dim", type=int, default=52, help='number of blendshapes:52')
    parser.add_argument("--feature_dim", type=int, default=832, help='number of feature dim')    
    parser.add_argument("--period", type=int, default=30, help='number of period')
    parser.add_argument("--device", type=str, default="cuda", help='device')
    parser.add_argument("--model_path", type=str, default="emotalk/pretrain_model/EmoTalk.pth",
                        help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="emotalk/temp_result/", help='path of the result')
    parser.add_argument("--max_seq_len", type=int, default=5000, help='max sequence length')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--post_processing", type=bool, default=True, help='whether to use post processing')
    parser.add_argument("--blender_path", type=str, default="emotalk/blender_ver_3_6/blender", help='path of blender')
    
    # extra arguments
    parser.add_argument("--python_render_path", type=str, default="emotalk/render_lm3d.py", help='path of python_render_path')
    parser.add_argument("--blend_path", type=str, default="emotalk/render_testing_92.blend", help='path of blend_path')
    parser.add_argument("--level", type=int, default=1, help='emotion level(0, 1)')
    parser.add_argument("--person", type=int, default=0, help='person style(0, 1, 2, ...,23)')
    parser.add_argument("--output_video", type=bool, default=True, help='whether to output video')
    parser.add_argument("--bs52_level", type=float, default=3, help='bs52_level')
    parser.add_argument("--lm468_bs_np_path", type=str, default="emotalk/temp_result/lm468_bs_np_path.npy", help='lm468_bs_np_path')
    
    args = parser.parse_args()
    
    test(args)
    render_video(args)

if __name__ == "__main__":
    # infer_by_func()
    # print(os.getcwd())
    main()
    
