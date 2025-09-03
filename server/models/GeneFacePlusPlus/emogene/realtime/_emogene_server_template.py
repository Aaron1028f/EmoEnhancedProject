import os, sys
sys.path.append('./')
import argparse
import gradio as gr
from emogene.realtime.emogene_stream import GeneFace2Infer
from utils.commons.hparams import hparams
import random
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import warnings
import torchvision
import tempfile
import shutil

torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

class GenerateRequest(BaseModel):
    audio_path: str

# prepare global variables
args = None 
inferer_instance = None

MODEL_INPUT_MAY = {
    # input output setting
    'out_name': "emogene/DATA/lk_temp.mp4",
    'drv_audio_name': "emogene/DATA/happy.wav",
    
    # model path params
    'audio2secc': 'checkpoints/audio2motion_vae',
    'postnet_dir': '',
    'head_model_dir': '',
    'torso_model_dir': 'checkpoints/motion2video_nerf/may_torso', 
    'use_emotalk': True,
    'device': 'cuda:0',
    
    # emogene settings
    'blend_path': "emotalk/render_testing_92.blend",
    'lm468_bs_np_path': "emotalk/temp_result/lm468_bs_np.npy",
    'bs_lm_area': 8,
    'debug': False,
    'use_emotalk': True,
    'level': 1,
    'person': 3,
    'output_video': False,
    'bs52_level': 2.0,
    
    # GeneFace++ seettings
    'blink_mode': 'none',
    'drv_pose': 'nearest',
    'lle_percent': 1,
    'temperature': 0,
    'mouth_amp': 0.4,
    'raymarching_end_threshold': 0.01,
    'fp16': False,
    'low_memory_usage': False
}


# FastAPI application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # START
    global args, inferer_instance
    
    print("Initializing model...")
    inferer_instance = GeneFace2Infer(
        audio2secc_dir=MODEL_INPUT_MAY['audio2secc'],
        postnet_dir=MODEL_INPUT_MAY['postnet_dir'],
        head_model_dir=MODEL_INPUT_MAY['head_model_dir'],
        torso_model_dir=MODEL_INPUT_MAY['torso_model_dir'],
        use_emotalk=MODEL_INPUT_MAY['use_emotalk'],
        device=MODEL_INPUT_MAY['device']
    )
    print("Model loaded.")

    # prewarm: run once with
    print('Start run once to prewarm the model')
    inp = MODEL_INPUT_MAY.copy()
    inp['drv_audio_name'] = "emogene/DATA/happy.wav"
    inferer_instance.infer_once(inp)
    print('Prewarming complete.')

    print("Application startup complete.")
    yield # App running
    
    # END
    print("Application shutting down. Cleaning up resources...")
    inferer_instance = None

app = FastAPI(lifespan=lifespan)

@app.post("/generate_full_video")
async def generate_full_video_api(request: GenerateRequest):
    """
    Generate a full video from the given audio file.
    """
    if not os.path.exists(request.audio_path):
        return {"error": f"Audio file not found: {request.audio_path}", "video_path": None}
    
    # set the input audio path and output video path
    inp = MODEL_INPUT_MAY.copy()
    inp['drv_audio_name'] = request.audio_path

    try:
        print("Starting inference for API request...")
        video_path = inferer_instance.infer_once(inp)
        print(f"API inference successful. Video at: {video_path}")
        return {"video_path": video_path, "error": None}
    except Exception as e:
        print(f"API inference failed: {e}")
        return {"video_path": None, "error": str(e)}
    
@app.post("/generate_streaming_video")
async def generate_streaming_video(request: GenerateRequest):
    """
    Generate a streaming video from the given audio file.
    """
    pass
    # if not os.path.exists(request.audio_path):
    #     return {"error": f"Audio file not found: {request.audio_path}", "video_path": None}

    # # set the input audio path and output video path
    # inp = MODEL_INPUT_MAY.copy()
    # inp['drv_audio_name'] = request.audio_path

    # try:
    #     print("Starting inference for API request...")
    #     video_path = inferer_instance.infer_once(inp)
    #     print(f"API inference successful. Video at: {video_path}")
    #     return {"video_path": video_path, "error": None}
    # except Exception as e:
    #     print(f"API inference failed: {e}")
    #     return {"video_path": None, "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=31000) 
    parser.add_argument("--server", type=str, default='0.0.0.0') 
    cli_args, _ = parser.parse_known_args()
    
    print("Starting FastAPI server...")
    uvicorn.run(app, host=cli_args.server, port=cli_args.port)