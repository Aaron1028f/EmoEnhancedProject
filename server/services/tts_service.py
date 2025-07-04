# step1: open the tts server
# cd server/models/tts_model/CosyVoice
# pyton runtime/python/modified_fastapi/server.py

# step2: parse the input and send the request to the tts server
# code copy and modified from CosyVoice runtime/python/fastapi/client.py

import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np

def main(args, prompt_sr=16000, target_sr=22050):
    url = "http://{}:{}/inference_{}".format(args.host, args.port, args.mode)
    if args.mode == 'sft':
        payload = {
            'tts_text': args.tts_text,
            'spk_id': args.spk_id
        }
        response = requests.request("GET", url, data=payload, stream=True)
    elif args.mode == 'zero_shot':
        payload = {
            'tts_text': args.tts_text,
            'prompt_text': args.prompt_text
        }
        files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
        response = requests.request("GET", url, data=payload, files=files, stream=True)
    elif args.mode == 'cross_lingual':
        payload = {
            'tts_text': args.tts_text,
        }
        files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
        response = requests.request("GET", url, data=payload, files=files, stream=True)
    else:
        payload = {
            'tts_text': args.tts_text,
            'spk_id': args.spk_id,
            'instruct_text': args.instruct_text
        }
        response = requests.request("GET", url, data=payload, stream=True)
    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    logging.info('save response to {}'.format(args.tts_wav))
    torchaudio.save(args.tts_wav, tts_speech, target_sr)
    logging.info('get response')
    
def parse_args(inp_text):
    inp_temp={
        'host': '0.0.0.0',
        'port': 50000,
        'mode': 'sft',
        # 'tts_text': '你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？',
        'tts_text': inp_text,
        'spk_id': '中文女',
        'prompt_text': '希望你以后能够做的比我还好呦。',
        # 'prompt_wav': '../../../asset/zero_shot_prompt.wav',
        'prompt_wav': 'server/models/tts_model/CosyVoice/asset/zero_shot_prompt.wav',
        'instruct_text': 'Theo \'Crimson\', is a fiery, passionate rebel leader. \
                         Fights with fervor for justice, but struggles with impulsiveness.',
        # 'tts_wav': 'demo.wav'
        'tts_wav': "/home/ykwei/project/server/services/demo_audio.wav"
    }
    # transform the input dictionary to argparse scope
    infer_args = argparse.Namespace(**inp_temp)
    main(infer_args)
    
    return infer_args.tts_wav


def cosyvoice_tts(inp_text):
    """
    Function to call the CosyVoice TTS service with the given input text.
    :param inp_text: Input text for TTS
    :return: Path to the generated audio file
    """
    # Parse the input text and get the audio file path
    audio_file_path = parse_args(inp_text)
    
    # Return the absolute path of the audio file
    # audio_file_path = f"server/models/tts_model/CosyVoice/{audio_file_path}"
    
    return audio_file_path
    
