from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
import os
import re
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from loguru import logger
import sys
import json
import traceback
import time

import httpx
import asyncio
import base64

# 簡體轉繁體
from opencc import OpenCC
# s2twp: Simplified to Traditional Chinese (Taiwan standard, with phrases)
cc = OpenCC('s2twp')
# cc = OpenCC('s2t')
# ==

logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)


class Config(BaseSettings):
    sv_thr: float = Field(0.3, description="Speaker verification threshold")
    chunk_size_ms: int = Field(300, description="Chunk size in milliseconds")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    bit_depth: int = Field(16, description="Bit depth")
    channels: int = Field(1, description="Number of audio channels")
    avg_logprob_thr: float = Field(-0.25, description="average logprob threshold")

config = Config()

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def contains_chinese_english_number(s: str) -> bool:
    # Check if the string contains any Chinese character, English letter, or Arabic number
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',
    model_revision='v1.0.0'
)

asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
    device="cuda:0",
    disable_update=True
)

model_asr = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",    
    device="cuda:0",
    disable_update=True
)

model_vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar = True,
    max_end_silence_time=500,
    # speech_noise_thres=0.6,
    disable_update=True,
)

reg_spks_files = [
    "speaker/speaker1_a_cn_16k.wav"
]

def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        reg_spk[k] = {
            "data": data,
            "sr":   sr,
        }
    return reg_spk

reg_spks = reg_spk_init(reg_spks_files)

def speaker_verify(audio, sv_thr):
    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], sv_thr)
        if res_sv["score"] >= sv_thr:
           hit = True
        logger.info(f"[speaker_verify] audio_len: {len(audio)}; sv_thr: {sv_thr}; hit: {hit}; {k}: {res_sv}")
    return hit, k


def asr(audio, lang, cache, use_itn=False):
    # with open('test.pcm', 'ab') as f:
    #     logger.debug(f'write {f.write(audio)} bytes to `test.pcm`')
    # result = asr_pipeline(audio, lang)
    start_time = time.time()
    result = model_asr.generate(
        input           = audio,
        cache           = cache,
        language        = lang.strip(),
        use_itn         = use_itn,
        batch_size_s    = 60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
        ).model_dump()
    )
    
# # --- 新增一個函式，專門用來呼叫 LLM API ---
# async def get_llm_response_stream(text: str, websocket: WebSocket):
#     """呼叫 roleplay_api 並將 LLM 的回應即時串流回客戶端"""
#     # 假設 roleplay_api.py 運行在 localhost:8000
#     llm_api_url = "http://localhost:28000/streaming_response"
#     try:
#         async with httpx.AsyncClient() as client:
#             async with client.stream("GET", llm_api_url, params={"user_input": text}, timeout=60) as response:
#                 # 檢查 API 呼叫是否成功
#                 if response.status_code != 200:
#                     error_payload = {"type": "error", "payload": f"LLM API Error: {response.status_code}"}
#                     await websocket.send_json(error_payload)
#                     return

#                 # 將收到的 LLM 文字流，包裝成我們的格式，再轉發給客戶端
#                 async for chunk in response.aiter_text():
#                     if chunk:
#                         llm_chunk_payload = {"type": "llm_chunk", "payload": chunk}
#                         await websocket.send_json(llm_chunk_payload)
#     except httpx.RequestError as e:
#         error_payload = {"type": "error", "payload": f"Could not connect to LLM API: {e}"}
#         await websocket.send_json(error_payload)    
        
# --- 新增一個函式，專門用來呼叫 TTS API 並串流音訊 ---
async def get_tts_audio_stream(text: str, websocket: WebSocket):
    """呼叫 TTS API (api_v2.py) 並將音訊串流回客戶端"""
    # 根據 api_v2.py 的文件設定參數
    REF_AUDIO_PATH = '/home/aaron/project/server/models/TTS/GPT-SoVITS/DATA/Feng_EP32/slicer/Feng_live_EP32.wav_0004170880_0004326720.wav'
    tts_api_url = "http://127.0.0.1:9880/tts"
    params = {
        "text": text,
        "text_lang": "zh",  # 假設 LLM 回覆是中文
        "ref_audio_path": REF_AUDIO_PATH, # 您需要提供一個參考音色
        "prompt_lang": "zh",
        "streaming_mode": True,
        "media_type": "wav", # 串流時建議用 wav 或 raw
        "parallel_infer": False
    }
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", tts_api_url, params=params, timeout=60) as response:
                if response.status_code != 200:
                    logger.error(f"TTS API Error: {response.status_code} - {await response.aread()}")
                    return

                # 接收音訊塊，用 Base64 編碼後轉發
                async for audio_chunk in response.aiter_bytes():
                    if audio_chunk:
                        # 將原始二進位音訊塊用 Base64 編碼成字串
                        encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                        tts_payload = {"type": "tts_audio_chunk", "payload": encoded_audio}
                        await websocket.send_json(tts_payload)
    except httpx.RequestError as e:
        logger.error(f"Could not connect to TTS API: {e}")
        error_payload = {"type": "error", "payload": f"Could not connect to TTS API: {e}"}
        await websocket.send_json(error_payload)


# --- 重新設計主流程函式，整合 LLM 和 TTS ---
async def run_llm_and_tts_pipeline(text: str, websocket: WebSocket):
    """
    呼叫 LLM API，將其文字流轉發給客戶端，同時湊成句子送去 TTS API 合成語音。
    """
    llm_api_url = "http://localhost:28000/streaming_response"
    sentence_buffer = ""
    # 定義句子的結束符號
    sentence_enders = {"。", "！", "？", "...", "，"}

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", llm_api_url, params={"user_input": text}, timeout=60) as response:
                if response.status_code != 200:
                    error_payload = {"type": "error", "payload": f"LLM API Error: {response.status_code}"}
                    await websocket.send_json(error_payload)
                    return

                async for llm_chunk in response.aiter_text():
                    if llm_chunk:
                        # 1. 立即將文字塊轉發給客戶端顯示
                        llm_payload = {"type": "llm_chunk", "payload": llm_chunk}
                        await websocket.send_json(llm_payload)

                        # 2. 將文字塊加入句子緩衝區
                        sentence_buffer += llm_chunk

                        # 3. 檢查緩衝區中是否有完整的句子
                        while any(ender in sentence_buffer for ender in sentence_enders):
                            first_ender_pos = -1
                            first_ender = ''
                            for ender in sentence_enders:
                                pos = sentence_buffer.find(ender)
                                if pos != -1 and (first_ender_pos == -1 or pos < first_ender_pos):
                                    first_ender_pos = pos
                                    first_ender = ender
                            
                            if first_ender_pos != -1:
                                # 提取一個完整的句子
                                sentence_to_speak = sentence_buffer[:first_ender_pos + len(first_ender)]
                                # 從緩衝區移除已提取的句子
                                sentence_buffer = sentence_buffer[first_ender_pos + len(first_ender):]
                                
                                # 4. 將完整句子送去 TTS 合成 (在背景任務中執行)
                                logger.info(f"Sending to TTS: {sentence_to_speak}")
                                asyncio.create_task(get_tts_audio_stream(sentence_to_speak, websocket))

                # 處理最後剩餘的文字 (如果有的話)
                if sentence_buffer.strip():
                    logger.info(f"Sending final part to TTS: {sentence_buffer}")
                    asyncio.create_task(get_tts_audio_stream(sentence_buffer, websocket))

    except httpx.RequestError as e:
        error_payload = {"type": "error", "payload": f"Could not connect to LLM API: {e}"}
        await websocket.send_json(error_payload)        

# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    info: str
    data: str

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    try:
        query_params = parse_qs(websocket.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['auto'])[0].lower()
        
        await websocket.accept()
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)

        cache = {}
        cache_asr = {}
        last_vad_beg = last_vad_end = -1
        offset = 0
        hit = False
        
        buffer = b""
        while True:
            data = await websocket.receive_bytes()
            # logger.info(f"received {len(data)} bytes")

            
            buffer += data
            if len(buffer) < 2:
                continue
                
            audio_buffer = np.append(
                audio_buffer, 
                np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
            )
            
            # with open('buffer.pcm', 'ab') as f:
            #     logger.debug(f'write {f.write(buffer[:len(buffer) - (len(buffer) % 2)])} bytes to `buffer.pcm`')
                
            buffer = buffer[len(buffer) - (len(buffer) % 2):]
   
            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)
                
                # with open('chunk.pcm', 'ab') as f:
                #     logger.debug(f'write {f.write(chunk)} bytes to `chunk.pcm`')
                    
                if last_vad_beg > 1:
                    if sv:
                        # speaker verify
                        # If no hit is detected, continue accumulating audio data and check again until a hit is detected
                        # `hit` will reset after `asr`.
                        if not hit:
                            hit, speaker = speaker_verify(audio_vad[int((last_vad_beg - offset) * config.sample_rate / 1000):], config.sv_thr)
                            if hit:
                                response = TranscriptionResponse(
                                    code=2,
                                    info="detect speaker",
                                    data=speaker
                                )
                                await websocket.send_json(response.model_dump())
                    else:
                        response = TranscriptionResponse(
                            code=2,
                            info="detect speech",
                            data=''
                        )
                        await websocket.send_json(response.model_dump())

                res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                # logger.info(f"vad inference: {res}")
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1: # speech begin
                            last_vad_beg = segment[0]                           
                        if segment[1] > -1: # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")
                            result = None if sv and not hit else asr(audio_vad[beg:end], lang.strip(), cache_asr, True)
                            logger.info(f"asr response: {result}")
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False
                            
                            if  result is not None:
                                # 簡體轉繁體
                                simplified_text = format_str_v3(result[0]['text'])
                                traditional_text = cc.convert(simplified_text)
                                # ==
                                
                                # 將 ASR 結果用新的格式傳送給客戶端
                                asr_payload = {"type": "asr_result", "payload": traditional_text}
                                await websocket.send_json(asr_payload)                                
                                # ==
                                
                                # await get_llm_response_stream(traditional_text, websocket)
                                # asyncio.create_task(get_llm_response_stream(traditional_text, websocket))
                                asyncio.create_task(run_llm_and_tts_pipeline(traditional_text, websocket))
                                
                                # (未來擴展) 4. 在這裡可以加入呼叫 TTS 的邏輯
                                # tts_audio = await get_tts_audio(llm_full_response)
                                # tts_payload = {"type": "tts_audio", "payload": base64.b64encode(tts_audio).decode()}
                                # await websocket.send_json(tts_payload)                                
                                
                                # ==================================================
                                # 原始僅ASR傳送方式
                                # response = TranscriptionResponse(
                                #     code=0,
                                #     info=json.dumps(result[0], ensure_ascii=False),
                                #     # data=format_str_v3(result[0]['text'])
                                #     data=traditional_text
                                # )
                                # await websocket.send_json(response.model_dump())
                                # ==================================================
                                
                        # logger.debug(f'last_vad_beg: {last_vad_beg}; last_vad_end: {last_vad_end} len(audio_vad): {len(audio_vad)}')

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\nCall stack:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    # parser.add_argument('--certfile', type=str, default='path_to_your_SSL_certificate_file.crt', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_SSL_certificate_file.key', help='SSL key file')
    args = parser.parse_args()
    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
