import os, sys
sys.path.append('./')
import argparse
import gradio as gr
from emogene.gene import GeneFace2Infer
from utils.commons.hparams import hparams
import random
import time
import uvicorn # <--- 1. 引入 uvicorn
from fastapi import FastAPI # <--- 2. 引入 FastAPI
from pydantic import BaseModel # <--- 3. 引入 Pydantic 用於定義請求模型
from contextlib import asynccontextmanager

# --- 修正開始: 抑制警告 ---
import warnings
import torchvision

# 抑制 torchvision 的 Beta 版本警告
# 根據警告訊息的建議，這是安全的作法
torchvision.disable_beta_transforms_warning()

# 抑制 librosa 中關於 pkg_resources 的棄用警告
# 這是一個上游套件的問題，我們可以安全地忽略它
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
# --- 修正結束: 抑制警告 ---



# (Inferer 類別和相關輔助函式 toggle_audio_file, ref_video_fn, generate_random_uuid 保持不變)
# ... 你的 Inferer 類別 ...
class Inferer(GeneFace2Infer):
    def infer_once_args(self, *args, **kargs):
        # ... 這個函式保持不變，因為 Gradio UI 仍在使用它 ...
        assert len(kargs) == 0      
        keys = [
            'drv_audio_name', 'blink_mode', 'temperature', 'lle_percent', 
            'mouth_amp', 'raymarching_end_threshold', 'fp16', 'a2m_ckpt', 
            'postnet_ckpt', 'head_ckpt', 'torso_ckpt', 'low_memory_usage',
            'use_emotalk', 'debug', 'blend_path', 'level', 'person', 
            'output_video', 'bs52_level', 'bs_lm_area'
        ]
        inp = {}
        out_name = None
        info = ""
        print(f"args {args}")
        try:
            for key_index in range(len(keys)):
                key = keys[key_index]
                inp[key] = args[key_index]
                if '_name' in key or '_ckpt' in key:
                    inp[key] = inp[key] if inp[key] is not None else ''
            if inp['drv_audio_name'] == '':
                info = "Input Error: Driving audio is REQUIRED!"
                raise ValueError
            inp['drv_pose'] = 'nearest'  
            reload_flag = False
            if inp['a2m_ckpt'] != self.audio2secc_dir: reload_flag = True
            if inp['postnet_ckpt'] != self.postnet_dir: reload_flag = True
            if inp['head_ckpt'] != self.head_model_dir: reload_flag = True
            if inp['torso_ckpt'] != self.torso_model_dir: reload_flag = True
            inp['out_name'] = f"temp/out_{generate_random_uuid()}.mp4"
            print(f"infer inputs : {inp}")
            try:
                if reload_flag:
                    self.__init__(inp['a2m_ckpt'], inp['postnet_ckpt'], inp['head_ckpt'], inp['torso_ckpt'], inp['use_emotalk'])
            except Exception as e: info = f"Reload ERROR: {e}"
            try:
                out_name = self.infer_once(inp)
            except Exception as e: info = f"Inference ERROR: {e}"
        except Exception as e:
            if info == "": info = f"WebUI ERROR: {e}"
        if len(info) > 0 :
            print(info)
            info_gr = gr.update(visible=True, value=info)
        else:
            info_gr = gr.update(visible=False, value=info)
        if out_name is not None and len(out_name) > 0 and os.path.exists(out_name):
            print(f"Succefully generated in {out_name}")
            video_gr = gr.update(visible=True, value=out_name)
        else:
            print(f"Failed to generate")
            video_gr = gr.update(visible=True, value=out_name)
        return video_gr, info_gr

def generate_random_uuid(len_uuid=16):
    prev_state = random.getstate()
    random.seed(time.time())
    s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    res = ''.join(random.choices(s, k=len_uuid))
    random.setstate(prev_state)
    return res
# ... 其他輔助函式 ...
def toggle_audio_file(choice):
    if choice == False: return gr.update(visible=True), gr.update(visible=False)
    else: return gr.update(visible=False), gr.update(visible=True)
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None: return gr.update(value=True)
    else: return gr.update(value=False)
    
# (genefacepp_demo 函式保持不變)
def genefacepp_demo(infer_obj, audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir, use_emotalk, device='cuda', warpfn=None):
    # ... 這個函式幾乎不變，但注意 infer_obj 是傳入的 ...
    # ... 這樣 API 和 Gradio 可以共享同一個模型實例 ...
    sep_line = "-" * 40

    # infer_obj = Inferer(
    #     audio2secc_dir=audio2secc_dir, 
    #     postnet_dir=postnet_dir,
    #     head_model_dir=head_model_dir,
    #     torso_model_dir=torso_model_dir,
    #     use_emotalk=use_emotalk,
    #     device=device,
    # )

    print(sep_line)
    print("Model loading is finished.")
    print(sep_line)
    with gr.Blocks(analytics_enabled=False) as genefacepp_interface:
        gr.Markdown("\
            <div align='center'> <h2> GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking Face Generation </span> </h2> \
            <a style='font-size:18px;color: #a0a0a0' href='https://arxiv.org/pdf/2305.00787.pdf'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'> Github </div>")
        
        sources = None
        
        
        
        with gr.Tabs(elem_id="genearted_video"):
            info_box = gr.Textbox(label="Error", interactive=False, visible=False)
            gen_video = gr.Video(label="Generated video", format="mp4", visible=True)    
            
            submit = gr.Button('Generate', elem_id="generate", variant='primary')
        
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="driven_audio"):
                    with gr.TabItem('Upload audio'):
                        with gr.Column(variant='panel'):
                            drv_audio_name = gr.Audio(label="Input audio (required)", sources=sources, type="filepath", value='data/raw/val_wavs/MacronSpeech.wav')
                             
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('General Settings'):
                        gr.Markdown("need help? please visit our [best practice page](https://baidu.com) for more detials")
                        with gr.Column(variant='panel'):
                            
                            # some emogene inputs
                            gr.Markdown("### Emogene Settings")
                            bs_lm_area = gr.Slider(minimum=0, maximum=2, step=1, label="BS Landmarks Area", value=1, info="The area of the landmarks to be used for BS generation. 0 means not using BS, 1 means the whole face, 2 means face without mouth.")

                            # some emotalk inputs
                            gr.Markdown("### EmoTalk and Other Additional Settings (optional)")
                            debug = gr.Checkbox(label="Debug Mode", value=False, info="Whether to enable debug mode.")
                            
                            use_emotalk = gr.Checkbox(label="Use EmoTalk to generate expressive animation", value=use_emotalk, info="Whether to use EmoTalk to generate expressive animation. If not checked, the generated video will be neutral.")
                            level = gr.Slider(minimum=0, maximum=1, step=1, label="EmoTalk Level", value=1, info="The level of emotion to be generated by EmoTalk. 0 means neutral, 1 means full emotion.")
                            person = gr.Slider(minimum=0, maximum=23, step=1, label="EmoTalk Person", value=1, info="The person to be generated by EmoTalk. 0 means the first person, 1 means the second person, and so on.")
                            output_video = gr.Checkbox(label="Output video", value=False, info="Whether to generate blender video.")
                            bs52_level = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, label="BS52 Level", value=3.0, info="The multiplier for the BS52 blendshape. Higher value means more exaggerated expression. Default is 3.0.")
                            
                            # GeneFace++ settings
                            gr.Markdown("### GeneFace++ Settings")
                            
                            blink_mode = gr.Radio(['none', 'period'], value='none', label='blink mode', info="whether to blink periodly")       
                            # blink_mode = gr.Radio(['none', 'period'], value='period', label='blink mode', info="whether to blink periodly")       
                            lle_percent = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="lle_percent",  value=0.0, info='higher-> drag pred.landmark closer to the training video\'s landmark set',)
                            temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="temperature",  value=0.0, info='audio to secc temperature',)
                            mouth_amp = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="mouth amplitude",  value=0.4, info='higher -> mouth will open wider, default to be 0.4',)
                            raymarching_end_threshold = gr.Slider(minimum=0.0, maximum=0.1, step=0.0025, label="ray marching end-threshold",  value=0.01, info='increase it to improve inference speed',)
                            fp16 = gr.Checkbox(label="Whether to utilize fp16 to speed up inference")
                            low_memory_usage = gr.Checkbox(label="Low Memory Usage Mode: save memory at the expense of lower inference speed. Useful when running a low audio (minutes-long).", value=False)
                            
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('Checkpoints'):
                        with gr.Column(variant='panel'):
                            ckpt_info_box = gr.Textbox(value="Please select \"ckpt\" under the checkpoint folder ", interactive=False, visible=True, show_label=False)
                            
                            audio2secc_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=audio2secc_dir, file_count='single', label='audio2secc model ckpt path or directory')
                            torso_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=torso_model_dir, file_count='single', label='torso model ckpt path or directory')
                            blend_path = gr.FileExplorer(glob="emotalk/*.blend", value="emotalk/render_testing_92.blend", file_count='single', label='blend file path for emotalk')
                            
                            postnet_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=postnet_dir, file_count='single', label='(optional) pose net model ckpt path or directory')
                            head_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=head_model_dir, file_count='single', label='(optional) head model ckpt path or directory (will be ignored if torso model is set)')
                            

        fn = infer_obj.infer_once_args
        if warpfn:
            fn = warpfn(fn)
        submit.click(
                    fn=fn, 
                    inputs=[ 
                        drv_audio_name,
                        blink_mode,
                        temperature,
                        lle_percent,
                        mouth_amp,
                        raymarching_end_threshold,
                        fp16,
                        audio2secc_dir,
                        postnet_dir,
                        head_model_dir,
                        torso_model_dir,
                        low_memory_usage,
                        # other added inputs
                        use_emotalk,
                        debug,
                        # emotalk inputs
                        blend_path,
                        level,
                        person,
                        output_video,
                        bs52_level,
                        # emogene inputs
                        bs_lm_area,
                    ], 
                    outputs=[
                        gen_video,
                        info_box,
                    ],
                    queue=False,
                    )

    print(sep_line)
    print("Gradio page is constructed.")
    print(sep_line)

    return genefacepp_interface

# --- FastAPI 整合開始 ---

# 4. 定義 API 請求的資料結構
class GenerateRequest(BaseModel):
    audio_path: str
    mouth_amp: float = 0.4
    temperature: float = 0.0
    # 你可以把所有需要的參數都在這裡定義，並給予預設值
    # 這樣外部程式呼叫時可以只傳送必要的 audio_path
    blink_mode: str = 'none'
    lle_percent: float = 0.0
    fp16: bool = False
    low_memory_usage: bool = False
    
    # extra
    output_dir: str = 'output'

# 5. 建立 FastAPI 應用
app = FastAPI()

# 6. 在全域範圍內初始化模型，避免每次請求都重新載入
#    這一步非常重要，能大幅提升 API 回應速度
args = None 
inferer_instance = None

# --- 修正開始 ---

# --- 修正開始: 統一啟動邏輯 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在應用程式啟動時載入模型、建立並掛載 Gradio UI。
    """
    global args, inferer_instance
    
    # --- 1. 解析參數 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", type=str, default='checkpoints/audio2motion_vae/model_ckpt_steps_400000.ckpt')
    parser.add_argument("--postnet_ckpt", type=str, default='')
    parser.add_argument("--head_ckpt", type=str, default='')
    parser.add_argument("--torso_ckpt", type=str, default='checkpoints/motion2video_nerf/may_torso/model_ckpt_steps_250000.ckpt') 
    parser.add_argument("--port", type=int, default=7860) 
    parser.add_argument("--server", type=str, default='0.0.0.0') 
    parser.add_argument("--use_emotalk", default=True, action='store_true')
    parser.add_argument("--blend_path", type=str, default='emotalk/render_testing_92.blend')
    args, _ = parser.parse_known_args()
    print("Command line arguments parsed.")

    # --- 2. 載入模型 ---
    print("Initializing model...")
    inferer_instance = Inferer(
        audio2secc_dir=args.a2m_ckpt, 
        postnet_dir=args.postnet_ckpt,
        head_model_dir=args.head_ckpt,
        torso_model_dir=args.torso_ckpt,
        use_emotalk=args.use_emotalk,
        device='cuda:0',
    )
    print("Model loaded.")

    # --- 3. 建立並掛載 Gradio UI ---
    demo = genefacepp_demo(
        infer_obj=inferer_instance,
        audio2secc_dir=args.a2m_ckpt,
        postnet_dir=args.postnet_ckpt,
        head_model_dir=args.head_ckpt,
        torso_model_dir=args.torso_ckpt,
        use_emotalk=args.use_emotalk,
    )
    
    # demo.queue()
    
    # 使用 gr.mount_gradio_app 將 Gradio 掛載到 FastAPI
    # 注意：這個函式會直接修改傳入的 app 物件
    gr.mount_gradio_app(app, demo, path="/ui")
    print("Gradio UI mounted at /ui.")
    
    print("Application startup complete.")
    yield # 應用程式在此處運行
    
    # --- 關閉時執行的程式碼 ---
    print("Application shutting down. Cleaning up resources...")
    inferer_instance = None
# --- 修正結束: 統一啟動邏輯 ---

# 2. 建立 FastAPI 應用，並傳入 lifespan 處理器
app = FastAPI(lifespan=lifespan)

# --- 修正結束 ---

# 7. 建立 API 端點
@app.post("/generate_video")
async def generate_video_api(request: GenerateRequest):
    """
    接收音訊路徑，生成影片並回傳影片路徑。
    """
    if not os.path.exists(request.audio_path):
        return {"error": f"Audio file not found: {request.audio_path}", "video_path": None}

    # 準備傳給 infer_once 的字典
    # inp = {
    #     'drv_audio_name': request.audio_path,
    #     'blink_mode': request.blink_mode,
    #     'temperature': request.temperature,
    #     'lle_percent': request.lle_percent,
    #     'mouth_amp': request.mouth_amp,
    #     'fp16': request.fp16,
    #     'low_memory_usage': request.low_memory_usage,
        
    #     # 使用從命令列載入的預設模型路徑
    #     'a2m_ckpt': args.a2m_ckpt,
    #     'postnet_ckpt': args.postnet_ckpt,
    #     'head_ckpt': args.head_ckpt,
    #     'torso_ckpt': args.torso_ckpt,
        
    #     # 填充其他必要的預設值
    #     'drv_pose': 'nearest',
    #     'raymarching_end_threshold': 0.01,
    #     'use_emotalk': args.use_emotalk,
    #     'debug': False,
    #     'blend_path': args.blend_path,
    #     'level': 1,
    #     'person': 1,
    #     'output_video': False,
    #     'bs52_level': 3.0,
    #     'bs_lm_area': 0,
    #     # 確保每次生成都有獨立的輸出檔名
    #     # 'out_name': f"temp/api_out_{generate_random_uuid()}.mp4"
    #     # 'out_name': "/home/ykwei/project/server/services/demo_video.mp4"
    #     'out_name': request.output_dir
    # }
    inp = {
        # file path
        'drv_audio_name': request.audio_path,
        'out_name': request.output_dir,
        
        # fixed arguments of GeneFace++
        'blink_mode': None,
        'temperature': 0.2, 
        'lle_percent': 1,
        'mouth_amp': 0.4,
        'fp16': False, 
        'low_memory_usage': False, 
        'drv_pose': 'nearest',
        'raymarching_end_threshold': 0.01,        
        'debug': False,
        
        # model arguments (在 lifespan() 中更改 )
        'a2m_ckpt': args.a2m_ckpt,
        'postnet_ckpt': args.postnet_ckpt,
        'head_ckpt': args.head_ckpt,
        'torso_ckpt': args.torso_ckpt,
        
        'blend_path': args.blend_path,
        
        # fixed arguments of EmoGene
        'use_emotalk': True,
        'output_video': False,
        'level': 1,
        'person': 3,
        'bs52_level': 2.5, # 2.5 for May, maybe 1.5 for Feng
        'bs_lm_area': 8, # 8 for May, 9 for Feng
    }
    
    try:
        print(f"API request payload: {inp}")
        print("Starting inference for API request...")
        # 直接呼叫核心推論函式
        video_path = inferer_instance.infer_once(inp)
        print(f"API inference successful. Video at: {video_path}")
        return {"video_path": video_path, "error": None}
    except Exception as e:
        print(f"API inference failed: {e}")
        return {"video_path": None, "error": str(e)}

# --- FastAPI 整合結束 ---

if __name__ == "__main__":
    # 解析參數僅為了取得 host 和 port，因為 uvicorn.run 需要它們
    # 真正的參數解析和模型載入在 lifespan 中進行
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000) 
    parser.add_argument("--server", type=str, default='0.0.0.0') 
    cli_args, _ = parser.parse_known_args()
    
    print("Starting FastAPI server...")
    uvicorn.run(app, host=cli_args.server, port=cli_args.port)