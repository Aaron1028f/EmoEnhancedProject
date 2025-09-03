# simple infer for EmoGene
# input: audio
# output: video


import os, sys
sys.path.append('./')
import argparse
import gradio as gr
# from inference.genefacepp_infer import GeneFace2Infer
from server.models.GeneFacePlusPlus.emogene.realtime.emogene_stream import GeneFace2Infer # [修改] 引用新的串流文件from utils.commons.hparams import hparams
from utils.commons.hparams import hparams
import random
import time
import subprocess # [新增] 導入 subprocess 模組
import traceback

class Inferer(GeneFace2Infer):
    def infer_once_args(self, *args, **kargs):
        assert len(kargs) == 0        
        
        keys = [
            'drv_audio_name', 'blink_mode', 'temperature', 'lle_percent', 'mouth_amp',
            'raymarching_end_threshold', 'fp16', 'a2m_ckpt', 'postnet_ckpt', 'head_ckpt',
            'torso_ckpt', 'low_memory_usage', 'use_emotalk', 'debug', 'blend_path',
            'level', 'person', 'output_video', 'bs52_level', 'bs_lm_area',
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
                
            if reload_flag:
                print("Changes of ckpt detected, reloading model...")
                self.__init__(inp['a2m_ckpt'], inp['postnet_ckpt'], inp['head_ckpt'], inp['torso_ckpt'], inp['use_emotalk'])
            
            # [修改] 核心串流邏輯
            # 1. 準備 batch data
            samples = self.prepare_batch_from_inp(inp)
            
            # 2. 準備 ffmpeg 命令
            audio_path = self.wav16k_name
            output_path = inp['out_name']
            
            command = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', '512x512', '-pix_fmt', 'rgb24', '-r', '25', '-i', '-',
                '-i', audio_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest', output_path
            ]

            # 3. 啟動 ffmpeg 子程序，並將其 stdin 設定為一個管道
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 4. 迭代生成器，將每一幀寫入 ffmpeg 的 stdin
            frame_count = 0
            for frame in self.stream_forward_system(samples, inp):
                try:
                    process.stdin.write(frame.tobytes())
                    frame_count += 1
                except (IOError, BrokenPipeError) as e:
                    print(f"ffmpeg process pipe broken: {e}")
                    break
            
            print(f"Streaming finished. Total frames processed: {frame_count}")
            
            # 5. 關閉管道並等待 ffmpeg 完成
            process.stdin.close()
            stderr_output = process.stderr.read().decode('utf-8')
            process.wait()
            if process.returncode != 0:
                info = f"FFmpeg Error: {stderr_output}"
                print(info)
            else:
                out_name = output_path

        except Exception as e:
            if info == "":
                info = f"WebUI/Inference ERROR: {e}\n{traceback.format_exc()}"
        
        # Gradio UI 更新
        if len(info) > 0:
            print(info)
            info_gr = gr.update(visible=True, value=info)
        else:
            info_gr = gr.update(visible=False, value=info)
            
        if out_name and os.path.exists(out_name):
            print(f"Successfully generated in {out_name}")
            video_gr = gr.update(visible=True, value=out_name)
        else:
            print(f"Failed to generate video.")
            video_gr = gr.update(visible=False)
            
        # Gradio 的生成器函數需要 yield 結果
        yield video_gr, info_gr

def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

# generate random uuid and do not disturb global random state
def generate_random_uuid(len_uuid = 16):
    prev_state = random.getstate()
    random.seed(time.time())
    s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    res = ''.join(random.choices(s, k=len_uuid))
    random.setstate(prev_state)
    return res

def genefacepp_demo(
    audio2secc_dir,
    postnet_dir,
    head_model_dir,
    torso_model_dir, 
    use_emotalk,
    # blend_path,
    device          = 'cuda',
    warpfn          = None,
    ):

    sep_line = "-" * 40

    infer_obj = Inferer(
        audio2secc_dir=audio2secc_dir, 
        postnet_dir=postnet_dir,
        head_model_dir=head_model_dir,
        torso_model_dir=torso_model_dir,
        use_emotalk=use_emotalk,
        device=device,
    )

    print(sep_line)
    print("Model loading is finished.")
    print(sep_line)
    with gr.Blocks(analytics_enabled=False) as genefacepp_interface:
        # gr.Markdown("\
        #     <div align='center'> <h2> GeneFace++: Generalized and Stable Real-Time Audio-Driven 3D Talking Face Generation </span> </h2> \
        #     <a style='font-size:18px;color: #a0a0a0' href='https://arxiv.org/pdf/2305.00787.pdf'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
        #     <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
        #     <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'> Github </div>")
        
        gr.Markdown("\
            <div align='center'> <h2> EmoGene: Combining EmoTalk and GeneFace++ to generate expressive 3D Talking Face based on audio emotion</span> </h2> </div> ")
            # \
            # <a style='font-size:18px;color: #a0a0a0' href='https://arxiv.org/pdf/2305.00787.pdf'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            # <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            # <a style='font-size:18px;color: #a0a0a0' href='https://baidu.com'> Github </div>")

        sources = None
               
        with gr.Tabs(elem_id="genearted_video"):
            info_box = gr.Textbox(label="Error", interactive=False, visible=False)
            gen_video = gr.Video(label="Generated video", format="mp4", visible=True)    
            
            submit = gr.Button('Generate', elem_id="generate", variant='primary')
        
        # 定義 CSS：利用 elem_id 選取內部 video 元素並限制高度
        # css = """
        # #generated_video video {
        #     max-height: 400px;
        # }
        # """

        # with gr.Blocks(analytics_enabled=False, css=css) as genefacepp_interface:
        #     gr.Markdown("## GeneFace++ Demo")
            
        #     with gr.Tabs(elem_id="genearted_video"):
        #         info_box = gr.Textbox(label="Error", interactive=False, visible=False)
        #         # 加上 elem_id 以便使用 CSS 選取這個元件的內部 video 元素
        #         gen_video = gr.Video(label="Generated video", format="mp4", visible=True, elem_id="generated_video")
                
        #         submit = gr.Button('Generate', elem_id="generate", variant='primary')

        
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="driven_audio"):
                    with gr.TabItem('Upload audio'):
                        with gr.Column(variant='panel'):
                            drv_audio_name = gr.Audio(label="Input audio (required)", sources=sources, type="filepath", value='data/raw/val_wavs/MacronSpeech.wav')
                             
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('General Settings'):
                        # gr.Markdown("need help? please visit our [best practice page](https://baidu.com) for more detials")
                        with gr.Column(variant='panel'):
                            
                            # some emogene inputs
                            gr.Markdown("### Emogene Settings")
                            bs_lm_area_info_str = """
                            The area of the landmarks to be used for BS generation.\n
                            0 means not using BS,\n
                            1 means the whole face,\n
                            2 means face without inner mouth.\n
                            3 means face without the whole mouth.\n
                            4 means using both the displacement of GeneFace and the emotalk lm468.\n
                            5 means using displacement of GeneFace for eye and eyebrow, and emotalk lm468 for the rest.\n
                            6 means dynamic ratio of GeneFace and emotalk lm468\n
                            7 means 6, but just use GeneFace eye landmarks.\n
                            """
                            bs_lm_area = gr.Slider(minimum=0, maximum=9, step=1, label="BS Landmarks Area", value=8, info=bs_lm_area_info_str)

                            # some emotalk inputs
                            gr.Markdown("### EmoTalk and Other Additional Settings (optional)")
                            debug = gr.Checkbox(label="Debug Mode", value=True, info="Whether to enable debug mode.")
                            
                            use_emotalk = gr.Checkbox(label="Use EmoTalk to generate expressive animation", value=use_emotalk, info="Whether to use EmoTalk to generate expressive animation. If not checked, the generated video will be neutral.")
                            level = gr.Slider(minimum=0, maximum=1, step=1, label="EmoTalk Level", value=1, info="The level of emotion to be generated by EmoTalk. 0 means neutral, 1 means full emotion.")
                            person = gr.Slider(minimum=0, maximum=23, step=1, label="EmoTalk Person", value=3, info="The person to be generated by EmoTalk. 0 means the first person, 1 means the second person, and so on.")
                            output_video = gr.Checkbox(label="Output video", value=False, info="Whether to generate blender video.")
                            bs52_level = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, label="BS52 Level", value=2.0, info="The multiplier for the BS52 blendshape. Higher value means more exaggerated expression. Default is 2.0.")
                            
                            # GeneFace++ settings
                            gr.Markdown("### GeneFace++ Settings")
                            
                            blink_mode = gr.Radio(['none', 'period'], value='none', label='blink mode', info="whether to blink periodly")       
                            # blink_mode = gr.Radio(['none', 'period'], value='period', label='blink mode', info="whether to blink periodly")       
                            lle_percent = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="lle_percent",  value=1.0, info='higher-> drag pred.landmark closer to the training video\'s landmark set',)
                            temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="temperature",  value=0.0, info='audio to secc temperature',)
                            mouth_amp = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="mouth amplitude",  value=0.4, info='higher -> mouth will open wider, default to be 0.4',)
                            raymarching_end_threshold = gr.Slider(minimum=0.0, maximum=0.1, step=0.0025, label="ray marching end-threshold",  value=0.01, info='increase it to improve inference speed',)
                            fp16 = gr.Checkbox(label="Whether to utilize fp16 to speed up inference")
                            low_memory_usage = gr.Checkbox(label="Low Memory Usage Mode: save memory at the expense of lower inference speed. Useful when running a low audio (minutes-long).", value=False)
                            
                            # submit = gr.Button('Generate', elem_id="generate", variant='primary')
                        
                    # with gr.Tabs(elem_id="genearted_video"):
                    #     info_box = gr.Textbox(label="Error", interactive=False, visible=False)
                    #     gen_video = gr.Video(label="Generated video", format="mp4", visible=True)
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
                    )

    print(sep_line)
    print("Gradio page is constructed.")
    print(sep_line)

    return genefacepp_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", type=str, default='checkpoints/audio2motion_vae/model_ckpt_steps_400000.ckpt')
    parser.add_argument("--postnet_ckpt", type=str, default='')
    parser.add_argument("--head_ckpt", type=str, default='')
    parser.add_argument("--torso_ckpt", type=str, default='checkpoints/motion2video_nerf/may_torso/model_ckpt_steps_250000.ckpt') 
    # parser.add_argument("--torso_ckpt", default='checkpoints/motion2video_nerf/feng_torso')
    
    parser.add_argument("--port", type=int, default=None) 
    parser.add_argument("--server", type=str, default='127.0.0.1') 
    
    parser.add_argument("--use_emotalk", default=True, action='store_true')
    # parser.add_argument("--blend_path", type=str, default='emotalk/feng_rigged.blend', help="the blend file path for emotalk, if not set, will use the default one")
    parser.add_argument("--blend_path", type=str, default='emotalk/render_testing_92.blend', help="the blend file path for emotalk, if not set, will use the default one")
    

    args = parser.parse_args()
    demo = genefacepp_demo(
        audio2secc_dir=args.a2m_ckpt,
        postnet_dir=args.postnet_ckpt,
        head_model_dir=args.head_ckpt,
        torso_model_dir=args.torso_ckpt,
        use_emotalk=args.use_emotalk,
        # blend_path=args.blend_path,
        device='cuda:0',
        warpfn=None,
    )
    demo.queue()
    demo.launch(server_name=args.server, server_port=args.port)
    # demo.launch(server_name=args.server, server_port=args.port, share=True)
    