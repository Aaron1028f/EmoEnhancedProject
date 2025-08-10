import os, sys
sys.path.append('./')
import argparse
import gradio as gr
from emogene.realtime.gene_stream import GeneFace2Infer
from utils.commons.hparams import hparams
import random
import time
import subprocess
import traceback
import uuid

class Inferer(GeneFace2Infer):
    def infer_once_args(self, *args, **kargs):
        assert 'stream_key' in kargs, "stream_key must be provided as a keyword argument."
        stream_key = kargs['stream_key']
        
        keys = [
            'drv_audio_name', 'blink_mode', 'temperature', 'lle_percent', 'mouth_amp',
            'raymarching_end_threshold', 'fp16', 'a2m_ckpt', 'postnet_ckpt', 'head_ckpt',
            'torso_ckpt', 'low_memory_usage', 'use_emotalk', 'debug', 'blend_path',
            'level', 'person', 'output_video', 'bs52_level', 'bs_lm_area'
        ]
        inp = {}
        info = ""
        
        yield gr.update(visible=True, value="準備中...")

        try:
            for key_index, key in enumerate(keys):
                inp[key] = args[key_index]
                if '_name' in key or '_ckpt' in key:
                    inp[key] = inp[key] if inp[key] is not None else ''
            
            if not inp['drv_audio_name']:
                raise ValueError("錯誤：必須提供驅動音訊！")

            inp['drv_pose'] = 'nearest'
            samples = self.prepare_batch_from_inp(inp)
            audio_path = self.wav16k_name
            
            if not stream_key:
                 raise ValueError("Stream key is missing!")
            rtmp_url = f"rtmp://localhost:19350/live/{stream_key}"
            
            command = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', '512x512', '-pix_fmt', 'rgb24', '-r', '25', '-i', '-',
                '-i', audio_path, 
                '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
                '-c:a', 'aac', '-ar', '44100', '-f', 'flv',
                rtmp_url
            ]

            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
            frame_count = 0
            yield gr.update(visible=True, value=f"串流已開始，請在下方播放器中觀看。\n推流地址: {rtmp_url}")
            
            for frame in self.stream_forward_system(samples, inp):
                try:
                    process.stdin.write(frame.tobytes())
                    frame_count += 1
                except (IOError, BrokenPipeError) as e:
                    print(f"ffmpeg process pipe broken: {e}")
                    break
            
            process.stdin.close()
            stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
            process.wait()
            
            if process.returncode != 0:
                info = f"FFmpeg Error:\n{stderr_output}"
            else:
                info = f"串流成功結束。總共處理了 {frame_count} 幀。"

        except Exception as e:
            info = f"WebUI/Inference ERROR: {e}\n{traceback.format_exc()}"
        
        print(info)
        yield gr.update(visible=True, value=info)

def genefacepp_demo(
    audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir, 
    use_emotalk, device='cuda', warpfn=None
):
    sep_line = "-" * 40
    infer_obj = Inferer(
        audio2secc_dir=audio2secc_dir, postnet_dir=postnet_dir,
        head_model_dir=head_model_dir, torso_model_dir=torso_model_dir,
        use_emotalk=use_emotalk, device=device,
    )
    print(sep_line, "\nModel loading is finished.\n", sep_line)

    with gr.Blocks(analytics_enabled=False) as genefacepp_interface:
        gr.Markdown("<div align='center'> <h2> EmoGene: Realtime HLS Streaming Demo </span> </h2> </div>")
        
        with gr.Row():
            with gr.Column(scale=2):
                info_box = gr.Textbox(label="狀態", interactive=False, visible=True, value="等待開始...")
                
                player_html = f"""
                <div id="player-container">
                    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
                    <video id="video-player" controls style="width: 100%; height: auto;"></video>
                </div>
                <script>
                window.setupHlsPlayer = (streamKey) => {{
                    if (!streamKey) {{
                        console.error("Received an empty streamKey.");
                        return;
                    }}
                    const video = document.getElementById('video-player');
                    const hlsUrl = `http://${{window.location.hostname}}:8080/hls/${{streamKey}}.m3u8`;
                    console.log('Setting up HLS player for URL:', hlsUrl);

                    if (Hls.isSupported()) {{
                        const hls = new Hls({{
                            startFragPrefetch: true,
                            fragLoadingMaxRetry: 6,
                            levelLoadingMaxRetry: 4,
                            fragLoadingRetryDelay: 1000,
                            levelLoadingRetryDelay: 1000,
                        }});
                        hls.loadSource(hlsUrl);
                        hls.attachMedia(video);
                        hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                            video.play().catch(e => console.error("Autoplay was prevented:", e));
                        }});
                        hls.on(Hls.Events.ERROR, function (event, data) {{
                            if (data.fatal) {{ console.error('Fatal HLS error:', data); }}
                        }});
                    }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                        video.src = hlsUrl;
                        video.addEventListener('loadedmetadata', function() {{
                            video.play().catch(e => console.error("Autoplay was prevented:", e));
                        }});
                    }}
                }};
                </script>
                """
                gr.HTML(player_html)
                submit = gr.Button('Generate Stream', elem_id="generate", variant='primary')

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem('音訊與主要設定'):
                        drv_audio_name = gr.Audio(label="Input audio (required)", type="filepath", value='data/raw/val_wavs/MacronSpeech.wav')
                        bs_lm_area = gr.Slider(minimum=0, maximum=9, step=1, label="BS Landmarks Area", value=8)
                        debug = gr.Checkbox(label="Debug Mode", value=True)
                        use_emotalk = gr.Checkbox(label="Use EmoTalk", value=use_emotalk)
                    with gr.TabItem('進階設定'):
                        level = gr.Slider(minimum=0, maximum=1, step=1, label="EmoTalk Level", value=1)
                        person = gr.Slider(minimum=0, maximum=23, step=1, label="EmoTalk Person", value=3)
                        bs52_level = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, label="BS52 Level", value=2.0)
                        blink_mode = gr.Radio(['none', 'period'], value='none', label='blink mode')
                        lle_percent = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="lle_percent", value=1.0)
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="temperature", value=0.0)
                        mouth_amp = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="mouth amplitude", value=0.4)
                        raymarching_end_threshold = gr.Slider(minimum=0.0, maximum=0.1, step=0.0025, label="ray marching end-threshold", value=0.01)
                    with gr.TabItem('模型路徑'):
                        audio2secc_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=audio2secc_dir, file_count='single', label='audio2secc model')
                        torso_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=torso_model_dir, file_count='single', label='torso model')
                        blend_path = gr.FileExplorer(glob="emotalk/*.blend", value="emotalk/render_testing_92.blend", file_count='single', label='blend file for emotalk')
                        postnet_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=postnet_dir, file_count='single', label='(optional) pose net model')
                        head_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=head_model_dir, file_count='single', label='(optional) head model')
        
        stream_key_trigger = gr.Textbox(label="Stream Key Trigger", visible=False)

        all_inputs = [
            drv_audio_name, blink_mode, temperature, lle_percent, mouth_amp,
            raymarching_end_threshold, gr.State(False), audio2secc_dir, postnet_dir,
            head_model_dir, torso_model_dir, gr.State(False), use_emotalk,
            debug, blend_path, level, person, gr.State(False), bs52_level, bs_lm_area
        ]
        
        # [修改] 步驟 1: 點擊按鈕，只做一件事：生成 stream_key 並返回
        def generate_key():
            return str(uuid.uuid4())

        submit.click(
            fn=generate_key,
            inputs=[],
            outputs=[stream_key_trigger]
        )

        # [修改] 步驟 2: 當 stream_key_trigger 的值改變時，觸發兩件事：
        # 1. 透過 _js 執行前端播放器設定
        # 2. 透過 fn 執行後端真正的推流任務
        stream_key_trigger.change(
            fn=infer_obj.infer_once_args,
            inputs=all_inputs + [stream_key_trigger], # 將所有 UI 輸入和 trigger 一起傳入
            outputs=[info_box],
            # [關鍵] _js 參數用在 .change() 事件上，這是相容的
            # `x` 就是 trigger 的新值 (stream_key)
            _js="(x) => { window.setupHlsPlayer(x); }"
        )

    print(sep_line, "\nGradio page is constructed.\n", sep_line)
    return genefacepp_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", type=str, default='checkpoints/audio2motion_vae/model_ckpt_steps_400000.ckpt')
    parser.add_argument("--postnet_ckpt", type=str, default='')
    parser.add_argument("--head_ckpt", type=str, default='')
    parser.add_argument("--torso_ckpt", type=str, default='checkpoints/motion2video_nerf/may_torso/model_ckpt_steps_250000.ckpt') 
    parser.add_argument("--port", type=int, default=None) 
    parser.add_argument("--server", type=str, default='127.0.0.1') 
    parser.add_argument("--use_emotalk", default=True, action='store_true')
    parser.add_argument("--blend_path", type=str, default='emotalk/render_testing_92.blend')
    args = parser.parse_args()
    demo = genefacepp_demo(
        audio2secc_dir=args.a2m_ckpt,
        postnet_dir=args.postnet_ckpt,
        head_model_dir=args.head_ckpt,
        torso_model_dir=args.torso_ckpt,
        use_emotalk=args.use_emotalk,
        device='cuda:0',
        warpfn=None,
    )
    demo.queue()
    demo.launch(server_name=args.server, server_port=args.port)
# ```// filepath: /home/aaron/project/server/models/GeneFacePlusPlus/emogene/realtime/webui.py
# import os, sys
# sys.path.append('./')
# import argparse
# import gradio as gr
# from emogene.realtime.gene_stream import GeneFace2Infer
# from utils.commons.hparams import hparams
# import random
# import time
# import subprocess
# import traceback
# import uuid

# class Inferer(GeneFace2Infer):
#     def infer_once_args(self, *args, **kargs):
#         assert 'stream_key' in kargs, "stream_key must be provided as a keyword argument."
#         stream_key = kargs['stream_key']
        
#         keys = [
#             'drv_audio_name', 'blink_mode', 'temperature', 'lle_percent', 'mouth_amp',
#             'raymarching_end_threshold', 'fp16', 'a2m_ckpt', 'postnet_ckpt', 'head_ckpt',
#             'torso_ckpt', 'low_memory_usage', 'use_emotalk', 'debug', 'blend_path',
#             'level', 'person', 'output_video', 'bs52_level', 'bs_lm_area'
#         ]
#         inp = {}
#         info = ""
        
#         yield gr.update(visible=True, value="準備中...")

#         try:
#             for key_index, key in enumerate(keys):
#                 inp[key] = args[key_index]
#                 if '_name' in key or '_ckpt' in key:
#                     inp[key] = inp[key] if inp[key] is not None else ''
            
#             if not inp['drv_audio_name']:
#                 raise ValueError("錯誤：必須提供驅動音訊！")

#             inp['drv_pose'] = 'nearest'
#             samples = self.prepare_batch_from_inp(inp)
#             audio_path = self.wav16k_name
            
#             if not stream_key:
#                  raise ValueError("Stream key is missing!")
#             rtmp_url = f"rtmp://localhost:19350/live/{stream_key}"
            
#             command = [
#                 'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
#                 '-s', '512x512', '-pix_fmt', 'rgb24', '-r', '25', '-i', '-',
#                 '-i', audio_path, 
#                 '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
#                 '-c:a', 'aac', '-ar', '44100', '-f', 'flv',
#                 rtmp_url
#             ]

#             process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
#             frame_count = 0
#             yield gr.update(visible=True, value=f"串流已開始，請在下方播放器中觀看。\n推流地址: {rtmp_url}")
            
#             for frame in self.stream_forward_system(samples, inp):
#                 try:
#                     process.stdin.write(frame.tobytes())
#                     frame_count += 1
#                 except (IOError, BrokenPipeError) as e:
#                     print(f"ffmpeg process pipe broken: {e}")
#                     break
            
#             process.stdin.close()
#             stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
#             process.wait()
            
#             if process.returncode != 0:
#                 info = f"FFmpeg Error:\n{stderr_output}"
#             else:
#                 info = f"串流成功結束。總共處理了 {frame_count} 幀。"

#         except Exception as e:
#             info = f"WebUI/Inference ERROR: {e}\n{traceback.format_exc()}"
        
#         print(info)
#         yield gr.update(visible=True, value=info)

# def genefacepp_demo(
#     audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir, 
#     use_emotalk, device='cuda', warpfn=None
# ):
#     sep_line = "-" * 40
#     infer_obj = Inferer(
#         audio2secc_dir=audio2secc_dir, postnet_dir=postnet_dir,
#         head_model_dir=head_model_dir, torso_model_dir=torso_model_dir,
#         use_emotalk=use_emotalk, device=device,
#     )
#     print(sep_line, "\nModel loading is finished.\n", sep_line)

#     with gr.Blocks(analytics_enabled=False) as genefacepp_interface:
#         gr.Markdown("<div align='center'> <h2> EmoGene: Realtime HLS Streaming Demo </span> </h2> </div>")
        
#         with gr.Row():
#             with gr.Column(scale=2):
#                 info_box = gr.Textbox(label="狀態", interactive=False, visible=True, value="等待開始...")
                
#                 player_html = f"""
#                 <div id="player-container">
#                     <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
#                     <video id="video-player" controls style="width: 100%; height: auto;"></video>
#                 </div>
#                 <script>
#                 window.setupHlsPlayer = (streamKey) => {{
#                     if (!streamKey) {{
#                         console.error("Received an empty streamKey.");
#                         return;
#                     }}
#                     const video = document.getElementById('video-player');
#                     const hlsUrl = `http://${{window.location.hostname}}:8080/hls/${{streamKey}}.m3u8`;
#                     console.log('Setting up HLS player for URL:', hlsUrl);

#                     if (Hls.isSupported()) {{
#                         const hls = new Hls({{
#                             startFragPrefetch: true,
#                             fragLoadingMaxRetry: 6,
#                             levelLoadingMaxRetry: 4,
#                             fragLoadingRetryDelay: 1000,
#                             levelLoadingRetryDelay: 1000,
#                         }});
#                         hls.loadSource(hlsUrl);
#                         hls.attachMedia(video);
#                         hls.on(Hls.Events.MANIFEST_PARSED, function() {{
#                             video.play().catch(e => console.error("Autoplay was prevented:", e));
#                         }});
#                         hls.on(Hls.Events.ERROR, function (event, data) {{
#                             if (data.fatal) {{ console.error('Fatal HLS error:', data); }}
#                         }});
#                     }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
#                         video.src = hlsUrl;
#                         video.addEventListener('loadedmetadata', function() {{
#                             video.play().catch(e => console.error("Autoplay was prevented:", e));
#                         }});
#                     }}
#                 }};
#                 </script>
#                 """
#                 gr.HTML(player_html)
#                 submit = gr.Button('Generate Stream', elem_id="generate", variant='primary')

#             with gr.Column(scale=1):
#                 with gr.Tabs():
#                     with gr.TabItem('音訊與主要設定'):
#                         drv_audio_name = gr.Audio(label="Input audio (required)", type="filepath", value='data/raw/val_wavs/MacronSpeech.wav')
#                         bs_lm_area = gr.Slider(minimum=0, maximum=9, step=1, label="BS Landmarks Area", value=8)
#                         debug = gr.Checkbox(label="Debug Mode", value=True)
#                         use_emotalk = gr.Checkbox(label="Use EmoTalk", value=use_emotalk)
#                     with gr.TabItem('進階設定'):
#                         level = gr.Slider(minimum=0, maximum=1, step=1, label="EmoTalk Level", value=1)
#                         person = gr.Slider(minimum=0, maximum=23, step=1, label="EmoTalk Person", value=3)
#                         bs52_level = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, label="BS52 Level", value=2.0)
#                         blink_mode = gr.Radio(['none', 'period'], value='none', label='blink mode')
#                         lle_percent = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="lle_percent", value=1.0)
#                         temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="temperature", value=0.0)
#                         mouth_amp = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="mouth amplitude", value=0.4)
#                         raymarching_end_threshold = gr.Slider(minimum=0.0, maximum=0.1, step=0.0025, label="ray marching end-threshold", value=0.01)
#                     with gr.TabItem('模型路徑'):
#                         audio2secc_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=audio2secc_dir, file_count='single', label='audio2secc model')
#                         torso_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=torso_model_dir, file_count='single', label='torso model')
#                         blend_path = gr.FileExplorer(glob="emotalk/*.blend", value="emotalk/render_testing_92.blend", file_count='single', label='blend file for emotalk')
#                         postnet_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=postnet_dir, file_count='single', label='(optional) pose net model')
#                         head_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=head_model_dir, file_count='single', label='(optional) head model')
        
#         stream_key_trigger = gr.Textbox(label="Stream Key Trigger", visible=False)

#         all_inputs = [
#             drv_audio_name, blink_mode, temperature, lle_percent, mouth_amp,
#             raymarching_end_threshold, gr.State(False), audio2secc_dir, postnet_dir,
#             head_model_dir, torso_model_dir, gr.State(False), use_emotalk,
#             debug, blend_path, level, person, gr.State(False), bs52_level, bs_lm_area
#         ]
        
#         # [修改] 步驟 1: 點擊按鈕，只做一件事：生成 stream_key 並返回
#         def generate_key():
#             return str(uuid.uuid4())

#         submit.click(
#             fn=generate_key,
#             inputs=[],
#             outputs=[stream_key_trigger]
#         )

#         # [修改] 步驟 2: 當 stream_key_trigger 的值改變時，觸發兩件事：
#         # 1. 透過 _js 執行前端播放器設定
#         # 2. 透過 fn 執行後端真正的推流任務
#         stream_key_trigger.change(
#             fn=infer_obj.infer_once_args,
#             inputs=all_inputs + [stream_key_trigger], # 將所有 UI 輸入和 trigger 一起傳入
#             outputs=[info_box],
#             # [關鍵] _js 參數用在 .change() 事件上，這是相容的
#             # `x` 就是 trigger 的新值 (stream_key)
#             _js="(x) => { window.setupHlsPlayer(x); }"
#         )

#     print(sep_line, "\nGradio page is constructed.\n", sep_line)
#     return genefacepp_interface

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--a2m_ckpt", type=str, default='checkpoints/audio2motion_vae/model_ckpt_steps_400000.ckpt')
#     parser.add_argument("--postnet_ckpt", type=str, default='')
#     parser.add_argument("--head_ckpt", type=str, default='')
#     parser.add_argument("--torso_ckpt", type=str, default='checkpoints/motion2video_nerf/may_torso/model_ckpt_steps_250000.ckpt') 
#     parser.add_argument("--port", type=int, default=None) 
#     parser.add_argument("--server", type=str, default='127.0.0.1') 
#     parser.add_argument("--use_emotalk", default=True, action='store_true')
#     parser.add_argument("--blend_path", type=str, default='emotalk/render_testing_92.blend')
#     args = parser.parse_args()
#     demo = genefacepp_demo(
#         audio2secc_dir=args.a2m_ckpt,
#         postnet_dir=args.postnet_ckpt,
#         head_model_dir=args.head_ckpt,
#         torso_model_dir=args.torso_ckpt,
#         use_emotalk=args.use_emotalk,
#         device='cuda:0',
#         warpfn=None,
#     )
#     demo.queue()
#     demo.launch(server_name=