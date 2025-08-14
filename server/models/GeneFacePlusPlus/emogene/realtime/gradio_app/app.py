import gradio as gr
import cv2
import numpy as np
import tempfile
import ffmpeg
import time
import os

def generate_video():
    """
    這個 generator 每秒產生 1 段短影片（含聲音），並即時送到前端播放
    """
    for i in range(5):  # 產生 5 段影片
        # 建立一個彩色影格
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame batch {i+1}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # 儲存單張圖片成暫存檔
        img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        cv2.imwrite(img_path, frame)

        # 產生對應的聲音檔（440Hz 正弦波）
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sr = 44100
        duration = 1
        t = np.linspace(0, duration, int(sr*duration), False)
        tone = (np.sin(2*np.pi*440*t) * 0.5).astype(np.float32)
        import soundfile as sf
        sf.write(audio_path, tone, sr)

        # 使用 ffmpeg 合成 1 秒影片 + 聲音
        video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        (
            ffmpeg
            .input(img_path, loop=1, t=1, framerate=25)
            .input(audio_path)
            .output(video_path, vcodec="libx264", pix_fmt="yuv420p", acodec="aac", shortest=None)
            .overwrite_output()
            .run(quiet=True)
        )

        # 清理圖片和音檔
        os.remove(img_path)
        os.remove(audio_path)

        # 送出影片給前端（這裡是即時串流）
        yield video_path

        time.sleep(0.1)  # 模擬生成延遲

with gr.Blocks() as demo:
    gr.Video(label="即時影片串流測試", streaming=True, autoplay=True).stream(generate_video)

if __name__ == "__main__":
    demo.launch()
