import gradio as gr
from time import sleep

def keep_repeating(video_file):
    for _ in range(10):
        # sleep(0.5)
        yield video_file

gr.Interface(keep_repeating,
             gr.Video(sources=["webcam"], format="mp4"),
             gr.Video(streaming=True, autoplay=True)
).launch()