# run with text and audio output

import gradio as gr
import requests
import time

def chat_with_audio(user_input: str, history):
    url = "http://localhost:8000/generate"
    response = requests.post(url, json={"text": user_input})
    response.raise_for_status()
    r = response.json()
    return r["answer"], r.get("audio_url", None)

def user(user_message, history):
    return "", history + [{"role": "user", "content": user_message}]

def bot(history):
    # 呼叫後端拿到完整回覆跟音檔 URL
    bot_message, audio_url = chat_with_audio(history[-1]["content"], history)
    # 在聊天紀錄裡先佔個位置
    history.append({"role": "assistant", "content": ""})

    # **第一步**：先把音檔 URL 傳給 audio_output
    # 這裡 chat 視窗內容還是空，audio element 會自動開始播放
    yield history, audio_url

    # **第二步**：打字機效果逐字更新 chat 窗口
    for char in bot_message:
        history[-1]["content"] += char
        time.sleep(0.05)         # 模擬打字延遲
        # audio_output 這一欄用 None，保持原來的播放狀態
        yield history, None

with gr.Blocks() as demo:
    chatbot      = gr.Chatbot(type="messages")
    audio_output = gr.Audio(label="語音輸出", autoplay=True, type="filepath", streaming=True)
    msg          = gr.Textbox()
    clear        = gr.Button("Clear")

    msg.submit(
        user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot,
        inputs=[chatbot],
        outputs=[chatbot, audio_output]
    )

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()




# # =================================================================================================
# # run with text and audio and video output

# import gradio as gr
# import requests
# import time

# def chat_with_audio(user_input: str, history):
#     url = "http://localhost:8000/generate"
#     response = requests.post(url, json={"text": user_input})
#     response.raise_for_status()
#     r = response.json()
#     return r["answer"], r.get("audio_url", None), r.get("video_url", None)

# def user(user_message, history):
#     return "", history + [{"role": "user", "content": user_message}]

# def bot(history):
#     # 呼叫後端拿到完整回覆跟音檔 URL
#     bot_message, audio_url, video_url = chat_with_audio(history[-1]["content"], history)
#     print('-'*50)
#     print('bot_message:', bot_message)
#     print('audio_url:', audio_url)
#     print('video_url:', video_url)
#     # 在聊天紀錄裡先佔個位置
#     history.append({"role": "assistant", "content": ""})

#     # **第一步**：先把音檔 URL 傳給 audio_output
#     # 這裡 chat 視窗內容還是空，audio element 會自動開始播放
#     yield history, audio_url, video_url

#     # **第二步**：打字機效果逐字更新 chat 窗口
#     for char in bot_message:
#         history[-1]["content"] += char
#         time.sleep(0.05)         # 模擬打字延遲
#         # audio_output 這一欄用 None，保持原來的播放狀態
#         yield history, None, video_url

# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column():
#             video_output = gr.Video(label="影片輸出", autoplay=True, format="mp4")
            
#         with gr.Column():
#             chatbot      = gr.Chatbot(type="messages")
#             audio_output = gr.Audio(label="語音輸出", autoplay=False, type="filepath", streaming=True)
#             msg          = gr.Textbox()
#             clear        = gr.Button("Clear")

#     msg.submit(
#         user,
#         inputs=[msg, chatbot],
#         outputs=[msg, chatbot],
#         queue=False
#     ).then(
#         bot,
#         inputs=[chatbot],
#         outputs=[chatbot, audio_output, video_output]
#     )

#     clear.click(lambda: None, None, chatbot, queue=False)

# if __name__ == "__main__":
#     demo.launch()