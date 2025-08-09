import gradio as gr
import requests
import websockets
import asyncio
import json
import uuid
from typing import List, Tuple, AsyncGenerator

# --- 設定 ---
FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
REST_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/chat/"
WEBSOCKET_URL = f"ws://{FASTAPI_HOST}:{FASTAPI_PORT}/ws/chat/"

# --- 主題與樣式 ---
theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#f0f4f8",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
    border_color_primary="#e0e7ff",
    block_title_text_weight="600"
)

custom_css = """
#chatbot {
    height: 600px;
    overflow: auto;
    border-radius: 12px;
}
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
    padding-top: 2rem !important;
}
footer { display: none !important; }
"""

# --- API 互動邏輯 ---

async def rest_chat_predict(message: str, chat_history: List[Tuple[str, str]], session_id: str) -> str:
    """
    使用 REST API 進行聊天。
    """
    if not session_id:
        return "錯誤：請先輸入一個 Session ID。"

    payload = {
        "user_input": message,
        "session_id": session_id
    }
    try:
        response = requests.post(REST_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        print(f"REST API 回應: {data}")  # 增加日誌以便觀察
        return data.get("response", "從伺服器收到了無效的回應。")
    except requests.exceptions.RequestException as e:
        print(f"REST API 請求錯誤: {e}")
        return f"連線到伺服器時發生錯誤：{e}"
    except json.JSONDecodeError:
        return "無法解析伺服器的回應，請檢查伺服器日誌。"

async def websocket_chat_predict(message: str, chat_history: List[Tuple[str, str]], session_id: str) -> AsyncGenerator[str, None]:
    """
    使用 WebSocket 進行串流聊天。
    """
    if not session_id:
        yield "錯誤：請先輸入一個 Session ID。"
        return

    full_response = ""
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            await websocket.send(json.dumps({"session_id": session_id}))
            await websocket.recv() # 等待伺服器確認連線
            await websocket.send(json.dumps({"user_input": message}))

            while True:
                data_str = await websocket.recv()
                data = json.loads(data_str)
                
                if data.get("type") == "stream":
                    chunk = data.get("data", "")
                    full_response += chunk
                    yield full_response
                elif data.get("type") == "stream_end":
                    break
                elif data.get("type") == "error":
                    error_message = data.get("data", "未知的 WebSocket 錯誤")
                    print(f"WebSocket 錯誤: {error_message}")
                    full_response += f"\n\n[伺服器錯誤]: {error_message}"
                    yield full_response
                    break

    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        error_msg = f"WebSocket 連線失敗: {e}. 請確認 FastAPI 伺服器是否已在 {FASTAPI_HOST}:{FASTAPI_PORT} 上運行。"
        print(error_msg)
        yield error_msg
    except Exception as e:
        error_msg = f"發生未預期的錯誤: {e}"
        print(error_msg)
        yield error_msg

# --- Gradio 介面 ---
with gr.Blocks(theme=theme, css=custom_css, title="RAG 聊天機器人測試介面") as demo:
    gr.Markdown(
        """
        # 🎭 RAG 聊天機器人測試介面
        使用此介面來測試您的 FastAPI 聊天伺服器。
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            session_id = gr.Textbox(
                label="對話 Session ID",
                info="為每個獨立的對話指定一個唯一的 ID，以維持對話歷史。",
                value=lambda: str(uuid.uuid4()),
                interactive=True
            )
        with gr.Column(scale=2):
             mode = gr.Radio(
                ["WebSocket (串流)", "REST (標準)"],
                label="連線模式",
                value="WebSocket (串流)",
                interactive=True
            )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, "https://img.icons8.com/plasticine/100/bot.png"),
        height=500
    )

    chat_input = gr.Textbox(
        show_label=False,
        placeholder="在這裡輸入您的訊息...",
        container=False,
        scale=7,
    )

    # 【已修正】聊天邏輯處理，採用官方推薦的 gr.update() 模式
    async def handle_chat_submission(message, chat_history, session_id_val, mode_val):
        if not message:
            return

        # 步驟 1: 立即顯示使用者訊息和一個等待中的位置
        chat_history.append((message, None))
        yield gr.update(value=""), gr.update(value=chat_history.copy())

        # 步驟 2: 呼叫後端並串流或獲取回應
        if mode_val == "WebSocket (串流)":
            generator = websocket_chat_predict(message, chat_history, session_id_val)
            async for partial_response in generator:
                # 持續更新機器人回應的位置
                chat_history[-1] = (message, partial_response)
                # 【修正】每次都 yield 一個新的列表副本，並使用 gr.update() 強制 Gradio 更新 UI
                yield gr.update(value=""), gr.update(value=chat_history.copy())
        else:
            # 標準模式一次性更新
            response = await rest_chat_predict(message, chat_history, session_id_val)
            chat_history[-1] = (message, response)
            # 【修正】這裡也使用 gr.update() 來確保更新
            yield gr.update(value=""), gr.update(value=chat_history.copy())

    # 將元件連接到處理函式
    chat_msg = chat_input.submit(
        handle_chat_submission,
        [chat_input, chatbot, session_id, mode],
        [chat_input, chatbot],
        queue=True
    )
    # 當串流結束後，重新啟用輸入框
    chat_msg.then(lambda: gr.update(interactive=True), None, [chat_input], queue=False)

    # 清除按鈕
    with gr.Row():
        clear_btn = gr.ClearButton([chat_input, chatbot], value="🗑️ 清除對話")
        new_session_btn = gr.Button("🔄️ 開啟新對話 (New Session)")

    def new_session():
        return [], str(uuid.uuid4())

    new_session_btn.click(
        new_session,
        inputs=[],
        outputs=[chatbot, session_id],
        queue=False
    )

if __name__ == "__main__":
    print("Gradio 介面啟動中...")
    print(f"請在瀏覽器中開啟 http://127.0.0.1:7860")
    demo.queue().launch(server_name="0.0.0.0")
