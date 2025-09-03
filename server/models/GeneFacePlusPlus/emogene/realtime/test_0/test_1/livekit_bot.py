# pip install livekit-api
# pip install livekit

import asyncio
import os
import numpy as np
from livekit import rtc, api
from server.models.GeneFacePlusPlus.emogene.realtime.emogene_stream import GeneFace2Infer # 從你現有的檔案中匯入
import logging

import cv2  # 用於影像處理

# --- LiveKit 設定 ---
# 從你的 LiveKit 專案儀表板取得
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "your_api_key")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "your_api_secret")

# --- 機器人與房間設定 ---
ROOM_NAME = "geneface-stream"
BOT_IDENTITY = "geneface-bot"

# --- GeneFace 模型設定 ---
# 根據你的設定填寫
A2M_CKPT = 'checkpoints/audio2motion_vae'
TORSO_CKPT = 'checkpoints/motion2video_nerf/may_torso'
POSTNET_CKPT = 'checkpoints/motion2video_nerf/may_postnet'
HEAD_CKPT = 'checkpoints/motion2video_nerf/may_head'
USE_EMOTALK = True

class GeneFaceBot:
    def __init__(self, infer_instance: GeneFace2Infer):
        self.infer_instance = infer_instance
        self.room = rtc.Room()
        self.video_source = rtc.VideoSource(512, 512)
        self.video_track = rtc.LocalVideoTrack.create_video_track("gene_face_video", self.video_source)

    async def start(self):
        logging.info(f"正在連線到 LiveKit 房間: {ROOM_NAME}...")
        
        # 產生 token 以加入房間
        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(BOT_IDENTITY)
            .with_name("GeneFace Bot")
            .with_grant(api.VideoGrant(room_join=True, room=ROOM_NAME, can_publish=True, can_subscribe=False))
            .to_jwt()
        )

        try:
            await self.room.connect(LIVEKIT_URL, token)
            logging.info("成功連線到房間")

            # 發佈視訊軌道
            await self.room.local_participant.publish_track(self.video_track)
            logging.info("視訊軌道已發佈")

            # 在這裡你可以等待一個觸發事件，例如來自前端的 Data Channel 訊息
            # 為了簡化，我們直接開始串流
            await self.run_inference_and_stream()

        except Exception as e:
            logging.error(f"連線或發佈時發生錯誤: {e}")
        finally:
            await self.room.disconnect()
            logging.info("已與房間中斷連線")

    async def run_inference_and_stream(self):
        """
        執行推論並將產生的幀串流到 LiveKit
        """
        logging.info("正在準備推論...")

        # --- 準備推論所需的輸入 (inp) ---
        # 這部分需要根據你的實際需求來設定，例如從前端接收音訊檔案
        # 這裡使用一個範例音訊檔案
        drv_audio_name = 'data/raw/val_wavs/zozo.wav'
        
        inp = {
            'a2m_ckpt': A2M_CKPT, 'postnet_ckpt': POSTNET_CKPT, 'head_ckpt': HEAD_CKPT,
            'torso_ckpt': TORSO_CKPT, 'drv_audio_name': drv_audio_name, 'drv_pose': 'static',
            'blink_mode': 'period', 'temperature': 0.2, 'mouth_amp': 0.4,
            'lle_percent': 0.2, 'debug': False, 'raymarching_end_threshold': 0.01,
            'low_memory_usage': False, 'use_emotalk': USE_EMOTALK,
            "blend_path": "emotalk/render_testing_92.blend", "level": 1, "person": 1,
            "output_video": False, "bs52_level": 3, "bs_lm_area": 1,
        }

        # 準備 batch 資料
        samples = self.infer_instance.prepare_batch_from_inp(inp)
        logging.info("推論準備完成，開始串流幀...")

        # 使用 stream_forward_system 生成器
        frame_generator = self.infer_instance.stream_forward_system(samples, inp)

        loop = asyncio.get_event_loop()
        
        # 逐一處理生成器產生的幀
        for frame_np in frame_generator:
            # frame_np 是 (H, W, C) 的 NumPy 陣列，顏色格式為 RGB
            
            # 將 NumPy 陣列轉換為 LiveKit 的 VideoFrame
            # 注意：VideoFrame 需要 BGR 格式，所以我們需要轉換顏色通道
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_frame = rtc.VideoFrame.from_ndarray(frame_bgr, format='bgr24')

            # 將幀推送到視訊源
            await self.video_source.capture_frame(video_frame)
            
            # 給予事件循環一點時間處理其他任務，避免阻塞
            await asyncio.sleep(1 / 25) # 假設影片幀率為 25 fps

        logging.info("所有幀都已串流完畢。")


async def main():
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.StreamHandler()])

    logging.info("正在初始化 GeneFace2Infer 模型...")
    # 載入模型實例
    infer_instance = GeneFace2Infer(
        audio2secc_dir=A2M_CKPT,
        postnet_dir=POSTNET_CKPT,
        head_model_dir=HEAD_CKPT,
        torso_model_dir=TORSO_CKPT,
        use_emotalk=USE_EMOTALK
    )
    logging.info("模型初始化完成。")

    bot = GeneFaceBot(infer_instance)
    await bot.start()


if __name__ == "__main__":
    # 在 Windows 上，可能會需要設定不同的事件循環策略
    # if sys.platform == "win32":
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程式已手動中斷")

# ```

# ---

# ### 3. 如何執行與運作

# 1.  **設定環境變數**：
#     在你的終端機中，設定 LiveKit 的連線資訊。
#     ```bash
#     export LIVEKIT_URL="ws://your-livekit-url"
#     export LIVEKIT_API_KEY="your_api_key"
#     export LIVEKIT_API_SECRET="your_api_secret"
#     ```
#     或者，你也可以直接在 `livekit_bot.py` 檔案中修改這些變數的值。

# 2.  **執行機器人**：
#     ```bash
#     python /home/aaron/project/server/models/GeneFacePlusPlus/emogene/realtime/livekit_bot.py
#     ```

# 3.  **前端接收**：
#     *   在你的前端應用程式中，使用 LiveKit Client SDK 連線到同一個房間 (`geneface-stream`)。
#     *   監聽 `TrackSubscribed` 事件。
#     *   當你訂閱到一個新的遠端視訊軌道時，檢查其參與者身份 (`participant.identity`) 是否為 `geneface-bot`。
#     *   如果是，就將這個軌道附加到 HTML 的 `<video>` 元素上，你就可以看到由 Python 伺服器即時生成和串流的頭像了。

# ### 程式碼重點說明：

# *   **`GeneFaceBot` 類別**：封裝了所有與 LiveKit 互動的邏輯。
# *   **`__init__`**：初始化時，除了 `rtc.Room`，最重要的是建立一個 `rtc.VideoSource` 和一個 `rtc.LocalVideoTrack`。`VideoSource` 是我們推送原始視訊幀的地方，而 `LocalVideoTrack` 則是將這個源發佈到房間的載體。
# *   **`start`**：處理連線、取得權杖 (token)、加入房間和發佈軌道的標準流程。
# *   **`run_inference_and_stream`**：
#     *   這是核心的串流邏輯。
#     *   它呼叫 `self.infer_instance.stream_forward_system()` 來取得一個幀生成器。
#     *   在 `for` 迴圈中，它從生成器取出每一幀 (`frame_np`)。
#     *   **關鍵轉換**：`rtc.VideoFrame.from_ndarray()` 是將 NumPy 陣列轉換為 LiveKit 內部格式的函式。**請注意**，OpenCV 和 WebRTC 通常偏好 BGR 格式，而你的模型輸出是 RGB，所以我們使用 `cv2.cvtColor` 進行了轉換。
#     *   `self.video_source.capture_frame(video_frame)` 是將這一幀推送出去的動作。
#     *   `asyncio.sleep(1 / 25)` 模擬了固定的幀率，並防止 CPU 佔用過高。

# 這個架構讓你可以將任何 Python 生成的圖像序列，透過高效能的 WebRTC 協議，低延遲地串流到瀏覽器中。# filepath: /home/aaron/project/server/models/GeneFacePlusPlus/emogene/realtime/livekit_bot.py
