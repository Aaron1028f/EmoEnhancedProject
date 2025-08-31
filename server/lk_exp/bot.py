import asyncio
from livekit import rtc

ROOM_URL = "ws://localhost:7880"  # Local server
ACCESS_TOKEN = "替換成從 /get_token 拿到的 token"

async def main():
    # 建立房間實例
    room = rtc.Room()
    await room.connect(url=ROOM_URL, token=ACCESS_TOKEN)

    @room.on("participant_connected")
    def on_participant_connected(participant):
        print(f"新參與者加入: {participant.identity}")

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"訂閱到 {participant.identity} 的 {track.kind} track")

        # 範例：播放收到的音訊 track 或送到你的 TTS/ASR pipeline
        # 這裡可以接上你的 ASR → LLM → TTS 流程
        # track.subscribe(callback=你的處理函數)

    # 保持程式運行
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
