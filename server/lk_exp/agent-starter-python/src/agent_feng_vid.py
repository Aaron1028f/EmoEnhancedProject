import logging

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# local services
from localLLM import LocalLLM

# from localTTS_GPTSoVITS import LocalTTS
from localTTS_indextts import LocalTTS

from livekit import rtc
import asyncio
from livekit import api
import os


logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        # llm=openai.LLM(model="gpt-4o-mini"),
        llm=LocalLLM(),
        
        # stt=openai.STT(model="whisper-1", language="zh"),
        stt=openai.STT(model='gpt-4o-transcribe'),

        # tts=openai.TTS(model='gpt-4o-mini-tts', voice="ash"),
        tts = LocalTTS(),
        
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        # preemptive_generation=False,  # for testing emo TTS
    )
    
    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        # # # # Disable audio output and transcription if not needed
        # room_output_options=RoomOutputOptions(
        #     # audio_enabled=False,          # 關閉 Agent 在房間的音訊發佈
        #     sync_transcription=False,     # 避免把文字同步綁到音訊輸出
        #     transcription_enabled=True, # 需要時可保留文字輸出        
        # ),
    )

    # Join the room and connect to the user
    await ctx.connect()

#     # 發佈後將本地音訊 track 靜音/停用（房間聽不到，但流程照跑）
#     await _mute_local_audio_track(ctx.room)

# # 新增：將本地已發佈的 audio track 設為靜音（移除 unpublish 路徑）
# async def _mute_local_audio_track(room: rtc.Room):
#     pub_audio = None
#     for _ in range(100):  # 最多等 10s
#         pubs = list(getattr(room.local_participant, "track_publications", {}).values())
#         for pub in pubs:
#             kind = getattr(pub, "kind", None) or getattr(getattr(pub, "track", None), "kind", None)
#             source = getattr(pub, "source", None) or getattr(getattr(pub, "track", None), "source", None)
#             if kind == rtc.TrackKind.KIND_AUDIO or source == rtc.TrackSource.SOURCE_MICROPHONE:
#                 pub_audio = pub
#                 break
#         if pub_audio:
#             break
#         await asyncio.sleep(0.1)

#     if not pub_audio:
#         logger.warning("No local audio publication found to mute")
#         return

#     # 1) 優先嘗試 publication 層級靜音
#     try:
#         set_muted = getattr(pub_audio, "set_muted", None)
#         if callable(set_muted):
#             await set_muted(True)
#             logger.info("Muted local audio via LocalTrackPublication.set_muted(True)")
#             return
#     except Exception as e:
#         logger.debug(f"set_muted failed: {e}")

#     # 2) 停用底層 track（仍保留 publication）
#     try:
#         track = getattr(pub_audio, "track", None)
#         if track is not None and hasattr(track, "set_enabled"):
#             track.set_enabled(False)
#             logger.info("Disabled local audio via LocalTrack.set_enabled(False)")
#             return
#     except Exception as e:
#         logger.debug(f"track.set_enabled(False) failed: {e}")

#     # 3) 伺服器端靜音（保留 publication，不建議 unpublish）
#     try:
#         url = os.getenv("LIVEKIT_URL")
#         api_key = os.getenv("LIVEKIT_API_KEY")
#         api_secret = os.getenv("LIVEKIT_API_SECRET")
#         if url and api_key and api_secret:
#             svc = api.RoomServiceClient(url=url, api_key=api_key, api_secret=api_secret)
#             sid = getattr(pub_audio, "track_sid", None) or getattr(pub_audio, "sid", None)
#             if sid is None and hasattr(pub_audio, "track"):
#                 sid = getattr(pub_audio.track, "sid", None)
#             if sid is None:
#                 raise RuntimeError("cannot resolve audio track sid for server mute")
#             await svc.mute_published_track(
#                 room.name, identity=room.local_participant.identity, track_sid=sid, muted=True
#             )
#             logger.info("Server-muted local audio via RoomServiceClient.mute_published_track")
#             return
#         else:
#             logger.warning("LIVEKIT_URL/API_KEY/API_SECRET not set; cannot server-mute")
#     except Exception as e:
#         logger.debug(f"server mute failed: {e}")

#     logger.warning("Failed to mute/disable local audio track (all fallbacks exhausted)")


# # 新增：將本地已發佈的 audio track 設為靜音（多重 fallback）
# async def _mute_local_audio_track(room: rtc.Room):
#     # 等待本地 audio publication 出現
#     pub_audio = None
#     for _ in range(100):  # 最多等 10s
#         pubs = list(getattr(room.local_participant, "track_publications", {}).values())
#         for pub in pubs:
#             # 盡量相容不同 SDK：檢查 pub.kind / pub.source / pub.track.kind
#             kind = getattr(pub, "kind", None) or getattr(getattr(pub, "track", None), "kind", None)
#             source = getattr(pub, "source", None) or getattr(getattr(pub, "track", None), "source", None)
#             if kind == rtc.TrackKind.KIND_AUDIO or source == rtc.TrackSource.SOURCE_MICROPHONE:
#                 pub_audio = pub
#                 break
#         if pub_audio:
#             break
#         await asyncio.sleep(0.1)

#     if not pub_audio:
#         logger.warning("No local audio publication found to mute")
#         return

#     # 1) 先嘗試 publication 層級靜音
#     try:
#         set_muted = getattr(pub_audio, "set_muted", None)
#         if callable(set_muted):
#             await set_muted(True)
#             logger.info("Muted local audio via LocalTrackPublication.set_muted(True)")
#             return
#     except Exception as e:
#         logger.debug(f"set_muted failed: {e}")

#     # 2) 嘗試停用底層 track（常見於部分版本）
#     try:
#         track = getattr(pub_audio, "track", None)
#         if track is not None and hasattr(track, "set_enabled"):
#             track.set_enabled(False)
#             logger.info("Disabled local audio via LocalTrack.set_enabled(False)")
#             return
#     except Exception as e:
#         logger.debug(f"track.set_enabled(False) failed: {e}")

#     # 3) 嘗試在本地端取消發佈（unpublish）
#     try:
#         # 部分 SDK 提供用 publication 物件或 track_sid 取消發佈
#         unpub = getattr(room.local_participant, "unpublish_track", None)
#         if callable(unpub):
#             # 嘗試以 track_sid 或直接傳 pub_audio
#             sid = getattr(pub_audio, "track_sid", None) or getattr(pub_audio, "sid", None)
#             if sid is not None:
#                 await unpub(sid)  # 某些版本接受 sid
#             else:
#                 await unpub(pub_audio)  # 某些版本接受 publication 物件
#             logger.info("Unpublished local audio track via LocalParticipant.unpublish_track(...)")
#             return
#     except Exception as e:
#         logger.debug(f"unpublish_track failed: {e}")

#     # 4) 最後手段：呼叫 LiveKit 伺服器端 API 將該 track 設為 muted
#     try:
#         url = os.getenv("LIVEKIT_URL")
#         api_key = os.getenv("LIVEKIT_API_KEY")
#         api_secret = os.getenv("LIVEKIT_API_SECRET")
#         if url and api_key and api_secret:
#             svc = api.RoomServiceClient(url=url, api_key=api_key, api_secret=api_secret)
#             # 找 publication sid
#             sid = getattr(pub_audio, "track_sid", None) or getattr(pub_audio, "sid", None)
#             if sid is None and hasattr(pub_audio, "track"):
#                 sid = getattr(pub_audio.track, "sid", None)
#             if sid is None:
#                 raise RuntimeError("cannot resolve audio track sid for server mute")
#             await svc.mute_published_track(
#                 room.name, identity=room.local_participant.identity, track_sid=sid, muted=True
#             )
#             logger.info("Server-muted local audio via RoomServiceClient.mute_published_track")
#             return
#         else:
#             logger.warning("LIVEKIT_URL/API_KEY/API_SECRET not set; cannot server-mute")
#     except Exception as e:
#         logger.debug(f"server mute failed: {e}")

#     logger.warning("Failed to mute/disable/unpublish local audio track (all fallbacks exhausted)")
    
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
