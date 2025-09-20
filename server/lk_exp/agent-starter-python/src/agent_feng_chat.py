# （保留你原本的 imports）
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

from livekit.agents.job import get_job_context  # 新增


# local LLM services
from localLLM import LocalLLM

# local TTS services
from localTTS_GPTSoVITS import LocalTTS
# from localTTS_indextts import LocalTTS

logger = logging.getLogger("agent")
load_dotenv(".env.local")


def filter_for_display(full_text: str) -> str:
    """
    範例過濾器：把方括號內容移除（例如 [internal note]）並移除一些示範式敏感 token。
    你可以改成任何邏輯：正則 / 黑名單 / summary / redact 等。
    """
    import re

    # remove bracketed annotations like [xxx]
    text = re.sub(r"\[.*?\]", "", full_text)
    # example: redact token "SECRET" (示範用)
    text = text.replace("SECRET", "[redacted]")
    # trim repeated whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # remove: ., !?
    text = re.sub(r"[.,!?]+", "", text)
    
    return text

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # 可選：安全取得 room 的小工具（先用 JobContext，退而求其次用 _room_io）
    def _get_room(self):
        try:
            return get_job_context().room
        except Exception:
            # 非公開 API，僅作為最後退路
            room_io = getattr(self.session, "_room_io", None)
            return getattr(room_io, "room", None) if room_io else None

    async def llm_node(self, chat_ctx, tools, model_settings=None):
        """
        - 以官方預設 llm_node 串流 chunk
        - 顯示給前端：透過 room 的 text stream/topic=lk.chat
        - 送給 TTS：原樣 yield chunk
        """
        writer = None
        room = self._get_room()  # 取得目前的 Room 實例
        try:
            if room is not None:
                try:
                    writer = await room.local_participant.stream_text(topic="lk.chat")
                except Exception:
                    logger.debug("開啟 text stream 失敗，將改用 send_text fallback")
                    writer = None
            else:
                logger.debug("找不到 Room 實例，略過文字串流")

            accumulated = ""
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                # 擷取 chunk 的文字
                text_piece = ""
                if isinstance(chunk, str):
                    text_piece = chunk
                else:
                    delta = getattr(chunk, "delta", None)
                    text_piece = (
                        getattr(delta, "content", None)
                        or getattr(delta, "text", "")
                        or getattr(chunk, "text", "")
                        or ""
                    )

                if text_piece:
                    accumulated += text_piece
                    display_text = filter_for_display(accumulated)

                    # 優先用 writer（streaming），否則用 send_text 單發
                    if writer:
                        try:
                            await writer.write(display_text)
                        except Exception:
                            logger.exception("writer.write 失敗，改用 send_text")
                            if room is not None:
                                try:
                                    await room.local_participant.send_text(display_text, topic="lk.chat")
                                except Exception:
                                    logger.exception("send_text 仍失敗")
                    else:
                        if room is not None:
                            try:
                                await room.local_participant.send_text(display_text, topic="lk.chat")
                            except Exception:
                                logger.exception("send_text 失敗")

                # 保持原樣給 TTS
                yield chunk
        finally:
            if writer:
                try:
                    await writer.aclose()
                except Exception:
                    logger.exception("關閉 text writer 失敗")


# 其餘程式（entrypoint, prewarm 等）保持不變 —— 你原本的程式碼繼續使用
# (把下面你原本的 entrypoint / prewarm 直接貼回，或保留現有)

# 例如：（你的 prewarm / entrypoint 維持原樣）
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=LocalLLM(),
        stt=openai.STT(model='gpt-4o-transcribe'),
        # tts=LocalTTS(),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
