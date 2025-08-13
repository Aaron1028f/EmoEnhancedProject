# source: https://stackoverflow.com/questions/75740652/fastapi-streamingresponse-not-streaming-with-generator-function

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import roleplay_emo
from roleplay_main import prepare_roleplay_chain

from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

class AppState:
    def __init__(self):
        self.roleplay_chain = prepare_roleplay_chain()
        self.chat_history = []

app_state = AppState()

async def roleplay_streamer(user_input: str):
    full_response = ""
    stream_input = {"input": user_input, "chat_history": app_state.chat_history}
    
    if not app_state.roleplay_chain:
        yield "Error: Roleplay chain not initialized."
        return

    for chunk in app_state.roleplay_chain.stream(stream_input):
        full_response += chunk
        yield chunk
        # yield f"data: {chunk}\n\n"

    # yield f"data: {full_response}\n\n"
    
    # manage the chat history
    app_state.chat_history.append(HumanMessage(content=user_input))
    app_state.chat_history.append(AIMessage(content=full_response))
    if len(app_state.chat_history) > 10:
        app_state.chat_history = app_state.chat_history[-10:]


@app.get("/streaming_response")
async def get_streaming_response(user_input: str):
    return StreamingResponse(roleplay_streamer(user_input), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)