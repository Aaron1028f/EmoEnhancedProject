# source: https://stackoverflow.com/questions/75740652/fastapi-streamingresponse-not-streaming-with-generator-function

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def fake_data_streamer():
    for i in range(10):
        yield b'some fake data'
        await asyncio.sleep(0.1)


@app.get("/streaming_response")
async def get_streaming_response():
    return StreamingResponse(fake_data_streamer(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
