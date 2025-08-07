## Rreprodece the `roleplay` conda environment

### `roleplay` environment
```bash
conda create -n roleplay python=3.13

# for langchain (basic roleplay program)
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv
pip install langchain-openai

# for tools
### yt2script
pip install yt-dlp youtube-transcript-api pyannote.audio
conda install cmake # for sentencepiece error
pip install youtube-transcript-api==1.1.1 # for error of not able to fetch transcript


```
### others
```bash
pip install fastapi
pip install uvicorn

pip install gradio
# pip install websocket
pip install websocket-client
```

### RUN
```bash
# run fastapi server
# --reload 參數會在您修改程式碼後自動重啟伺服器，方便開發
uvicorn main:app --reload

# run simple client example
cd /home/aaron/project/server/models/LLM/fastapi-roleplay-project
python app/client_testing/sample_gradio_client.py
```