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