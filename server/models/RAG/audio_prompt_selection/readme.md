## Run Audio Prompt Selection Server

```bash
# run server
conda activate roleplay
cd server/models/RAG/audio_prompt_selection
python server.py

# run testing client
python test_client.py retrieve -q "安樂死" -k 3

```

## Run emotion classification script
```bash
conda activate funasr
cd server/models/RAG/audio_prompt_selection
python emotion_classification.py


```