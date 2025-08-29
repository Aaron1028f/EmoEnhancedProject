### RUN THE SERVER
```bash

# run ASR and VAD server
conda activate api4sensevoice
cd server/models/ASR/api4sensevoice/
python server_wss.py

# run LLM and RAG server
conda activate roleplay
cd server/models/LLM/src/
python roleplay_api.py

# run TTS server (GPT-SoVits)
conda activate GPTSoVits
cd server/models/TTS/GPT-SoVITS/
python api_v2.py
# python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

```

### RUN CLIENT
```bash
# go server/models/ASR/api4sensevoice/client_wss.html
# just use run live server using vscode plug-in


```