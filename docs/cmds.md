### 目前可以執行的模型

#### server and client(temp), now only support text QA
- rag_service
```bash
# 執行環境:
# conda: geneface
# directory: server/services
cd server/services
uvicorn rag_service:app --port 8001

```

- llm_service
```bash
# 執行環境:
# conda: geneface
# directory: server/services
cd server/services
uvicorn llm_service:app --port 8002
```
- orchestrator
```bash
# 執行環境:
# conda: geneface
# directory: server/services
cd server/services
uvicorn orchestrator:app --port 8000
```

client_temp
```bash
# 執行環境:
# conda: geneface
# directory: server/services
cd server/services
python client_temp.py
```

#### TTS model
- cosyvoice

```bash
# 執行環境:
# conda: cosyvoice
# directory: server/models/tts_model/CosyVoice
python webui.py

# server
cd server/models/tts_model/CosyVoice
python runtime/python/modified_fastapi/server.py
```

- cosyvoice2 (use this for demo of the ability of cosyvoice 2)
```bash
# 執行環境:
# conda: cosyvoice
# directory: server/models/tts_model/CosyVoice2-0.5B
python app.py
```



#### talking head model
- geneface++ 

```bash
# 執行環境:
# conda: geneface
# directory: server/models/GeneFacePlusPlus
python emogene/app_emogene.py # just gradio app

python emogene/api_emogene.py # providing basic api and also available for using as gradio app

```

