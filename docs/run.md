## service part
```bash
# geneface conda environment
conda activate geneface

cd server/services
uvicorn rag_service:app --port 8001

cd server/services
uvicorn llm_service:app --port 8002

cd server/services
uvicorn orchestrator:app --port 8000

# ------------------------------------------------
# cosycoice conda environment
conda activate cosyvoice

cd server/models/tts_model/CosyVoice
python runtime/python/modified_fastapi/server.py
# ------------------------------------------------

cd server/models/GeneFacePlusPlus
python emogene/api_emogene.py

# ------------------------------------------------

# run the gradio app (just for simple demo),
conda activate cosyvoice
cd server
python app.py

```