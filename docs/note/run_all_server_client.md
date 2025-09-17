## RUN THE SERVERs
```bash
# =====================================================================
# =====================================================================
# run livekit (cloud), (if run local, modify the .env.local setting)
conda activate livekit
cd server/lk_exp/agent-starter-python/
uv run python src/agent_feng_vid.py dev
# =====================================================================
# =====================================================================
# run LLM and RAG server
conda activate roleplay
cd server/models/LLM/src/
# python roleplay_api.py
python roleplay_api_for_lk.py
# =====================================================================
# =====================================================================
# run TTS server (GPT-SoVits), faster but lower quality
conda activate GPTSoVits
cd server/models/TTS/GPT-SoVITS/
python api_v2_lk_save_wav.py
# python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
# `notify_emogene_fire_and_forget()` call EmoGene server, comment out if only audio chat is needed
# ---------------------------------------------------------------------
# run TTS server (index-tts2), slower but higher quality (also with emotion control support)
cd server/models/TTS/index-tts
conda activate indextts
CUDA_VISIBLE_DEVICES=0 uv run indextts/api_indextts.py --cuda_kernel --port 40000 --deepspeed
# =====================================================================
# =====================================================================
# run EmoGene server
conda activate geneface_py310
cd server/models/GeneFacePlusPlus/
python emogene/realtime/emogene_lk_server.py
# =====================================================================
# =====================================================================
```

## RUN CLIENT

[livekit playground](https://agents-playground.livekit.io/#cam=1&mic=1&screen=1&video=1&audio=1&chat=1&theme_color=cyan)

Option1: **just CLICK THE LINK TO OPEN TEMP WEBUI, and choose cloud**

Option2: run the following command to start the local server
```bash
# RUN THE LOCAL LIVEKIT SERVER
cd server/lk_exp/server_src_bin/
# /home/aaron/project/server/lk_exp/server_src_bin/livekit-server --dev
/home/aaron/project/server/lk_exp/server_src_bin/livekit-server --dev --bind 0.0.0.0

# GENERATE access token (url, access token), WHICH ARE NEEDED WHEN JOINING A ROOM
./lk token create \
  --api-key devkey --api-secret secret \
  --join --room test_room --identity test_user \
  --valid-for 24h

### IMPORTANT NOTE ###
# when joining the room locally, the URL in the web to join the room should be:
# LIVEKIT_URL=wss://localhost:7880

```


## not used
```bash
# # run ASR and VAD server
# conda activate api4sensevoice
# cd server/models/ASR/api4sensevoice/
# python server_wss.py

# go server/models/ASR/api4sensevoice/client_wss.html
# just use run live server using vscode plug-in

```