## Run the Servers

For each server, open a new terminal and run the following commands:

### Step1: run livekit (cloud)

> Before running this server: 
> - check `server/lk_exp/agent-starter-python/src/agent_feng_vid.py` to set the VOLUME of the TTS (line 1).

```bash
conda activate livekit
cd server/lk_exp/agent-starter-python/

uv run python src/agent_feng_vid.py dev
```
---


### Step2: run LLM and RAG server for roleplay

> Before running this server:
> 1. Check `server/models/LLM/src/roleplay_api_for_lk.py` and `server/models/LLM/src/roleplay_main.py` to set the parameters for RAG and LLM.
> 2. Check `server/models/LLM/src/.env` to set your OPENAI_KEY and other environment variables.

```bash
conda activate roleplay
cd server/models/LLM/src/

python roleplay_api_for_lk.py
```
---

### Step3: run TTS server (IndexTTS2)
> Before running this server:
> 1. Check `server/models/TTS/index-tts/api_indextts.py` to set whether to generate talking head video or not. (line 1)

```bash
cd server/models/TTS/index-tts
conda activate indextts

CUDA_VISIBLE_DEVICES=1 uv run indextts/api_indextts.py --cuda_kernel --port 40000
```
---

### Step4: run RAG server for audio prompt selection
```bash
conda activate roleplay
cd server/models/RAG/audio_prompt_selection
python server.py
```

### Step5: run EmoGene server
```bash
conda activate geneface_py310
cd server/models/GeneFacePlusPlus/
python emogene/realtime/emogene_lk_server3.py 
# python emogene/realtime/emogene_lk_server.py (with no placeholder image)
# remember to modify the ROOM_NAME in emogene_lk_server3.py

```
> Before running this server:
> 1. Make sure to create `server/models/GeneFacePlusPlus/emogene/realtime/.env.local`, the details are in the end of this file.
> 2. Set `ROOM_NAME` and `NUM_INFERER` in `server/models/GeneFacePlusPlus/emogene/realtime/emogene_lk_server3.py`.

## Run Client (Web App)

Register Livekit and create an online playground room, when you enter the room, hard code the room name in `server/models/GeneFacePlusPlus/emogene/realtime/emogene_lk_server3.py` make sure you can connect to the livekit playground room successfully. 


[**LiveKit PlayGround**](https://agents-playground.livekit.io/#cam=1&mic=1&screen=1&video=1&audio=1&chat=1&theme_color=cyan)



## Set up the `.env.local` file for the EmoGene server

Make sure to create the `.env.local` file in `server/models/GeneFacePlusPlus/emogene/realtime/` with the following content:

```
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
OPENAI_API_KEY=
```