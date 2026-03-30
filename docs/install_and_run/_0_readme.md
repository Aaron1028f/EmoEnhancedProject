# How to install this project

## Introduction

Follow the instructions in the document to install the project. Each part needs independent installation and environment setup, and you can choose to install the part you need.

1. `docs/install/_1_emogene.md`
2. `docs/install/_2_roleplay.md`
3. `docs/install/_3_emotional_tts.md`
4. `docs/install/_4_frontend.md`

## Code Structure

- `server/models` : All the models are in this directory.
    - `server/models/GeneFacePlusPlus`: Talking head generation
    - `server/models/LLM`: LLM roleplay with RAG
    - `server/models/TTS/index-tts`: TTS model
    - `server/models/RAG/audio_prompt_selection`: RAG for audio prompt selection

- `server/lk_exp` : Livekit for frontend and model communication, this is needed when running chatbot web app.


> Files in the repo but not mentioned above are just for experimental purposes.

## Run the chatbot
Go check `docs/install/run_chatbot.md` for the instruction to run the servers and the client for the chatbot demo.



