# Install LLM Roleplay with RAG


## 1. Install Conda Environment
```bash
conda create -n roleplay python=3.13

# for langchain (basic roleplay program)
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv
pip install langchain-openai

# other 
pip install fastapi
pip install uvicorn
pip install gradio
pip install websocket-client

```

## 2. Code for LLM Roleplay with RAG

### 2.1 Copy Code
Copy all the content in `server/models/LLM`

### 2.2 Set .env file
Create `server/models/LLM/src/.env`, and set your API KEYs (Must have OPENAI API KEY, since the embedding model in the code only supports OpenAI)
```.env
# content of .env file, remember to set your own API keys
GOOGLE_API_KEY=
OPENAI_API_KEY=
```
### 2.3 Check all the important files

#### DATA: server/models/LLM/data
- `data/prompt_for_sysprompt.txt`: Prompt for generating system prompt of the customized character.



#### CODE: server/models/LLM/src
- `src/roleplay_emo.py`: Demo code runs in terminal.

- `src/roleplay_api_for_lk.py`: the FastAPI server code for LLM roleplay, will be used in the chatbot (called by livekit).

- `src/sysprompt_ep1-3.txt`: System prompts for your customized character. You can modify this to create your own character.




### 2.4 Run LLM roleplay Demo

```bash
# simple demo in terminal
conda activate roleplay
cd server/models/LLM/src/

# Run the demo, make sure to set .env file with your API keys
# Also, you need to check this code and set parameters for your own needs.
python roleplay_emo.py 

```

## 3. Code for audio prompt selection RAG server 
### 3.1 Copy Code
Copy all the content in `server/models/RAG/audio_prompt_selection`

### 3.2 Test simple demo for RAG server
See `server/models/RAG/audio_prompt_selection/readme.md`
