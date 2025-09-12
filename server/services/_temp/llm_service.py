# uvicorn llm_service:app --port 8002

# llm_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# global state
from dotenv import load_dotenv
load_dotenv()  # 讀取 .env 檔案中的環境變
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# history state and system prompt
system_prompt = """
你是一位神經內科醫師(方醫師)，喜歡用直白的語氣和生動的舉例，說明艱深的醫學知識，幫助大家更了解自己的身體健康。
相關資訊提到的問題與回答都是你自己曾經的回答，請使用這些資訊的語氣和風格來回答問題。
"""

conversation_history = [
    {"role": "system", "content": system_prompt}
]

app = FastAPI(title="LLM Service")

class LLMRequest(BaseModel):
    query: str # user query
    context: str # rag output

class LLMResponse(BaseModel):
    answer: str # LLM response

@app.post("/generate", response_model=LLMResponse)
def generate(req: LLMRequest):
    # prompt = req.context + "\nUser: " + req.query
    prompt = f"使用者的問題：{req.query}\n\n相關上下文：{req.context}\n\n請回答上述問題，並使用生動的舉例和直白的語氣在100個字內回答。"
    
    conversation_history.append({"role": "user", "content": prompt})
    
    # call OpenAI API (use gpt-4.1), just use the last 5 conversation turns
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=system_prompt,
        input=conversation_history[-10:],  # use the last 5 turns for context
        max_output_tokens=150,
    )
    
    text = response.output_text
    
    conversation_history.append({"role": "assistant", "content": text})
    
    
    print("Conversation History:", conversation_history)
    
    return LLMResponse(answer=text)
    
