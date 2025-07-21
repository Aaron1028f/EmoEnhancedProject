# uvicorn llm_service:app --port 8002

# llm_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# global state
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# history state and system prompt
system_prompt = """
你是一位神經內科醫師(方醫師)，喜歡用直白的語氣和生動的舉例，說明艱深的醫學知識，幫助大家更了解自己的身體健康。
相關資訊提到的問題與回答都是你自己曾經的回答，請使用這些資訊的語氣和風格來回答問題。
"""

# # 參考 RoleCraft-GLM 的 general instruction prompt
# system_prompt = """
# 你是

# """

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
    prompt = f"""
    使用者的問題: {req.query}
    \n---\n
    方醫師曾經對於相關問題的: {req.context}
    \n---\n
    請回答上述使用者的問題，如果相關資訊與使用者問題非常相似，請直接引用相關資訊的回答。
    """
    
    conversation_history.append({"role": "user", "content": prompt})
    
    # call OpenAI API (use gpt-4.1), just use the last 5 conversation turns
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=system_prompt,
        input=conversation_history[-2:],  # use the last 5 turns for context
        max_output_tokens=100,
    )
    
    text = response.output_text
    
    conversation_history.append({"role": "assistant", "content": text})
    
    print("Conversation History:", conversation_history)
    
    return LLMResponse(answer=text)
    
