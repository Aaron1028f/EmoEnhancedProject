from openai import OpenAI

# 全域 state
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

system_prompt = "You are a helpful assistant."
conversation_history = [
    {"role": "system", "content": system_prompt}
]

def generate(req):
    # 將使用者的新訊息寫入對話歷史
    user_msg = req["context"] + "\nUser: " + req["query"]
    conversation_history.append({"role": "user", "content": user_msg})
    # 呼叫 API
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=system_prompt,
        input=conversation_history,
        max_output_tokens=100,
    )
    # 取出回應文字
    text = response.output_text
    # 將模型回應加入對話歷史
    conversation_history.append({"role": "assistant", "content": text})
    print("Conversation History:", conversation_history)
    return {"answer": text}

def main():
    print("多輪對話模式（輸入 'exit' 以結束）:")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        res = generate({"query": user_input, "context": ""})
        print("Assistant:", res.get("answer"))

if __name__ == "__main__":
    main()


