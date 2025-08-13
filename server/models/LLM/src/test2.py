import httpx
import sys

url = "http://localhost:8000/streaming_response"

# request with user input
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    print("AI: ", end="")
    sys.stdout.flush()

    # 用於儲存 (標籤, 句子) 的列表
    tagged_sentences = []
    
    # 用於解析的暫存變數
    current_tag = ""
    current_content = ""
    in_tag = False
    is_closing_tag = False

    try:
        with httpx.stream("GET", url, params={"user_input": user_input}, timeout=60) as response:
            response.raise_for_status()
            for text_chunk in response.iter_text():
                for char in text_chunk:
                    if char == '<':
                        # 遇到 '<'，先檢查是否需要儲存上一段內容
                        # 只有在 current_tag 和 current_content 都有值時才儲存
                        if current_tag and current_content:
                            tagged_sentences.append((current_tag, current_content.strip()))
                            # =========================
                            # 可以在這邊將 tagged_sentences 傳給TTS做處理
                            # =========================
                        
                        # 重設狀態以準備處理新標籤
                        in_tag = True
                        is_closing_tag = False
                        current_content = "" # 內容在新標籤開始時就重設
                        current_tag = ""     # 標籤也重設

                    elif char == '>':
                        in_tag = False
                    elif in_tag:
                        if char == '/':
                            is_closing_tag = True
                        # 只在不是結束標籤時才記錄標籤名稱
                        elif not is_closing_tag:
                            current_tag += char
                    else:
                        # 如果不在標籤內，則是對使用者可見的內容
                        current_content += char
                        print(char, end="", flush=True)
        
        # 處理串流結束後最後一組未儲存的內容
        if current_tag and current_content:
            tagged_sentences.append((current_tag, current_content.strip()))

    except httpx.RequestError as exc:
        print(f"\n請求錯誤: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"\n伺服器錯誤: {exc.response.status_code}")

    print() # 確保 AI 回覆後換行

    # 在串流結束後，印出擷取到的列表以供驗證
    print("-" * 20)
    print("擷取到的標籤與內容：")
    print(tagged_sentences)
    print("-" * 20)