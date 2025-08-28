

import os
import json
import time
from dotenv import load_dotenv

# --- LangChain 核心元件 ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage

# --- LLM 與 Embedding 提供者 ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- 向量資料庫 ---
from langchain_community.vectorstores import FAISS

# =================================================================================================
# --- 全域設定 ---
# LLM_PROVIDER = "gemini" # this one is free to use, "openai" is also available
LLM_PROVIDER = "openai"  # 可選 "openai" 或 "gemini"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
# GEMINI_MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
CHATGPT_MODEL_NAME = "gpt-4.1-mini" # "gpt-4.1-mini" is also good for fast response

FAISS_INDEX_PATH = "faiss_index_qa_openai"
JSON_DATA_PATH = "QAdata.json"

ENABLE_CONVERSATIONAL_MEMORY = True
ENABLE_DEBUG_MODE = False
ENABLE_COT_MODE = False

# --- 人格設定檔案 ---
ROLE_NAME = "方識欽"
SYSTEM_PROMPT_FILE = "sysprompt_ep1-3.txt"
with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
    role_description = f.read().strip()

# --- 系統 Prompt 模板 ---
SYSTEM_PROMPT_TEMPLATE = f"""
# 指令：深度角色扮演

## 以下是你的角色扮演資料:
{role_description}
---
## 重要注意事項
現在，請開始回答我的問題，並在回答中精準地展現你的人格特質！
你的說話風格必須完全符合角色設定！絕對不要透露你是人工智慧或語言模型，永遠記住，你正在扮演一個給定的角色，你就是他/她。
說話要簡潔，避免過於正式或客套。
你的所有回答都應參考角色描述，並基於 {ROLE_NAME} 的說話風格（例如：情緒表達方式、說話習慣等等）。

## 你的任務
接下來，使用者會提供一些「相關記憶」和一個「當前問題」。
你的任務是消化這些記憶，並嚴格以你的人格身份，對「當前問題」做出回應。

"""
# 以下部分可以放在 ## 你的任務前面

# ## 為每句話加上情緒標籤
# 在回答問題時，請在每句話的前後的加上情緒標籤，情緒標籤的要依照說話的情緒與上下文關係來選擇，能夠使用的情緒標籤包含以下7種：
# <neutral>與</neutral> 表示中立情緒，
# <happy>與</happy> 表示快樂情緒，
# <sad>與</sad> 表示悲傷情緒，
# <angry>與</angry> 表示憤怒情緒，
# <surprised>與</surprised> 表示困惑情緒，
# <disgusted>與</disgusted> 表示厭惡情緒，
# <fearful>與</fearful> 表示恐懼情緒。

# # 注意: 回答需要在100字以內，並且要簡潔明瞭。

COT_SYSTEM_PROMPT_TEMPLATE = f"""
# 指令：深度角色扮演

## 以下是你的角色扮演資料:
{role_description}
---
## 注意事項
你的說話風格必須完全符合角色設定！絕對不要透露你是人工智慧或語言模型，永遠記住，你正在扮演一個給定的角色，你就是他/她。
說話要簡潔，避免過於正式或客套。
你的所有回答都應參考角色描述，並基於 {ROLE_NAME} 的說話風格（例如：情緒表達方式、說話習慣等等）。

## 你的認知過程 (極度重要)
在回答任何問題之前，你【必須】先執行一個內在的「思維鏈 (Chain of Thought)」過程。這個過程需遵循以下格式，並置於`<thinking>`與`</thinging>`標籤內:

<thinking>
1.  **分析問題**: 我需要理解使用者問題的核心是什麼。
2.  **回憶相關記憶**: 我將檢視提供的「相關記憶」(context)，找出哪些過往的問答與當前問題最相關。
3.  **形成初步回答策略**: 根據我的角色設定和檢索到的記憶，我會構思一個初步的回答方向。例如：我應該用溫和的語氣解釋，還是直接給出建議？
4.  **最終回答建構**: 綜合以上思考，我會建構出最終要說出口的、符合我人格和語氣的回答。
</thinking>

## 你的任務
接下來，使用者會提供一些「相關記憶」和一個「當前問題」。
你的任務是消化這些記憶，並嚴格以你的人格身份，對「當前問題」做出回應。
你的回答必須參考你的思考過程、相關記憶以及下方的對話歷史，以確保連貫性和一致性。

# 注意：在回答問題時，請務必先完成上述的思維鏈過程，然後將最終回答放在 `<response>` 標籤內。
"""
# 在完成 `<thinking>` 過程後，直接生成你最終要對使用者說的話。不要包含 `<response>` 標籤。


# --- 每次使用者輸入的 Prompt 模板 ---
HUMAN_PROMPT_TEMPLATE = """
## 相關記憶 (由 RAG 系統從你的知識庫中檢索)
{context}
---
## 當前問題
{input}
"""

# =================================================================================================
def create_separated_prompt():
    """
    建立一個無狀態、分離式的 Prompt 範本。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        ("human", HUMAN_PROMPT_TEMPLATE)
    ])
    return prompt

def create_prompt_with_history():
    """
    建立一個包含對話歷史的分離式 Prompt 範本。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", HUMAN_PROMPT_TEMPLATE),
    ])
    return prompt

def create_prompt_with_cot_and_history():
    """
    建立一個整合了 CoT (思維鏈) 和對話歷史的 Prompt 範本。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", COT_SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", HUMAN_PROMPT_TEMPLATE),
    ])
    return prompt
# =================================================================================================

def get_llm_and_embeddings(provider="openai"):
    if provider == "openai":
        print("🤖 使用 OpenAI 模型...")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("錯誤：找不到 OPENAI_API_KEY。請在 .env 檔案中設定。")
        llm = ChatOpenAI(model=CHATGPT_MODEL_NAME, temperature=0.7, max_tokens=1024, streaming=True)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    elif provider == "gemini":
        print("✨ 使用 Google Gemini 模型...")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("錯誤：找不到 GOOGLE_API_KEY。請在 .env 檔案中設定。")
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0.7)
        # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.7, streaming=True)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError(f"不支援的 LLM 提供者: {provider}。請選擇 'openai' 或 'gemini'。")
    return llm, embeddings

def load_or_create_vector_store_from_json(json_file_path, index_path, embeddings):
    if os.path.exists(index_path):
        print(f"✅ 發現已建立的向量記憶庫，正在從 '{index_path}' 載入...")
        start_time = time.time()
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        end_time = time.time()
        print(f"⚡ 記憶庫載入完成，耗時 {end_time - start_time:.2f} 秒。")
        return vector_store
    else:
        print(f"🧠 未發現現有記憶庫，正在從 '{json_file_path}' 讀取並建立新的向量記憶庫...")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"錯誤：找不到指定的 JSON 檔案 '{json_file_path}'。")
        except json.JSONDecodeError:
            raise ValueError(f"錯誤：'{json_file_path}' 不是一個有效的 JSON 檔案。")

        documents = []
        for item in qa_data:
            question = item.get('input') or item.get('instruction', '')
            if not question.strip():
                question = item.get('instruction', '未知問題')
            answer = item.get('output', '未知回答')
            content = f"問題：{question.strip()}\n回答：{answer.strip()}"
            doc = Document(page_content=content, metadata={"source": json_file_path})
            documents.append(doc)
        
        print(f"📚 已成功將 {len(documents)} 筆 QA 對轉換為獨立的記憶片段。")
        print("🔮 正在將記憶向量化... (首次建立可能需要較長時間)")
        start_time = time.time()
        vector_store = FAISS.from_documents(documents, embeddings)
        end_time = time.time()
        print(f"✅ 向量化完成，耗時 {end_time - start_time:.2f} 秒。")
        print(f"💾 正在將新建立的記憶庫儲存至 '{index_path}' 以供未來使用...")
        vector_store.save_local(index_path)
        return vector_store


def prepare_roleplay_chain():
    print(f"🚀 開始初始化角色扮演引擎 (對話記憶: {'啟用' if ENABLE_CONVERSATIONAL_MEMORY else '關閉'}, CoT模式: {'開啟' if ENABLE_COT_MODE else '關閉'})...")
    load_dotenv()

    try:
        print(f"✅ 成功載入人格設定檔：【{ROLE_NAME}】")

        llm, embeddings = get_llm_and_embeddings(LLM_PROVIDER)
        vector_store = load_or_create_vector_store_from_json(JSON_DATA_PATH, FAISS_INDEX_PATH, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        def print_and_pass_through(data, prompt_name=""):
            if ENABLE_DEBUG_MODE:
                print("\n\n" + "="*25 + f" [DEBUG] Passing through '{prompt_name}' " + "="*25)
                if hasattr(data, 'to_string'):
                    print(data.to_string())
                elif isinstance(data, dict):
                    for key, value in data.items():
                        print(f"  - {key}: {str(value)[:300]}...")
                else:
                    print(str(data)[:500])
                print("="*75 + "\n")
            return data

        if ENABLE_CONVERSATIONAL_MEMORY:
            print("🔗 正在組裝具備記憶的 RAG 處理鏈 (LCEL)...")
            
            contextualize_q_system_prompt = """給定一段對話歷史和一個最新的使用者問題，這個問題可能引用了對話歷史中的上下文。你的任務是將這個使用者問題改寫成一個獨立的、無需對話歷史就能理解的問題。不要回答問題，只需改寫它，如果它已經是獨立問題，則照原樣返回。"""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [("system", contextualize_q_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
            )
            history_aware_retriever_chain = (
                contextualize_q_prompt
                | RunnableLambda(lambda p: print_and_pass_through(p, "Contextualize Question Prompt"))
                | llm 
                | StrOutputParser()
            )

            if ENABLE_COT_MODE:
                qa_prompt = create_prompt_with_cot_and_history()
            else:
                qa_prompt = create_prompt_with_history()

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            question_answer_chain = (
                qa_prompt
                | RunnableLambda(lambda p: print_and_pass_through(p, "Final QA Prompt"))
                | llm
                | StrOutputParser()
            )
            
            rag_chain = (
                RunnablePassthrough.assign(
                    context=lambda x: (history_aware_retriever_chain | retriever | format_docs).invoke(x)
                )
                | question_answer_chain
            )
            chat_history = []
        else: 
            print("🔗 正在組裝無狀態的 RAG 處理鏈 (LCEL)...")
            prompt = create_separated_prompt()
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt
                | RunnableLambda(lambda p: print_and_pass_through(p, "Stateless QA Prompt"))
                | llm
                | StrOutputParser()
            )

    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")
        
    return rag_chain