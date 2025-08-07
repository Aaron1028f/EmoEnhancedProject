import os
import json
import time
from dotenv import load_dotenv
from typing import AsyncGenerator, Dict, List

# --- LangChain 核心元件 ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- LLM 與 Embedding 提供者 ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- 向量資料庫 ---
from langchain_community.vectorstores import FAISS

# --- 專案模組 ---
from config import (
    LLM_PROVIDER, GEMINI_MODEL_NAME, CHATGPT_MODEL_NAME,
    FAISS_INDEX_PATH, JSON_DATA_PATH, ENABLE_CONVERSATIONAL_MEMORY,
    ENABLE_DEBUG_MODE, ENABLE_COT_MODE, ROLE_NAME, SYSTEM_PROMPT_FILE,
    SYSTEM_PROMPT_TEMPLATE, COT_SYSTEM_PROMPT_TEMPLATE, HUMAN_PROMPT_TEMPLATE
)

class RolePlayRAG:
    """
    一個封裝了 RAG 聊天機器人所有邏輯的類別。
    """
    def __init__(self):
        print("🚀 開始初始化角色扮演引擎...")
        load_dotenv()
        self._load_role_description()
        self.llm, self.embeddings = self._get_llm_and_embeddings()
        self.vector_store = self._load_or_create_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.rag_chain = self._create_rag_chain()
        self.chat_histories: Dict[str, List[HumanMessage | AIMessage]] = {}
        print(f"🎭 角色扮演系統已就緒！化身：【{ROLE_NAME}】")
        print(f"   - 當前使用模型: {LLM_PROVIDER.upper()}")
        print(f"   - 對話記憶模式: {'啟用' if ENABLE_CONVERSATIONAL_MEMORY else '關閉'}")
        print(f"   - CoT 模式: {'開啟' if ENABLE_COT_MODE and ENABLE_CONVERSATIONAL_MEMORY else '關閉'}")


    def _load_role_description(self):
        try:
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                role_description = f.read().strip()
            self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(role_description=role_description, ROLE_NAME=ROLE_NAME)
            self.cot_system_prompt = COT_SYSTEM_PROMPT_TEMPLATE.format(role_description=role_description, ROLE_NAME=ROLE_NAME)
            print(f"✅ 成功載入人格設定檔：【{ROLE_NAME}】")
        except FileNotFoundError:
            raise RuntimeError(f"錯誤：找不到人格設定檔 '{SYSTEM_PROMPT_FILE}'。")

    def _get_llm_and_embeddings(self):
        if LLM_PROVIDER == "openai":
            print("🤖 使用 OpenAI 模型...")
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("錯誤：找不到 OPENAI_API_KEY。請在 .env 檔案中設定。")
            llm = ChatOpenAI(model=CHATGPT_MODEL_NAME, temperature=0.7, max_tokens=1024, streaming=True)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        elif LLM_PROVIDER == "gemini":
            print("✨ 使用 Google Gemini 模型...")
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("錯誤：找不到 GOOGLE_API_KEY。請在 .env 檔案中設定。")
            llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0.7, streaming=True)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 仍然可以使用 OpenAI 的 embedding
        else:
            raise ValueError(f"不支援的 LLM 提供者: {LLM_PROVIDER}。請選擇 'openai' 或 'gemini'。")
        return llm, embeddings

    def _load_or_create_vector_store(self):
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"✅ 發現已建立的向量記憶庫，正在從 '{FAISS_INDEX_PATH}' 載入...")
            start_time = time.time()
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
            print(f"⚡ 記憶庫載入完成，耗時 {time.time() - start_time:.2f} 秒。")
            return vector_store
        else:
            print(f"🧠 未發現現有記憶庫，正在從 '{JSON_DATA_PATH}' 讀取並建立新的向量記憶庫...")
            try:
                with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"錯誤：找不到指定的 JSON 檔案 '{JSON_DATA_PATH}'。")
            except json.JSONDecodeError:
                raise ValueError(f"錯誤：'{JSON_DATA_PATH}' 不是一個有效的 JSON 檔案。")

            documents = []
            for item in qa_data:
                question = item.get('input') or item.get('instruction', '未知問題')
                answer = item.get('output', '未知回答')
                content = f"問題：{question.strip()}\n回答：{answer.strip()}"
                doc = Document(page_content=content, metadata={"source": JSON_DATA_PATH})
                documents.append(doc)
            
            print(f"� 已成功將 {len(documents)} 筆 QA 對轉換為獨立的記憶片段。")
            print("🔮 正在將記憶向量化... (首次建立可能需要較長時間)")
            start_time = time.time()
            vector_store = FAISS.from_documents(documents, self.embeddings)
            print(f"✅ 向量化完成，耗時 {time.time() - start_time:.2f} 秒。")
            print(f"💾 正在將新建立的記憶庫儲存至 '{FAISS_INDEX_PATH}' 以供未來使用...")
            vector_store.save_local(FAISS_INDEX_PATH)
            return vector_store

    def _print_and_pass_through(self, data, prompt_name=""):
        if ENABLE_DEBUG_MODE:
            print("\n\n" + "="*25 + f" [DEBUG] Passing through '{prompt_name}' " + "="*25)
            if hasattr(data, 'to_string'):
                print(data.to_string())
            elif isinstance(data, dict):
                for key, value in data.items():
                    print(f"   - {key}: {str(value)[:300]}...")
            else:
                print(str(data)[:500])
            print("="*75 + "\n")
        return data

    def _create_rag_chain(self):
        if ENABLE_CONVERSATIONAL_MEMORY:
            print("🔗 正在組裝具備記憶的 RAG 處理鏈 (LCEL)...")
            
            contextualize_q_system_prompt = """給定一段對話歷史和一個最新的使用者問題，這個問題可能引用了對話歷史中的上下文。你的任務是將這個使用者問題改寫成一個獨立的、無需對話歷史就能理解的問題。不要回答問題，只需改寫它，如果它已經是獨立問題，則照原樣返回。"""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [("system", contextualize_q_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
            )
            history_aware_retriever_chain = (
                contextualize_q_prompt
                | RunnableLambda(lambda p: self._print_and_pass_through(p, "Contextualize Question Prompt"))
                | self.llm 
                | StrOutputParser()
            )

            qa_prompt_template = self.cot_system_prompt if ENABLE_COT_MODE else self.system_prompt
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", HUMAN_PROMPT_TEMPLATE),
            ])

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            question_answer_chain = (
                qa_prompt
                | RunnableLambda(lambda p: self._print_and_pass_through(p, "Final QA Prompt"))
                | self.llm
                | StrOutputParser()
            )
            
            rag_chain = (
                RunnablePassthrough.assign(
                    context=lambda x: (history_aware_retriever_chain | self.retriever | format_docs).invoke(x)
                )
                | question_answer_chain
            )
            return rag_chain
        else: 
            print("🔗 正在組裝無狀態的 RAG 處理鏈 (LCEL)...")
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", HUMAN_PROMPT_TEMPLATE)
            ])
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            rag_chain = (
                {"context": self.retriever | format_docs, "input": RunnablePassthrough()}
                | prompt
                | RunnableLambda(lambda p: self._print_and_pass_through(p, "Stateless QA Prompt"))
                | self.llm
                | StrOutputParser()
            )
            return rag_chain

    def _get_session_history(self, session_id: str) -> List[HumanMessage | AIMessage]:
        """根據 session_id 獲取或創建對話歷史"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        return self.chat_histories[session_id]

    def clear_history(self, session_id: str):
        """清理指定 session_id 的對話歷史"""
        if session_id in self.chat_histories:
            del self.chat_histories[session_id]
            print(f"🧹 已清除 Session ID '{session_id}' 的對話歷史。")

    async def get_response(self, user_input: str, session_id: str) -> str:
        """處理單次請求並回傳完整答案"""
        if ENABLE_CONVERSATIONAL_MEMORY:
            chat_history = self._get_session_history(session_id)
            stream_input = {"input": user_input, "chat_history": chat_history}
            full_response = await self.rag_chain.ainvoke(stream_input)
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=full_response))
            # 限制歷史長度
            if len(chat_history) > 10:
                self.chat_histories[session_id] = chat_history[-10:]
        else:
            full_response = await self.rag_chain.ainvoke(user_input)
            
        return full_response

    async def stream_response(self, user_input: str, session_id: str) -> AsyncGenerator[str, None]:
        """處理串流請求並逐步回傳結果"""
        full_response = ""
        if ENABLE_CONVERSATIONAL_MEMORY:
            chat_history = self._get_session_history(session_id)
            stream_input = {"input": user_input, "chat_history": chat_history}
            
            async for chunk in self.rag_chain.astream(stream_input):
                full_response += chunk
                yield chunk
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=full_response))
            if len(chat_history) > 10:
                self.chat_histories[session_id] = chat_history[-10:]
        else:
            async for chunk in self.rag_chain.astream(user_input):
                yield chunk