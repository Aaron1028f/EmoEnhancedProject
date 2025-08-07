from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    聊天請求的資料模型。
    """
    user_input: str = Field(
        ...,
        description="使用者輸入的文字內容。",
        examples=["你好嗎？"]
    )
    session_id: str = Field(
        ...,
        description="用於追蹤對話上下文的唯一識別碼。",
        examples=["user123_session456"]
    )

class ChatResponse(BaseModel):
    """
    聊天回應的資料模型。
    """
    response: str = Field(
        ...,
        description="聊天機器人生成的回應內容。",
        examples=["<neutral>我很好，謝謝關心。</neutral>"]
    )
