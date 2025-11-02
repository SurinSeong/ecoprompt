from pydantic import BaseModel, Field
from typing import Optional

# 응답
class ChatResponse(BaseModel):
    message_id: str    # UUID
    llm_response: str[Optional]    # None일 수 있음. => 에러일 때
    error_message: str[Optional]


# 요청
class ChatRequest(BaseModel):
    user_input: str
    message_id: str    # UUID


# structured outputs
class Responses(BaseModel):
    """LLM 답변"""
    message_id: str = Field(..., description="The UUID of the user input")    # UUID
    chosen: str = Field(..., description="The correct answer about the user input")
    rejected: str = Field(..., description="The wrong answer about the user input")
    