from pydantic import BaseModel, Field
from typing import Optional

# 응답
class ChatResponse(BaseModel):
    request_id: str    # UUID
    llm_response: str | None    # None일 수 있음. => 에러일 때


# 요청 : JAVA 서버에서 보낸 JSON 데이터와 일치 시키기
class ChatRequest(BaseModel):
    user_input: str
    request_id: str    # UUID


# structured outputs
class Responses(BaseModel):
    """LLM 답변"""
    request_id: str = Field(..., description="The UUID of the user input")    # UUID
    chosen: str = Field(..., description="The correct answer about the user input")
    rejected: str = Field(..., description="The wrong answer about the user input")
    
