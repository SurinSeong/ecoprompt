from pydantic import BaseModel
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