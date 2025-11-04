# 사용자 질의 응답 스키마
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import Optional
from uuid import UUID

# 응답
"""
public record LlmResponse(
    String token,
    Integer sequenceId
) 
"""
class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    sequence_id: Optional[int] = Field(default=None, alias="sequenceId")   
    token: Optional[str] = None    # None일 수 있음. => 에러일 때
    
# 요청 : JAVA 서버에서 보낸 JSON 데이터와 일치 시키기
"""
public record LlmRequest(
    String personalPrompt,
    String userInput,
    String messageUUID
)
"""
class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    user_input: str = Field(alias="userInput")
    personal_prompt: str = Field(alias="personalPrompt")
    message_uuid: UUID = Field(alias="messageUUID")