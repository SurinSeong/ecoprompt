from pydantic import BaseModel

# 응답
class ChatResponse(BaseModel):
    request_id: str    # UUID
    llm_response: str | None    # None일 수 있음. => 에러일 때


# 요청 : JAVA 서버에서 보낸 JSON 데이터와 일치 시키기
class ChatRequest(BaseModel):
    user_input: str
    request_id: str    # UUID