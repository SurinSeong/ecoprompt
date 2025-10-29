import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import generate_response
from app.models.llm_client import llm_generate

router = APIRouter()

@router.post("/test", response_model=ChatResponse)
def test_chat(request: ChatRequest):
    """테스트용"""
    user_input = request.user_input
    request_id = request.message_id

    llm_response = llm_generate(user_input, request_id)

    return {
        "request_id": request_id,
        "llm_response": llm_response,
    }


@router.post("", response_model=ChatResponse, status_code=200)
def chat(request: ChatRequest):
    """사용자 질문 받아서 LLM 답변(선호/비선호) 생성하기"""

    user_input = request.user_input
    message_id = request.message_id

    try:
        print("LLM 답변 생성중..")
        total_response = generate_response(user_input, message_id)

        async def event_stream():
            async for chunk in total_response:
                payload = {"delta": chunk}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            # 스트리밍 완료를 알리는 이벤트
            yield "event: end_of_stream\ndata: {}\n\n"

        # # 선호 응답
        # llm_response = total_response["chosen"]

        # # 비선호 응답
        # rejected_response = total_response["rejected"]
        # MongoDB에 저장하기

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return {"error_message": e}