import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import generate_sse_stream
from app.models.llm_client import llm_generate
from app.models.llm_loader import get_llm_engine, load_sampling_params, get_tokenizer

router = APIRouter()

@router.post("/test", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    """테스트용 답변 받기"""
    user_input = request.user_input
    request_id = request.request_id

    llm_response = await llm_generate(user_input, request_id)

    return {
        "request_id": request_id,
        "llm_response": llm_response,
    }

@router.post("")
async def stream(request: ChatRequest, llm=Depends(get_llm_engine), tokenizer=Depends(get_tokenizer)):
    """vLLM 스트림을 받아 SSE 형식으로 클라이언트에게 응답"""
    sampling_params = load_sampling_params()
    request_id = request.request_id
    user_input = request.user_input

    # 최종 결과를 담을 컨테이너 생성
    final_responses = {"chosen": "", "rejected": ""}

    # 서비스 함수 호출
    stream_generator = generate_sse_stream(
        llm,
        request_id,
        user_input,
        sampling_params,
        final_responses
    )

    # DB에 rejected 저장하기

    return StreamingResponse(stream_generator, media_type="text/event-stream")
