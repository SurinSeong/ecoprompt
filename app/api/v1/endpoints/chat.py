from fastapi import APIRouter
from app.services.chat import get_response

router = APIRouter()

@router.post("/", status_code=200)
def chat(model):
    """LLM에게 응답 받기(SSE)"""
    print("LLM 답변 스트리밍..")

    total_response = get_response()

    return {"response": total_response}