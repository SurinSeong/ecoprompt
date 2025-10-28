from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import get_response

router = APIRouter()

@router.post("/", response_model=ChatResponse, status_code=200)
def chat(request: ChatRequest):
    """LLM에게 응답 받기(SSE)"""

    user_input = request.user_input
    message_id = request.message_id
    try:
        print("LLM 답변 생성중..")
        total_response = get_response(user_input)

        # 선호 응답
        llm_response = total_response["chosen"]

        # 비선호 응답
        rejected_response = total_response["rejected"]
        # MongoDB에 저장하기

        return {"message_id": message_id, "llm_response": llm_response}
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return {"error_message": e}