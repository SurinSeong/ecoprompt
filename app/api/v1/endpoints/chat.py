from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import generate_sse_stream, stream_response
from app.models.llm_loader import get_llm_engine, get_llm, load_sampling_params, get_tokenizer
from app.models.load_for_rag import get_vector_store

router = APIRouter()

@router.post("")
async def chat(request: ChatRequest, llm=Depends(get_llm), tokenizer=Depends(get_tokenizer), vector_store=Depends(get_vector_store)):
    """
    스트림 답변 제공
    - 아직 chosen, rejected 구분하지 않음.
    """
    user_input = request.user_input
    personal_prompt = request.personal_prompt
    message_uuid = request.message_uuid

    chain = stream_response(vector_store=vector_store, llm=llm, tokenizer=tokenizer)
    
    # 사용자 대화 히스토리 불러오기
    
    payload = {
        "personal_prompt": personal_prompt,
        "question": user_input,
        "history": "",
        "context": ""
    }

    async def event_generator():
        state = "SEEK_OPEN_CHOSEN"
        chosen_response = ""
        rejected_response = ""

        OPEN_C  = "<CHOSEN>"
        CLOSE_C = "</CHOSEN>"
        OPEN_R  = "<REJECTED>"
        CLOSE_R = "</REJECTED>"
        
        sequence_id = -1
        try:
            for chunk in chain.stream(payload):
                if not chunk:
                    continue
                
                sequence_id += 1

                if (state == "SEEK_OPEN_CHOSEN") and (OPEN_C in chunk):
                    state = "FOUND_CHOSEN"
                    yield f"data: {ChatResponse(sequence_id=sequence_id, token="START").model_dump_json()}\n"
                    continue
                

                if (state == "FOUND_CHOSEN") and (OPEN_C not in chunk) and (CLOSE_C not in chunk):
                    chosen_response += chunk
                    yield f"data: {ChatResponse(sequence_id=sequence_id, token=chunk).model_dump_json()}\n"
                    continue

                
                if (state == "FOUND_CHOSEN") and (OPEN_C not in chunk) and (CLOSE_C in chunk):
                    state = "END_CHOSEN"
                    yield f"data: {ChatResponse(sequence_id=sequence_id, token='DONE').model_dump_json()}\n"
                    continue


                if (state == "END_CHOSEN") and (OPEN_R in chunk):
                    state = "FOUND_REJECTED"
                    continue


                if (state == "FOUND_REJECTED") and (OPEN_R not in chunk) and (CLOSE_R not in chunk):
                    rejected_response += chunk
                    continue


                if (state == "FOUND_REJECTED") and (OPEN_R not in chunk) and (CLOSE_R in chunk):
                    break

            print(f"[CHOSEN]\n{chosen_response}")

        except Exception as e:
            # 오류 시 스트림 종료
            yield f"data: [ERROR] {type(e).__name__}: {e}\n\n"

        print(f"[REJECTED]\n{rejected_response}")
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# @router.post("")
# async def stream(request: ChatRequest, llm=Depends(get_llm_engine), tokenizer=Depends(get_tokenizer)):
#     """vLLM 스트림을 받아 SSE 형식으로 클라이언트에게 응답"""
#     sampling_params = load_sampling_params()
#     request_id = request.request_id
#     user_input = request.user_input

#     # 최종 결과를 담을 컨테이너 생성
#     final_responses = {"chosen": "", "rejected": ""}

#     # 서비스 함수 호출
#     stream_generator = generate_sse_stream(
#         llm,
#         request_id,
#         user_input,
#         sampling_params,
#         final_responses
#     )

#     # DB에 rejected 저장하기

#     return StreamingResponse(stream_generator, media_type="text/event-stream")
