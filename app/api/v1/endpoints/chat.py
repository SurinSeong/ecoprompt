import asyncio
from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import stream_chosen_response_vllm, generate_rejected_response_vllm, find_question_type
from app.services.routing import parse_router_response
from app.models.llm_loader import get_llm_engine_1, get_llm_engine_2, get_tokenizer_1, get_tokenizer_2
# from app.models.vectordb_loader import get_vector_store
from app.models.mongodb_loader import get_mongodb
from app.services.use_mongodb import find_chatting_id, get_chat_history
from app.models.prompt_template import chosen_prompt, rejected_prompt
from app.core.concurrency import get_limiter
from app.core.config import base_settings

router = APIRouter()

@router.post("/vllm", status_code=status.HTTP_201_CREATED)
async def chat_vllm(request: ChatRequest, llm_engine_1=Depends(get_llm_engine_1), llm_engine_2=Depends(get_llm_engine_2), tokenizer_1=Depends(get_tokenizer_1), tokenizer_2=Depends(get_tokenizer_2), mongo_client=Depends(get_mongodb)):
    """
    스트림 답변 제공 (동시성 제어 포함)
    """
    limiter = get_limiter()

    try:
        # 동시 요청 수 제한
        async with limiter.limit():

            user_input = request.user_input
            personal_prompt = request.personal_prompt
            message_uuid = str(request.message_uuid)
            
            try:
                # 사용자 대화 히스토리 불러오기
                chatting_id = await find_chatting_id(mongo_client, message_uuid)
                print(f"[채팅 번호] {chatting_id}")

                chat_history = await get_chat_history(mongo_client, int(chatting_id))
                print(f"[채팅 기록] {chat_history}")

            except Exception as e:
                print(f"ERROR {e}")
                chatting_id = None
                print(f"[채팅 번호] {chatting_id}\n=> 채팅 기록이 없습니다.")
                chat_history = ""

            # 질문 라우팅
            router_chain = await find_question_type(llm_engine_2=llm_engine_2, tokenizer_2=tokenizer_2)
            router_payload = {
                "message_uuid": message_uuid,
                "question": user_input,
            }
            print(router_payload)

            router_response = await router_chain.ainvoke(router_payload)
            print(f"[ROUTER]\n{router_response}")

            question_type = parse_router_response(router_response)

            # 답변 생성 체인
            chosen_chain = stream_chosen_response_vllm(llm_engine_1=llm_engine_1, llm_engine_2=llm_engine_2, tokenizer_1=tokenizer_1, tokenizer_2=tokenizer_2, prompt_type="chosen", question_type=question_type)
            chosen_payload = {
                "message_uuid": message_uuid,
                "service_prompt": chosen_prompt,
                "personal_prompt": personal_prompt,
                "question": user_input,
                "history": chat_history,
                "context": ""    # 벡터 DB 연결해봐야 함.
            }

            rejected_chain = generate_rejected_response_vllm(llm_engine_1=llm_engine_1, llm_engine_2=llm_engine_2, tokenizer_1=tokenizer_1, tokenizer_2=tokenizer_2, prompt_type="rejected", question_type=question_type)
            rejected_payload = {
                "message_uuid": message_uuid + "-rejected",
                "service_prompt": rejected_prompt,
                "personal_prompt": personal_prompt,
                "question": user_input,
                "history": chat_history,
                "context": ""    # 벡터 DB 연결해봐야 함.
            }
            
                    
            async def event_generator():
                chosen_response = ""
                sequence_id = 0
                rejected_task = None

                try:
                    timeout = base_settings.request_timeout

                    async with asyncio.timeout(timeout):
                        yield f"data: {ChatResponse(sequence_id=sequence_id, token='START').model_dump_json()}\n\n"
                    
                        # rejected 생성을 백그라운드에서 시작
                        rejected_task = asyncio.create_task(rejected_chain.ainvoke(rejected_payload))

                        async for chunk in chosen_chain.astream(chosen_payload):

                            if not chunk:
                                continue
                            
                            sequence_id += 1
                            chosen_response += chunk

                            yield f"data: {ChatResponse(sequence_id=sequence_id, token=chunk).model_dump_json()}\n\n"

                        yield f"data: {ChatResponse(sequence_id=sequence_id+1, token='DONE').model_dump_json()}\n\n"
                        print(f"[CHOSEN]\n{chosen_response}")
                    
                        rejected_response = await rejected_task

                        yield f"data: {ChatResponse(sequence_id=-1, token=rejected_response).model_dump_json()}\n\n"
                        print(f"[REJECTED]\n{rejected_response}")

                except asyncio.TimeoutError:
                    # ⏰ 타임아웃 발생
                    print(f"⏰ [{message_uuid}] 타임아웃 발생 ({base_settings.request_timeout}초)")
                    yield f"data: {ChatResponse(sequence_id=-999, token='ERROR: Request timeout. Please try again with a shorter prompt.').model_dump_json()}\n\n"
                    
                    # rejected task 취소
                    if rejected_task and not rejected_task.done():
                        rejected_task.cancel()

                except Exception as e:
                    # 오류 시 스트림 종료
                    print(f"❌ [{message_uuid}] 에러 발생: {type(e).__name__}: {str(e)}")
                    yield f"data: [ERROR] {type(e).__name__}: {e}\n\n"

                    # rejected task 취소
                    if rejected_task and not rejected_task.done():
                        rejected_task.cancel()

            return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    except asyncio.TimeoutError:
        # 상위 타임아웃
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "server timeout. Please try again later."}
        )

    except Exception as e:
        # 일반 에러
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Internal error: {str(e)}"}
        )
