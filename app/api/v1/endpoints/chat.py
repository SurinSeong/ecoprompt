import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import stream_response_vllm
from app.models.llm_loader import get_llm_engine, get_tokenizer
# from app.models.vectordb_loader import get_vector_store
from app.models.mongodb_loader import get_mongodb
from app.services.use_mongodb import find_chatting_id, get_chat_history
from app.models.prompt_template import chosen_prompt, rejected_prompt

router = APIRouter()

# @router.post("/transformers")
# async def chat(request: ChatRequest, llm=Depends(get_llm), tokenizer=Depends(get_tokenizer), vector_store=Depends(get_vector_store), mongo_client=Depends(get_mongodb)):
#     """
#     스트림 답변 제공
#     """
#     user_input = request.user_input
#     personal_prompt = request.personal_prompt
#     message_uuid = str(request.message_uuid)
    
#     try:
#         # 사용자 대화 히스토리 불러오기
#         chatting_id = await find_chatting_id(mongo_client, message_uuid)
#         print(f"[채팅 번호] {chatting_id}")

#         chat_history = await get_chat_history(mongo_client, int(chatting_id))
#         print(f"[채팅 기록] {chat_history}")

#     except Exception as e:
#         print(f"ERROR {e}")
#         chatting_id = None
#         print(f"[채팅 번호] {chatting_id}\n=> 채팅 기록이 없습니다.")
#         chat_history = ""

#     # 답변 생성 체인
#     chain = stream_response(vector_store=vector_store, llm=llm, tokenizer=tokenizer)
    
#     payload = {
#         "personal_prompt": personal_prompt,
#         "question": user_input,
#         "history": chat_history,
#         "context": ""    # 벡터 DB 연결해봐야 함.
#     }

#     async def event_generator():
#         state = "SEEK_OPEN_CHOSEN"
#         chosen_response = ""
#         rejected_response = ""

#         OPEN_C  = "<CHOSEN>"
#         CLOSE_C = "</CHOSEN>"
#         OPEN_R  = "<REJECTED>"
#         CLOSE_R = "</REJECTED>"
        
#         sequence_id = -1
#         try:
#             async for chunk in chain.astream(payload):
#                 # print(chunk)
#                 if not chunk:
#                     continue
                
#                 sequence_id += 1

#                 if (state == "SEEK_OPEN_CHOSEN") and (OPEN_C in chunk):
#                     state = "FOUND_CHOSEN"
#                     yield f"data: {ChatResponse(sequence_id=sequence_id, token="START").model_dump_json()}\n\n"
#                     continue
                

#                 if (state == "FOUND_CHOSEN") and (OPEN_C not in chunk) and (CLOSE_C not in chunk):
#                     chosen_response += chunk
#                     yield f"data: {ChatResponse(sequence_id=sequence_id, token=chunk).model_dump_json()}\n\n"
#                     continue

                
#                 if (state == "FOUND_CHOSEN") and (OPEN_C not in chunk) and (CLOSE_C in chunk):
#                     state = "END_CHOSEN"
#                     if chunk.strip() != CLOSE_C:
#                         chunk = chunk.replace(CLOSE_C, "")
#                         yield f"data: {ChatResponse(sequence_id=sequence_id, token=chunk).model_dump_json()}\n\n"
#                         yield f"data: {ChatResponse(sequence_id=sequence_id+1, token='DONE').model_dump_json()}\n\n"
#                         continue

#                     else:
#                         yield f"data: {ChatResponse(sequence_id=sequence_id, token='DONE').model_dump_json()}\n\n"
#                         continue


#                 if (state == "END_CHOSEN") and (OPEN_R in chunk):
#                     state = "FOUND_REJECTED"
#                     continue


#                 if (state == "FOUND_REJECTED") and (OPEN_R not in chunk) and (CLOSE_R not in chunk):
#                     # print(chunk)
#                     rejected_response += chunk
#                     continue


#                 if (state == "FOUND_REJECTED") and (OPEN_R not in chunk) and (CLOSE_R in chunk):
#                     break

#             print(f"[CHOSEN]\n{chosen_response}")
#             print(f"[REJECTED]\n{rejected_response}")

#             yield f"data: {ChatResponse(sequence_id=-1, token=rejected_response).model_dump_json()}\n\n"
        
#         except Exception as e:
#             # print(chunk)
#             # 오류 시 스트림 종료
#             yield f"data: [ERROR] {type(e).__name__}: {e}\n\n"

        

#         # # rejected response 저장하기
#         # if chatting_id:
#         #     result = await save_rejected_response(mongo_client, int(chatting_id), message_uuid, rejected_response)
#         #     if result:
#         #         print(f"[답변 저장] 완료")
#         # else:
#         #     print("chatting_id가 없어 답변을 저장할 수 없습니다.")

#     return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("")
async def chat_vllm(request: ChatRequest, llm_engine=Depends(get_llm_engine), tokenizer=Depends(get_tokenizer), mongo_client=Depends(get_mongodb)):
    """
    스트림 답변 제공
    """
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

    # 답변 생성 체인
    chosen_chain = stream_response_vllm(llm_engine=llm_engine, tokenizer=tokenizer, prompt_type="chosen")
    chosen_payload = {
        "message_uuid": message_uuid,
        "service_prompt": chosen_prompt,
        "personal_prompt": personal_prompt,
        "question": user_input,
        "history": chat_history,
        "context": ""    # 벡터 DB 연결해봐야 함.
    }

    rejected_chain = stream_response_vllm(llm_engine=llm_engine, tokenizer=tokenizer, prompt_type="rejected")
    rejected_payload = {
        "message_uuid": message_uuid,
        "service_prompt": rejected_prompt,
        "personal_prompt": personal_prompt,
        "question": user_input,
        "history": chat_history,
        "context": ""    # 벡터 DB 연결해봐야 함.
    }
    
            
    async def event_generator():
        chosen_response = ""

        sequence_id = 0

        try:
            yield f"data: {ChatResponse(sequence_id=sequence_id, token='START').model_dump_json()}\n\n"
            
            async for chunk in chosen_chain.astream(chosen_payload):
                # print(chunk)
                if not chunk:
                    continue
                
                sequence_id += 1
                chosen_response += chunk

                yield f"data: {ChatResponse(sequence_id=sequence_id, token=chunk).model_dump_json()}\n\n"

            yield f"data: {ChatResponse(sequence_id=sequence_id+1, token='DONE').model_dump_json()}\n\n"
            print(f"[CHOSEN]\n{chosen_response}")
        
            rejected_response = ""
            async for chunk in rejected_chain.astream(rejected_payload):
                if chunk:
                    rejected_response += chunk

            yield f"data: {ChatResponse(sequence_id=-1, token=rejected_response).model_dump_json()}\n\n"
            print(f"[REJECTED]\n{rejected_response}")

        except Exception as e:
            # print(chunk)
            # 오류 시 스트림 종료
            yield f"data: [ERROR] {type(e).__name__}: {e}\n\n"
            # raise HTTPException(status_code=500, detail="LLM Generation Error")
    

    return StreamingResponse(event_generator(), media_type="text/event-stream")
