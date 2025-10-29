import asyncio
import time

# # 모델에게 답변 받기
async def generate_response(user_input: str, message_id: str):
    return ""

#     all_responses = {}    # 2개의 응답을 저장할 딕셔너리
#     response_texts = ["", ""]

#     while True:
#         request_outputs = llm.step()

#         for output in request_outputs:
#             if output.request_id == message_id:

#                 # output.outputs는 n개의 SampleOutput을 포함
#                 for i, sample_output in enumerate(output.outputs):
#                     text = sample_output.text

#                     # 1.스트리밍 처리하기 : 선호 답변
#                     if i == 0:
#                         new_text = text[len(response_texts[i]):]
#                         response_texts[i] = text

#                         # 띄어쓰기 단위로 새로운 텍스트를 yield
#                         for word in new_text.split(" "):
#                             if word:    # 빈 문자열은 제외한다.
#                                 yield {"chunk": word + " "}    # <-- 딕셔너리 형태로 yield

#                     # 요청이 완료되면 종료 (각 응답의 최종 텍스트를 저장)
#                     if output.finished:
#                         # n개의 응답을 모두 저장
#                         all_responses[i] = text
                
#                 # 요청이 완료되면 스트리밍을 종료하고 최종 응답을 반환
#                 if output.finished:
#                     chosen_text = all_responses.get(0, "")
#                     rejected_text = all_responses.get(1, "")

#                     final_result = {
#                         "chosen": chosen_text,
#                         "rejected": rejected_text
#                     }

#                     return final_result
                
#         await asyncio.sleep(0.1)

