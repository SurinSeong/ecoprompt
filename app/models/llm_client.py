# llm 호출 관련
import re

from app.models.llm_loader import tokenizer, llm, sampling_params

# SYSTEM_MESSAGE
SYSTEM_MESSAGE = "당신은 친절하고 똑똑한 SSAFY AI 챗봇입니다. SSAFY (Samsung SW AI Academy For Youth) 재학생들을 위해 알맞은 답변을 제공하세요. 항상 한국어로 답변해주세요."

# 공통 메시지 생성 및 프롬프트 구성 함수
def build_prompt(user_input: str, context: str = "") -> str:
    full_context = f"{context}\n\n{user_input}" if context else user_input

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": full_context}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# 비동기 generator에서 최종 응답만 수집하기
async def get_last_output(agen) -> str:
    last_text = ""
    async for item in agen:
        if item.outputs and item.outputs[0].text:
            last_text = item.outputs[0].text

    return last_text


# 공통 llm 호출 함수
async def llm_generate(prompt_str: str, request_id: str) -> str:
    agen = llm.generate(prompt_str, sampling_params, request_id=request_id)
    result = await get_last_output(agen)
    return result if result else "empty:" + prompt_str 


total_responses = {}

# 스트리밍 기반 챗봇 응답 함수
def get_chat_response_stream(user_input: str, request_id: str):
    sent_text = ""
    prompt_str = build_prompt(user_input=user_input)
    agen = llm.generate(prompt_str, sampling_params, request_id=request_id)

    async def _wrapper():
        nonlocal sent_text
        async for result in agen:
            # 응답이 부족하거나 outputs이 비어있으면 건너뛴다.
            if not result.outputs or len(result.outputs) < 2:
                continue
            
            # chosen 답변 - 토큰
            chosen_text = result.outputs[0].text
            new_text = chosen_text[len(sent_text):]
            sent_text = chosen_text

            for word in re.findall(f'\s+|\S+', new_text):
                # 사용자에게 chosen 응답 전송
                yield word

            # rejected 답변 - 누적
            rejected_text = result.outputs[1].text

            if result.finished:
                total_responses["chosen"] = chosen_text
                total_responses["rejected"] = rejected_text
                total_responses["request_id"] = request_id
                break
            
    return _wrapper()
