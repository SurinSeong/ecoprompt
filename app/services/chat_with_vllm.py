from vllm import SamplingParams

from langchain_core.runnables import RunnableLambda

def get_sampling_params(prompt_type: str) -> SamplingParams:
    """prompt_type에 따라 SamplingParams 객체를 생성합니다."""
    if prompt_type == "chosen":
        temperature = 0.3

    elif prompt_type == "rejected":
        temperature = 0.9

    return SamplingParams(
        max_tokens=2048,
        temperature=temperature,
        top_p=0.95,
        seed=42,
        repetition_penalty=1.01,
        frequency_penalty=0.2,
        presence_penalty=0.1
    )

def stream_response(llm_engine, tokenizer, prompt_type):
    """
    Langchain LCEL을 사용하여 vLLM 스트리밍 체인을 구성합니다.
        llm_engine: vllm을 통한 llm 모델 서빙 엔진
        tokenizer: llm에 알맞은 tokenizer
        prompt_type: chosen or rejected
    """

    def build_prompt_with_template(user_info: dict) -> str:
        """
        user_info 딕셔너리를 받아 Chat Template을 적용한 최종 프롬프트를 생성합니다.
            x: {
                "service_prompt": str
                "question" : str,
                "history": str,
                "context": str,
                "personal_prompt": str
            }
        """
        # 입력 데이터 타입 강제 변환 및 기본값 설정
        service_prompt = str(user_info.get("service_prompt", ""))
        question = str(user_info.get("question", ""))
        history = str(user_info.get("history", ""))
        context = str(user_info.get("context", ""))
        personal_prompt = str(user_info.get("personal_prompt", ""))

        if not isinstance(question, str):
            question = str(question)

        if not isinstance(history, str):
            history = str(history)

        if not isinstance(context, str):
            context = str(context)

        if not isinstance(personal_prompt, str):
            personal_prompt = str(personal_prompt)

        system_prompt = (
            service_prompt + 
            "\n---\n[사용자 지침]\n" + personal_prompt + 
            "\n\n[History]\n" + history + 
            "\n\n[Context]\n" + context +
            "\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    make_prompt = RunnableLambda(build_prompt_with_template)

    async def call_vllm_engine(prompt: str, message_uuid: str):
        """vLLM 엔진을 호출하여 비동기 스트리밍을 시작한다."""
        request_id = message_uuid
        sampling_params = get_sampling_params(prompt_type)
        result_generator = llm_engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        async for request_output in result_generator:
            for completion in request_output.outputs:
                new_text = completion.text
                if new_text:
                    yield new_text

            if request_output.finished:
                yield "DONE"
    
    chain = make_prompt | RunnableLambda(call_vllm_engine)

    return chain
    