from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from vllm.sampling_params import RequestOutputKind
from vllm import SamplingParams

from app.models.prompt_template import routing_prompt


load_dotenv()


def get_sampling_params(prompt_type: str) -> SamplingParams:
    """prompt_type에 따라 SamplingParams 객체를 생성합니다."""
    if prompt_type == "chosen":
        temperature = 0.3
        return SamplingParams(
            max_tokens=2048,
            temperature=temperature,
            top_p=0.95,
            seed=42,
            repetition_penalty=1.01,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            output_kind=RequestOutputKind.DELTA,
        )

    elif prompt_type == "rejected":
        temperature = 0.9
        return SamplingParams(
            max_tokens=2048,
            temperature=temperature,
            top_p=0.95,
            seed=42,
            repetition_penalty=1.01,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            output_kind=RequestOutputKind.DELTA,
        )

def get_router_sampling_params(tokenizer) -> SamplingParams:

    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    return SamplingParams(
        max_tokens=64,
        temperature=0.1,
        top_p=0.95,
        seed=42,
        frequency_penalty=1.3,
        stop_token_ids=eos_token_id
    )

async def find_question_type(llm_engine_2, tokenizer_2):
    """사용자 질문의 타입을 분류한다."""

    def build_prompt_with_routing_template(user_info: dict) -> str:
        """
        """
        question = str(user_info.get("question", ""))

        messages = [
            {"role": "system", "content": routing_prompt},
            {"role": "user", "content": question},
        ]

        return tokenizer_2.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    make_prompt_route = RunnableLambda(build_prompt_with_routing_template)

    async def call_vllm_engine_router(inputs: dict):
        """vLLM Qwen 엔진을 호출하여 비동기 스트리밍을 시작한다."""
        request_id = inputs.get("message_uuid", "")
        prompt = inputs.get("prompt", "")
        sampling_params = get_router_sampling_params(tokenizer_2)
        result_generator = llm_engine_2.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        async for request_output in result_generator:
            if request_output.outputs:
                result = request_output.outputs[0].text
            
            if request_output.finished:
                break

        return result

    router_chain = (
        RunnableParallel(
            prompt=make_prompt_route,
            message_uuid=lambda x: x["message_uuid"]
        )
        | RunnableLambda(call_vllm_engine_router)
    )

    return router_chain


def stream_response_vllm(llm_engine_1, llm_engine_2, tokenizer_1, tokenizer_2, prompt_type, question_type):
    """
    Langchain LCEL을 사용하여 vLLM 스트리밍 체인을 구성합니다.
        llm_engine: vllm을 통한 llm 모델 서빙 엔진
        tokenizer: llm에 알맞은 tokenizer
        prompt_type: chosen or rejected
    """

    def build_prompt_with_qwen_template(user_info: dict) -> str:
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

        return tokenizer_1.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def build_prompt_with_llama_template(user_info: dict) -> str:
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

        return tokenizer_2.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    make_prompt_qwen = RunnableLambda(build_prompt_with_qwen_template)

    make_prompt_llama = RunnableLambda(build_prompt_with_llama_template)

    async def call_vllm_engine_1(inputs: dict):
        """vLLM Qwen 엔진을 호출하여 비동기 스트리밍을 시작한다."""
        request_id = inputs.get("message_uuid", "")
        prompt = inputs.get("prompt", "")
        sampling_params = get_sampling_params(prompt_type)
        result_generator = llm_engine_1.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        async for request_output in result_generator:
            for completion in request_output.outputs:
                new_text = completion.text
                if new_text:
                    yield new_text

            if request_output.finished:
                return
    
    async def call_vllm_engine_2(inputs: dict):
        """vLLM Qwen 엔진을 호출하여 비동기 스트리밍을 시작한다."""
        request_id = inputs.get("message_uuid", "")
        prompt = inputs.get("prompt", "")
        sampling_params = get_sampling_params(prompt_type)
        result_generator = llm_engine_2.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        async for request_output in result_generator:
            for completion in request_output.outputs:
                new_text = completion.text
                if new_text:
                    yield new_text

            if request_output.finished:
                return

    if question_type == "code" or question_type == "algorithm":        
        qwen_chain = (
            RunnableParallel(
                prompt=make_prompt_qwen,
                message_uuid=lambda x: x["message_uuid"]
            )
            | RunnableLambda(call_vllm_engine_1)
        )
        return qwen_chain

    if question_type == "ssafy" or question_type == "general":
        llama_chain = (
            RunnableParallel(
                prompt=make_prompt_llama,
                message_uuid=lambda x: x["message_uuid"]
            )
            | RunnableLambda(call_vllm_engine_2)
        )

        return llama_chain