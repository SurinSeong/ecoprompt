import os

os.environ["VLLM_USE_V1"] = "1"

import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from transformers import AutoTokenizer

MODEL_PATH_1 = "./local-models/Qwen2.5-Coder-0.5B-Instruct"    # 원본 모델 경로
MODEL_PATH_2 = "./local-models/Llama-3.2-1B-Instruct"       # llama 소형 모델

# MODEL_PATH = "./local-models/Midm-2.0-Mini-Instruct"

tokenizer_1 = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH_1,
    trust_remote_code=True
)

tokenizer_2 = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH_2,
)
tokenizer_2.pad_token = tokenizer_2.eos_token
tokenizer_2.padding_side = "left"

chosen_prompt = """사용자와의 이전 대화 기록과 주어진 컨텍스트를 참고해서, 사용자가 선호할만한 답변을 반환하세요.
아래 출력 형식을 100% 준수해주세요.

[출력 형식]
<ANSWER>
답변
</ANSWER>

---
[규칙]:
1) 무조건 답변은 태그 안에, 마크다운 형태로 출력해주세요. 
2) 선호할만한 답변은 사용자의 질문에 알맞은 답변입니다. 자세하고 정확하고 친절한 답변을 제공하세요. 코드를 알려달라는 질문에는 코드블록을 사용해서 코드를 출력하고 그에 대한 설명도 함께 제공해주세요.
3) 반드시 출력형식을 지켜서 답변을 생성해주세요.
4) Context가 있으면 우선 활용하고, 없으면 아는 내용은 반환하고, 모호한 내용은 모른다고 솔직히 말한 뒤 질문 재요청 또는 검색을 제안하세요.
5) 이전 대화와 문맥이 이어지면 History를 반영해주세요.
6) 민감/위험 주제는 안전 가이드를 준수하세요.
7) </ANSWER> 이후에는 답변을 제공하지 마세요.
8) 사용자 지침이 있다면 꼭 참고해서 답변하세요.
9) 기본적으로 한국어로 답변하세요.

---
"""

rejected_prompt = """
사용자와의 이전 대화 기록과 주어진 컨텍스트를 참고해서, 사용자가 선호하지 않을만한 답변을 반환하세요.
아래 출력 형식을 100% 준수해주세요.

[출력 형식]
<ANSWER>
답변
</ANSWER>

---
[규칙]:
1) 무조건 답변은 태그 안에, 마크다운 형태로 출력해주세요. 
2) 선호하지 않을만한 답변은 사용자의 질문에 알맞지 않은 답변입니다. 질문에 알맞지 않은 답변을 생성해주세요.
3) 반드시 출력형식을 지켜서 답변을 생성해주세요.
4) 이전 대화와 문맥이 이어지면 History를 반영해주세요.
5) 민감/위험 주제는 안전 가이드를 준수하세요.
6) </ANSWER> 이후에는 답변을 제공하지 마세요.
7) 사용자 지침이 있다면 꼭 참고해서 답변하세요.
8) 기본적으로 한국어로 답변하세요.

---
"""

async def stream_response(engine_1: AsyncLLM, engine_2: AsyncLLM, prompt: str, request_id: str) -> None:
    print(f"\n🚀 Prompt: {prompt}")
    print("💬 Response: ", end="", flush=True)

    # smapling params 설정
    sampling_params_1 = SamplingParams(
        max_tokens=2048,
        temperature=0.3,
        top_p=0.95,
        seed=42,
        output_kind=RequestOutputKind.DELTA,
        repetition_penalty=1.01,
        frequency_penalty=0.2,
        presence_penalty=0.1
    )

    sampling_params_2 = SamplingParams(
        max_tokens=2048,
        temperature=0.9,
        top_p=0.95,
        seed=42,
        repetition_penalty=1.01,
        frequency_penalty=0.2,
        presence_penalty=0.1
    )

    chosen_messages = [
        {"role": "system", "content": chosen_prompt},
        {"role": "user", "content": prompt}
    ]

    prompt_1 = tokenizer_1.apply_chat_template(
        chosen_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    rejected_messages = [
        {"role": "system", "content": rejected_prompt},
        {"role": "user", "content": prompt}
    ]

    prompt_2 = tokenizer_2.apply_chat_template(
        rejected_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        async for output in engine_1.generate(
            prompt=prompt_1, sampling_params=sampling_params_1, request_id=request_id
        ):
            for completion in output.outputs:
                new_text = completion.text
                if new_text:
                    print(new_text, end="", flush=True)

            if output.finished:
                print("\n✅ Generation complete!")
                break

        async for output in engine_2.generate(
            prompt=prompt_2, sampling_params=sampling_params_2, request_id=request_id
        ):  
            total_answer = ""

            if output.finished:
                total_answer = output.outputs[0].text
                print(total_answer, flush=True)
                print("\n✅ Generation complete!")
                break


    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        raise


async def main():
    print("🔧 Initializing AsyncLLM...")
    engine_args_1 = AsyncEngineArgs(
        model=MODEL_PATH_1,
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        trust_remote_code=True,
        quantization="fp8",
        max_model_len=8192

    )
    engine_1 = AsyncLLM.from_engine_args(engine_args_1)

    engine_args_2 = AsyncEngineArgs(
        model=MODEL_PATH_2,
        enforce_eager=True,
        gpu_memory_utilization=0.65,
        quantization="bitsandbytes",
        max_model_len=8192
    )
    engine_2 = AsyncLLM.from_engine_args(engine_args_2)

    try:
        prompt = "파이썬 merge sort에 대해 코드 작성하고 간단하게 설명해줘."
        print("🎯 Running streaming examples...")

        request_id = "stream-example-1"

        await stream_response(engine_1, engine_2, prompt, request_id)

        # if i < len(prompts):
        #     await asyncio.sleep(0.5)

        print("\n🎉 All streaming examples completed!")
    
    finally:
        print("🔧 Shutting down engine...")
        engine_1.shutdown()
        engine_2.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")