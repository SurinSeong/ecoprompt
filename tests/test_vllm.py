import os

os.environ["VLLM_USE_V1"] = "1"

import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from transformers import AutoTokenizer

MODEL_PATH = "./local-models/Qwen2.5-Coder-0.5B-Instruct"    # 원본 모델 경로

# MODEL_PATH = "./local-models/Midm-2.0-Mini-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    trust_remote_code=True
)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

rag_prompt = """사용자와의 이전 대화 기록과 주어진 컨텍스트를 참고해서, 답변을 반환하세요.
아래 출력 형식을 100% 준수해주세요.

[출력 형식]
<CHOSEN>
선호 답변
</CHOSEN>

<REJECTED>
비선호 답변
</REJECTED>

---
[규칙]:
1) 무조건 모든 답변은 각 태그 안에 마크다운 형태로 출력해주세요. 
2) 선호 답변은 사용자의 질문에 알맞은 답변입니다. 자세하고 정확하고 친절한 답변을 제공하세요. 코드를 알려달라는 질문에는 코드블록을 사용해서 코드를 출력하고 그에 대한 설명도 함께 <CHOSEN> 태그 안에 제공해주세요.
3) 비선호 답변은 사용자가 궁금증을 해결하기에는 부족하다고 생각하는 답변입니다. 알맞지 않은 답변을 <REJECTED> 태그 안에 작성하세요.
4) 반드시 출력형식을 지켜서, 선호 답변, 비선호 답변 모두 생성해주세요.
5) Context가 있으면 우선 활용하고, 없으면 아는 내용은 반환하고, 모호한 내용은 모른다고 솔직히 말한 뒤 질문 재요청 또는 검색을 제안하세요.
6) 이전 대화와 문맥이 이어지면 History를 반영해주세요.
7) 민감/위험 주제는 안전 가이드를 준수하세요.
8) </REJECTED> 이후에는 답변을 제공하지 마세요.
9) 사용자 지침이 있다면 꼭 참고해서 답변하세요.
10) 기본적으로 한국어로 답변하세요.

---
"""

async def stream_response(engine: AsyncLLM, prompt: str, request_id: str) -> None:
    print(f"\n🚀 Prompt: {prompt!r}")
    print("💬 Response: ", end="", flush=True)

    # smapling params 설정
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.3,
        top_p=0.95,
        seed=42,
        output_kind=RequestOutputKind.DELTA,
        repetition_penalty=1.01,
        frequency_penalty=0.2,
        presence_penalty=0.1
    )

    messages = [
        {"role": "system", "content": rag_prompt},
        {"role": "user", "content": prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    try:
        async for output in engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id
        ):
            for completion in output.outputs:
                new_text = completion.text
                if new_text:
                    print(new_text, end="", flush=True)

            if output.finished:
                print("\n✅ Generation complete!")
                break

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        raise


async def main():
    print("🔧 Initializing AsyncLLM...")
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enforce_eager=True,
        gpu_memory_utilization=0.75,
        trust_remote_code=True,
        # quantization="bitsandbytes",
        max_model_len=8192

    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        prompts = [
            "파이썬 merge sort에 대해 코드 작성하고 간단하게 설명해줘.",
        ]
        print(f"🎯 Running {len(prompts)} streaming examples...")

        for i, prompt in enumerate(prompts, 1):
            request_id = f"stream-example-{i}"
            await stream_response(engine, prompt, request_id)

            if i < len(prompts):
                await asyncio.sleep(0.5)

        print("\n🎉 All streaming examples completed!")
    
    finally:
        print("🔧 Shutting down engine...")
        engine.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")