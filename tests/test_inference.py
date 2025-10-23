from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/Llama-3.1-Korean-8B-Instruct",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    trust_remote_code=True,
    dtype="auto",
    max_model_len=2048,
    gpu_memory_utilization=0.8
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=2048
)

def chat(instruction):
    # 사용자 프롬프트 추가
    messages = [
        {"role": "system", "content": "당신은 훌륭한 AI 비서입니다. 짧은 답변을 제시하고, 다음으로 상세 설명을 해주세요."},
        {"role": "user", "content": instruction}
    ]

    output = llm.chat(messages, sampling_params=sampling_params)
    text = output[0].outputs[0].text

    return text


# 함수 실행
instruction = "파이썬 merge sort 코드와 자세한 설명도 같이 제시해줘."

llm_answer = chat(instruction)

print(llm_answer)