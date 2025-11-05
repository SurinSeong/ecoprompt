import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "./local-models/Llama-SSAFY-8B/v_latest"    # 원본 모델 경로

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="./local-models/Llama-SSAFY-8B/v_latest"
    )

llm = LLM(
    model=MODEL_PATH,
    dtype=torch.bfloat16,
    max_model_len=4096,
    gpu_memory_utilization=0.75,
    tensor_parallel_size=2,
    quantization="bitsandbytes",
    load_format="bitsandbytes"
)

sampling_params_chosen = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=512,
    n=1
)

sampling_params_rejected = SamplingParams(
    temperature=1.5,
    top_p=0.95,
    max_tokens=512,
    n=1
)


rag_prompt = """사용자와의 이전 대화 기록과 주어진 컨텍스트를 참고해서, 한국어로 알맞은 답변을 반환하세요.
아래 출력 형식을 100% 준수하고, 태그 밖의 어떤 문자도 출력하지 마세요.

[출력 형식]
<ANSWER>
답변
</ANSWER>

---
[규칙]:
1) 태그 외의 문장, 접두사/접미사("답변:", "설명:"), 메타 코멘트, 인용부호, 마크다운 제목 등을 절대 출력하지 말 것.
2) 사용자의 질문에 알맞은 답변을 자세히 출력해주세요. 특히, 코드를 알려달라는 요청에는 코드블록을 사용해서 코드를 출력하고 그에 대한 간단한 설명도 제공해주세요.
3) Context가 있으면 우선 활용하고, 없으면 아는 내용은 반환하고, 모호한 내용은 모른다고 솔직히 말한 뒤 질문 재요청 또는 검색을 제안하세요.
4) 이전 대화와 문맥이 이어지면 History를 반영해주세요.
5) 민감/위험 주제는 안전 가이드를 준수하세요.
6) </ANSWER> 이후에는 답변을 제공하지 마세요.
7) 사용자 지침이 있다면 꼭 참고해서 답변하세요.

---
"""

def chat(user_input, personal_prompt, chat_history, context):

    def build_prompt_with_template(x: dict) -> str:
        user_input = x.get("user_input", "")
        chat_history = x.get("chat_history", "")
        context = x.get("context", "")
        personal_prompt = x.get("personal_prompt", "")

        if not isinstance(user_input, str): user_input = str(user_input)
        if not isinstance(chat_history, str): chat_history = str(chat_history)
        if not isinstance(context, str): context = str(context)
        if not isinstance(personal_prompt, str): personal_prompt = str(personal_prompt)

        system_prompt = (
            rag_prompt + 
            "\n---\n[사용자 지침]\n" + personal_prompt +
            "\n\n[History]\n" + chat_history +
            "\n\n[Context]\n" + context +
            "\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    payload = {
        "user_input": user_input,
        "chat_history": chat_history,
        "context": context,
        "personal_prompt": personal_prompt
    }

    input = build_prompt_with_template(payload)

    output_chosen = llm.generate(
        input,
        sampling_params=sampling_params_chosen,
    )
    chosen_text = output_chosen[0].outputs[0].text

    output_rejected = llm.generate(
        input,
        sampling_params=sampling_params_rejected
    )
    rejected_text = output_rejected[0].outputs[0].text


    return chosen_text, rejected_text


# 함수 실행
personal_prompt = ""
chat_history = ""
context = ""
user_input = "파이썬 merge sort 코드와 자세한 설명도 같이 제시해줘."

chosen, rejected = chat(user_input, personal_prompt, chat_history, context)

print("[선호응답]")
print(chosen)
print("[비선호응답]")
print(rejected)