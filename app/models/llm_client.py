from transformers import AutoTokenizer

# SYSTEM_MESSAGE
SYSTEM_MESSAGE = "당신은 친절하고 똑똑한 SSAFY AI 챗봇입니다. SSAFY (Samsung SW AI Academy For Youth) 재학생들을 위해 알맞은 답변을 제공하세요. 항상 한국어로 답변해주세요."

# 공통 메시지 생성 및 프롬프트 구성 함수
def build_prompt(tokenizer: AutoTokenizer, user_input: str, context: str = "") -> str:
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

