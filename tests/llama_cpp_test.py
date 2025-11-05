from llama_cpp import Llama

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

llm = Llama(
    model_path="./local-models/Llama-SSAFY-8B_q4_k_m.gguf",
    chat_format="llama-3"
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an ai assistant who perfectly answer the question."},
        {
            "role": "user",
            "content": "What is the sum of the first 100 natural numbers?"
        }
    ]
)

print(response["choices"][0]["message"]["content"])