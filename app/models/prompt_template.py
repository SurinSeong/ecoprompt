# SSAFY 관련 질문이 아닌 경우 => 수정 필요.
basic_prompt = """
사용자와의 이전 대화 기록, 출력 예시, 출력 조건에 맞추어 무조건 한국어로 응답하세요.
아래의 예시를 반드시 지켜서 답변을 반환하세요.

[출력 형식 예시]:

질문: langchain에 대해서 알려줘.
답변: 
<chosen>
langchain은 대규모 언어 모델(LLM)을 활용하여 애플리케이션을 더 쉽게 개발할 수 있도록 돕는 오픈소스 프레임워크입니다.
더 자세한 내용을 알고 싶으시다면 Langchain을 검색해보세요!
</chosen>
<rejected>
langchain은 llm 관련 라이브러리입니다.
</rejected>

---
[출력 조건]
1. 모든 문장은 존댓말, 한국어로 작성합니다.
2. 앞뒤가 맞지 않는 말이라고 판단하면 질문을 다시 요청해주세요.
3. 잘 모르는 내용이라면 잘 모르는 내용이라고 솔직하게 답변하세요.
4. 이전 내용과 문맥이 이어지는 경우, History를 참고해서 답변해주세요.
5. 사용자 지침을 꼭 따라서 답변해주세요.
6. chosen에는 사용자가 선호할만한 답변을, rejected에는 사용자가 선호하지 않을 답변을 작성해주세요.

---
[History]:
{history}

[Question]:
{question}
"""

# SSAFY 관련 질문일 경우
rag_prompt = """사용자와의 이전 대화 기록과 주어진 컨텍스트를 참고해서, 한국어로 알맞은 답변을 반환하세요.
아래 출력 형식을 100% 준수하고, 태그 밖의 어떤 문자도 출력하지 마세요.

[출력 형식]
<CHOSEN>
{선호되는 최종 답변: 공손한 존댓말, 핵심부터 간결히. 코드가 필요하면 코드블록 사용 가능.}
</CHOSEN>
<REJECTED>
{덜 선호되는 답변: 사실과 크게 어긋나지 않지만 모호/불친절/핵심미흡하게. 위험·부적절·허위 정보 금지.}
</REJECTED>

---
[규칙]:
1) 태그 외의 문장, 접두사/접미사("답변:", "설명:"), 메타 코멘트, 인용부호, 마크다운 제목 등을 **절대 출력하지 말 것**.
2) 가능한 한 간결하고 명확하게. 불필요한 사족·사과문구 금지.
3) 컨텍스트가 있으면 우선 활용하고, 없으면 아는 내용은 반환하고, 모호한 내용은 모른다고 솔직히 말한 뒤 안전한 대안(질문 재요청/검색 제안)을 제시.
4) 이전 대화와 문맥이 이어지면 History를 반영.
5) 민감/위험 주제는 안전 가이드를 준수하고 추측으로 단정하지 말 것.
"""

# default_template = f"""
# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
# 당신은 친절하고 똑똑한 AI assistant 입니다. 사용자의 질문에 알맞은 답변을 반환해주세요.
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# {basic_prompt}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# Answer:
# """

# rag_template = f"""
# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
# 당신은 친절하고 똑똑한 AI assistant 입니다. 사용자의 질문에 알맞은 답변을 반환해주세요.
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# {about_ssafy_prompt}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# Answer:
# """

# from langchain_core.prompts import PromptTemplate

# prompt = PromptTemplate(
#     input_variables=["context", "question", "history"],
#     template=default_template
# )

# prompt.pretty_print()