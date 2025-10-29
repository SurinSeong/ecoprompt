# SSAFY 관련 질문이 아닌 경우 => 수정 필요.
basic_prompt = """
사용자와의 이전 대화 기록, 출력 형식, 예시, 출력 조건에 맞추어 응답하세요.
---
[History]:
{history}

[Question]:
{question}
"""

# SSAFY 관련 질문일 경우 => 수정 필요.
about_ssafy_prompt = """
사용자와의 이전 대화 기록, 출력 형식, 예시, 출력 조건에 맞추어 응답하세요.

[출력 형식]:
1. 

---
[출력 형식 예시]:
질문: "SSAFY 출결 관련 내용 알려줘."


---
[출력 조건]:
1. 모든 문장은 존댓말로 작성합니다.

---
[History]:
{history}

[Context]:
{context}

[Question]:
{question}
"""