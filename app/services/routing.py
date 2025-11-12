def parse_router_response(response: str) -> str:
    """라우터 응답을 안전하게 파싱합니다."""
    if not response:
        return "general"  # 기본값
    
    # 응답을 소문자로 변환하고 공백 제거
    response_clean = response.lower().strip()
    
    # 가능한 패턴들을 체크
    valid_types = ["code", "algorithm", "ssafy", "general"]
    
    # 직접 매칭 시도
    for valid_type in valid_types:
        if valid_type in response_clean:
            return valid_type
    
    # 키워드 기반 분류 (fallback)
    if any(keyword in response_clean for keyword in ["코드", "프로그래밍", "개발", "함수"]):
        return "code"
    elif any(keyword in response_clean for keyword in ["알고리즘", "자료구조", "문제해결"]):
        return "algorithm"
    elif any(keyword in response_clean for keyword in ["ssafy", "싸피", "교육", "과정"]):
        return "ssafy"
    else:
        return "general"  # 기본값