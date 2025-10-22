# ecoprompt
SSAFY 자율 프로젝트

## 프로젝트 구조

```
llm/
├── app/                           # 메인 애플리케이션 패키지
│   ├── api/                       # API 엔드포인트
│   │   └── v1/                    # API v1 버전
│   │       ├── endpoints/         # API 엔드포인트 구현
│   │       │   ├── 
│   │       └── 
│   ├── core/                      # 설정, 보안, 유틸성 모듈
│   │   ├── config.py              # 환경 변수 로드, 전역 설정
│   │   └── 
│   ├── schemas/                   # 스키마 모음
│   │   └── 
│   ├── services/                  # 비즈니스 로직 서비스
│   │   └── 
├── tests/                         # 테스트 코드
│   └── test_ai.py                 # AI API 테스트
│ 
├── main.py                        # 애플리케이션 진입점
├── pyproject.toml                 # Python 의존성 패키지
├── env.example                    # 환경 변수 설정 예시
└── README.md                      # 프로젝트 문서
```

## 환경설정

1. uv 설치
```
curl -LsSf https://astral.sh/uv/install.sh | sh

# 이후 나오는 거 하고 터미널 껐다가 켜기
```

2. 환경 설정하기
```
# 가상환경 설정
uv init --python 3.12
uv venv
source .venv/Scripts/activate

# 필수 라이브러리 설치
uv add fastapi vllm==0.10.2 matplotlib math-verify genism qdrant-client langchain langchain-community langchain-core

uv add datasets==4.1.1    # 서비스 실행할 때는 필요없음
```
