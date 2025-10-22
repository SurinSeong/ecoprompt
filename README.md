# ecoprompt

- SSAFY LLM 어플리케이션 AI 서버

## 프로젝트 구조

```
llm/
├── llama.cpp/                     # 양자화
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

# bash의 경우
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# zsh의 경우
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 설치 확인
uv --version
```

2. 가상 환경 설정하기
```
# 가상환경 설정
uv init --python 3.12
uv venv
source .venv/Scripts/activate

# 필수 라이브러리 설치
uv add fastapi vllm==0.10.2 datasets==4.1.1 matplotlib math-verify genism qdrant-client langchain langchain-community langchain-core
uv add accelerate
```

3. llama.cpp 설치
```
git clone https://github.com/ggml-org/llama.cpp.git

[환경설정]
# llama.cpp 폴더로 이동
cd llama.cpp

# llama.cpp의 readme처럼 build하기
cmake -B build
cmake --build build --config release

# build 중 혹시 에러 발생시
apt-get update
apt-get install -y libcurl4-openssl-dev pkg-config
```

4. 원하는 모델 설치

- Huggingface에서 원하는 모델을 찾아 app/services/model_download.py를 통해 설치한다.

