# ecoprompt

- SSAFY LLM 어플리케이션 AI 서버

## 프로젝트 구조

```
[api]    →  [schemas]   →  [services]   →  [core / model]
엔드포인트   데이터검증       비즈니스로직       환경설정 / 추론

llm/
├── llama.cpp/                              
├── app/                                    # 메인 애플리케이션 패키지
│   ├── api/                                # API 엔드포인트
│   │   └── v1/                             # API v1 버전
│   │       ├── endpoints/                  # API 엔드포인트 구현
│   │       │   ├── llm.py
│   │       │   ├── health.py
│   │       │   └── ...
│   │       └── routers.py
│   ├── core/                               # 설정, 보안, 유틸성 모듈
│   │   ├── config.py                       # 환경 변수 로드, 전역 설정
│   │   └── celery_app.py                   # 비동기
│   ├── schemas/                            # Pydantic 스키마 모음
│   │   ├── llm_request.py                  # 요청 데이터 구조
│   │   └── llm_response.py                 # 응답 데이터 구조
│   └── services/                           # 비즈니스 로직, 모델 추론 로직
│       ├── llm_service.py
│       └── model_loader.py
│
├── models/                                 # 사용할 모델
│   └── Llama-3.1-Korean-8B-Instruct/
│
├── data/                                   # 학습에 사용할 데이터 
│   ├── code_data/                          # 코딩 파인튜닝 관련
│   └── qa_data/                            # 사용자 질의응답 관련
│
├── training/                               # LLM 학습 관련
│   └── fine_tuning.py                
├── tests/                                  # 테스트 코드
│   └── test_inference.py                   # LLM 추론 테스트
│ 
├── main.py                                 # FastAPI 애플리케이션 엔트리포인트
├── pyproject.toml                          # Python 의존성 패키지
├── env.example                             # 환경 변수 설정 예시
└── README.md                               # 프로젝트 문서
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

# 파인튜닝용 라이브러리 설치
uv add accelerate peft bitsandbytes trl
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

```bash
uv run app/services/model_download.py
```

5. 파인튜닝용 데이터셋

- LoRA 파인튜닝을 위한 데이터셋을 `data` 폴더에 담는다.

