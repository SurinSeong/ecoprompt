FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 사전 폴더 구축하기
WORKDIR /app
COPY . /app
RUN mkdir -p model

# 파이썬 환경 설정 - uv 설치에 필요한 curl을 추가하고, 기존의 python3-pip 대신 uv 설치를 준비합니다.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    curl \
    # PPA 추가를 위한 패키지
    software-properties-common && \ 
    # deadsnakes PPA 추가
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    # Python 3.12 핵심 패키지 및 venv 모듈 설치
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv && \ 
    # uv 설치
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    # uv 실행 파일을 PATH에 등록된 위치로 이동
    mv /root/.cargo/bin/uv /usr/local/bin/uv && \
    # apt 캐시 정리
    rm -rf /var/lib/apt/lists/*

# 필수 패키지 설치 - uv
RUN uv pip install \
    fastapi \
    vllm==0.10.2 \
    datasets==4.1.1 \
    matplotlib \
    math-verify \
    gensim \
    qdrant-client

# 컨테이너 포트
EXPOSE 8000

# 컨테이너 시작하기
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]