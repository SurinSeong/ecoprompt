FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 사전 폴더 구축하기
WORKDIR /app
RUN mkdir -p model

# 파이썬 환경 설정
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    python3-pip \
    python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# 필수 패키지 설치
RUN pip3 install --no-cache-dir --upgrade pip && \ 
    pip3 install --no-cache-dir \
    vllm==0.10.2 \
    datasets==4.1.1 \
    matplotlib \
    math-verify \
    gensim

CMD [ "/bin/bash" ]