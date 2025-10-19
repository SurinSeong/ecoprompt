# ecoprompt
SSAFY 자율 프로젝트

## llm-test

- 환경 설정

1. Ubuntu 실행
2. 필수 패키지 설치
```
# 시스템 패키지 업데이트/업그레이드
sudo apt update && sudo apt upgrade -y

# pip 패키지 설치
sudo apt install -y python3-pip

# venv 패키지 설치
sudo apt install -y python3.12-venv

# 가상 환경 생성 및 활성화
python3 -m venv practice-env
source practice-env/bin/activate

# 실습 필수 패키지 설치
pip3 install --upgrade pip
pip3 install vllm==0.10.2 litellm==1.77.5 datasets==4.1.1 matplotlib math-verify
```

3. 세팅 확인
```
# GPU 활성화 확인
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); [print(f'Device {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

4. Jupyter 환경 세팅
```
# jupyterlab, ipkernel 설치
pip3 install -U jupyterlab ipykernel

# jupyterlab 포트 설정 및 실행
jupyter lab --no-browser --ip=0.0.0.0 --port=8890
```

5. vllm 이용해서 gguf 모델 사용하기