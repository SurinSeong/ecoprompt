# lm-evaluation-harness 평가

## 1. 설치

```bash
# 프로젝트 다운로드
git clone https://github.com/EleutherAI/lm-evaluation-harness

# 프로젝트 디렉터리로 이동
cd lm-evaluation-harness

# Harness에서 사용하는 의존성 패키지 설치
uv pip install -e .
```

- 설치 완료 후 lm_eval 명령어로 Harness 실행 가능!

## 2. 실행

```bash
# 예시
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### 인자 살펴보기

- `--model`: 테스트 진행할 모델의 위치 선택
- `--model-args`: 모델을 실행하기 위해 필요한 인자 설정. pretrained=/path/to/model/
- `--tasks`: 모델이 어떤 작업을 테스트할지 결정. ','로 구분해 여러 개의 작업을 선택할 수 있고, 작업 그룹을 기입해 그룹에 속한 모든 작업을 테스트할 수 있음.
- `--device`: 모델이 사용할 기기 설정 (ex. `cuda`, `cuda:0`, `cpu`, `mps`)
- `--batch_size`: 평가에 사용할 배치 크기 설정. 양수 정수로 설정 or `auto`로 설정해 메모리에 맞는 가장 큰 배치 크기를 자동으로 선택

## 3. 커스텀 모델 테스트하기

- 파인튜닝한 단일 모델이나 RAG LLM 어플리케이션은 Harness에 따로 등록되어 있지 않기 때문에 LM이라는 추상 클래스를 개발하여 새로운 모델로 추가해야 한다.
- 작업 또한 각자의 니즈에 따라 작성한 데이터 세트와 프롬프트, 지표를 테스트하기 위해 YAML 파일을 이용해 커스터마이징할 수 있다.