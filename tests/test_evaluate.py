# """
# lm_eval --model hf \
#     --model_args pretrained=sh2orc/Llama-3.1-Korean-8B-Instruct \
#     --tasks humaneval,kmmlu,mbpp \
#     --device cuda:0


# lm_eval --model hf \
#     --model_args pretrained=sh2orc/Llama-3.1-Korean-8B-Instruct \
#     --tasks llama3 \
#     --device cuda:0
# """
# 파인튜닝 전 기존 모델 성능평가
import os
from lm_eval import evaluator

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
SEED = 42

# 평가 설정
model = "hf"
model_args = "pretrained=sh2orc/Llama-3.1-Korean-8B-Instruct"
tasks = ["kmmlu"]    # "humaneval", "mbpp"

# GPU 설정
device = "cuda:0"

# 평가 실행
results = evaluator.simple_evaluate(
    model=model,
    model_args=model_args,
    tasks=tasks,
    device=device,
    batch_size="auto",
    seed=SEED,
)

# 결과 출력
print(evaluator.make_table(results))