# 파인튜닝 전 기존 모델 성능평가
import os, json
from lm_eval import evaluator
from app.core.config import evaluate_settings, base_settings

# 코드 평가 (task 실행 중 실제 코드 실행)를 허용하기 위한 안전 장치
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# DPO 파인튜닝한 모델 경로 확인
if base_settings.model_path:
    last_model = base_settings.model_path + "/" + os.listdir(base_settings.model_path)[-1]
    

def evaluate_model(model_path):
    
    # 평가 실행
    results = evaluator.simple_evaluate(
        model=evaluate_settings.eval_model,
        model_args=f"pretrained={model_path},trust_remote_code=True",
        tasks=evaluate_settings.eval_tasks,
        num_fewshot=evaluate_settings.eval_num_fewshot,
        seed=evaluate_settings.eval_seed,
        device=evaluate_settings.eval_device,
        log_samples=evaluate_settings.eval_log_samples,
        batch_size=evaluate_settings.eval_batch_size,
        output_path=evaluate_settings.eval_output_path
    )
    
    # 결과 출력
    print(evaluator.make_table(results))

    return results