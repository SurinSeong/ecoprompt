import json

from fastapi import APIRouter
from app.core.config import evaluate_settings
from app.services.evaluate import evaluate_model

router = APIRouter()

@router.post("/", status_code=200)
def evaluate(model_path: str):
    """모델 평가하기"""
    print(f"평가할 task: {evaluate_settings.eval_tasks}")

    results = evaluate_model(model_path=model_path)

    with open(f"{evaluate_settings.eval_output_path}/{evaluate_settings.eval_tasks}-results.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("평가 결과 저장이 완료되었습니다.")

    return {"is_completed": True}
