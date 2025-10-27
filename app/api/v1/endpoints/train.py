from fastapi import APIRouter
from app.core.config import train_settings
from app.services.train import train_model

router = APIRouter()

@router.post("/", status_code=200)
def train(model_path: str):
    """모델 학습하기"""
    print(f"학습할 모델: {model_path}")

    history = train_model(model_path)

    return {"is_completed": True}