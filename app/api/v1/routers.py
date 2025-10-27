from fastapi import APIRouter

# 각 엔드포인트별로 라우터 임포트
from app.api.v1.endpoints import evaluate, train, chat

# API v1에 대한 메인 라우터 만들기
api_router = APIRouter()

# prefix와 tag를 포함
api_router.include_router(
    evaluate.router,
    prefix="/evaluate",
    tags=["evaluate"]
)

api_router.include_router(
    train.router,
    prefix="/train",
    tags=["train"]
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"]
)