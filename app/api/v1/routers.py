from fastapi import APIRouter
from app.api.v1.endpoints import train, chat, pdf

# API에 대한 메인 라우터 만들기
api_router = APIRouter()

api_router.include_router(
    chat.router,
    prefix="/prompt-response",
    tags=["chat"]
)

api_router.include_router(
    train.router,
    prefix="/train",
    tags=["train"]
)

api_router.include_router(
    pdf.router,
    prefix="/pdf",
    tags=["pdf"]
)