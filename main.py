from fastapi import FastAPI
from app.api.v1.routers import api_router

# 1. FastAPI 인스턴스 생성
app = FastAPI(
    title="Ecoprompt",
    description="API Documentation",
    version="1.0.0"
)

# 2. 경로 작동 함수 (Route Operation) 정의
@app.get("/")
def read_root():
    """
    HTTP GET 요청이 루트 경로('/')로 들어왔을 때 실행되는 함수
    """
    return {"message": "Hello, FastAPI"}

# 라우터 연결
app.include_router(api_router, prefix="/api/v1")