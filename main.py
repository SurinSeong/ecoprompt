from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.v1.routers import api_router
from app.models.llm_loader import load_llm_engine, llm_engine, load_tokenizer, llm_tokenizer

# lifespan 컨텍스트 관리자 정의
@asynccontextmanager
async def lifespan_manager(app: FastAPI):

    # 🚀 서버 시작 (Startup) 로직
    load_tokenizer()    # Tokenizer
    await load_llm_engine()    # LLM 모델 로드 (GPU 메모리 상주 시작)
    print("Application startup complete!")

    # yield가 실행되면 서버가 요청을 받기 시작함.
    yield

    # 🛑 서버 종료 (Shutdown) 로직
    if llm_engine is not None:
        pass


# 1. FastAPI 인스턴스 생성
app = FastAPI(
    title="Ecoprompt Main LLM",
    description="API Documentation",
    version="1.0.0",
    lifespan=lifespan_manager
)

# 2. 경로 작동 함수 (Route Operation) 정의
@app.get("/")
def read_root():
    """
    HTTP GET 요청이 루트 경로('/')로 들어왔을 때 실행되는 함수
    """
    return {"message": "Hello, FastAPI"}

@app.get("/health")
def health():
    return {"status": "ok"}

# 라우터 연결
app.include_router(api_router, prefix="/api/v1/ai")