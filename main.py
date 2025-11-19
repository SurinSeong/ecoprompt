import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.v1.routers import api_router
from app.models.llm_loader import load_tokenizers, llm_tokenizer_1, llm_tokenizer_2, load_llm_engines, llm_engine_1, llm_engine_2
# from app.models.vectordb_loader import load_vectordb, vector_store, load_embedding_model, embedding_model
from app.models.mongodb_loader import load_mongodb, mongo_client
from app.core.concurrency import get_limiter
from app.core.config import base_settings

# lifespan 컨텍스트 관리자 정의
@asynccontextmanager
async def lifespan_manager(app: FastAPI):

    # 🚀 서버 시작 (Startup) 로직
    print("="*60)
    print("🚀 서버 시작 중...")
    print("="*60)

    # 동시성 제어 초기화
    limiter = get_limiter()
    print(f"✅ 동시성 제어: 최대 {base_settings.max_concurrent_requests}개 동시 요청")
    print(f"✅ 요청 타임아웃: {base_settings.request_timeout}초")

    load_tokenizers()    # Tokenizer
    # load_embedding_model()
    # load_vectordb()
    await load_llm_engines()
    await load_mongodb()    # MongoDB 로드

    print("="*60)
    print("✅ Application startup complete!")
    print("="*60)

    # yield가 실행되면 서버가 요청을 받기 시작함.
    yield

    # 🛑 서버 종료 (Shutdown) 로직
    print("\n🛑 서버 종료 중...")

    if limiter is not None:
        stats = limiter.get_stats()
        print(f"📊 최종 통계:")
        print(f"   - 총 처리 요청: {stats['total']}")
        print(f"   - 현재 활성 요청: {stats['active']}")

    # if vector_store is not None:
    #     pass

    if llm_tokenizer_1 is not None:
        pass

    if llm_tokenizer_2 is not None:
        pass

    # if embedding_model is not None:
    #     pass

    if mongo_client is not None:
        pass

    if llm_engine_1 is not None:
        pass

    if llm_engine_2 is not None:
        pass


# cuda 확인
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

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

@app.get("/stats")
def get_stats():
    """동시성 통계 확인"""
    try:
        limiter = get_limiter()
        return limiter.get_stats()
    
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    """헬스 체크 + 통계"""
    try:
        limiter = get_limiter()
        stats = limiter.get_stats()
        return {
            "status": "ok",
            "concurrency": stats,
            "config": {
                "max_concurrent_requests": base_settings.max_concurrent_requests,
                "request_timeout": base_settings.request_timeout
            }
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "concurrency": {"error": str(e)}
        }

# 라우터 연결
app.include_router(api_router, prefix="/api/v1/ai")