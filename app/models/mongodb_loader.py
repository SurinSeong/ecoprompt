import os
from dotenv import load_dotenv
from pymongo import AsyncMongoClient
from typing import Optional

load_dotenv()

# 전역변수 설정
mongo_client: Optional[AsyncMongoClient] = None

async def load_mongodb():
    """mongodb 초기화"""
    global mongo_client

    if mongo_client is not None:
        print("MongoDB already loaded.")
        return
    
    print("⏳ Starting MongoDB Load from loader...")

    try:
        mongo_client = AsyncMongoClient(
            os.getenv("MONGO_URL")
        )
        print("✅ MongoDB loaded successfully.")
    
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")


def get_mongodb() -> AsyncMongoClient:
    """초기화된 MongoDB 객체 반환"""
    global mongo_client

    if mongo_client is None:
        raise RuntimeError("MongoDB is not initialized.")
    return mongo_client
