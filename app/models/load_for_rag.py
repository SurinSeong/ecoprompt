import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 전역 변수 정의
embedding_model: Optional[HuggingFaceEmbeddings] = None
vector_store: Optional[QdrantVectorStore] = None

def load_embedding_model():
    """RAG를 위한 임베딩 모델 로드"""
    global embedding_model

    # 이미 있다면
    if embedding_model is not None:
        print("Embedding Model already loaded.")
        return
    
    # 없다면
    print("Starting Embedding Model Load from loader...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/multilingual-e5-small-ko")
        print("✅ Model loaded successfully.")

    except Exception as e:
        print(f"❌ Failed to load Embedding model: {e}")


def get_embedding_model() -> HuggingFaceEmbeddings:
    """초기화된 임베딩 모델 객체 반환"""
    global embedding_model

    if embedding_model is None:
        raise RuntimeError("Embedding Model is not initialized.")
    return embedding_model


def load_vectordb():
    """qdrant 벡터 db 로드"""
    global vector_store
    
    if vector_store is not None:
        print("Vector Store already loaded.")
        return
    
    # 없다면
    print("Starting Vector Store Load from loader...")
    try:

        vector_store = QdrantVectorStore.from_existing_collection(
            url="https://c5c72aa1-571c-4980-8091-f3fe5f10b794.us-west-1-0.aws.cloud.qdrant.io:6333",
            collection_name="ssafy",
            embedding=embedding_model,
            api_key=QDRANT_API_KEY
        )
        print("Vector Store loaded successfully.")
    
    except Exception as e:
        print(f"Failed to load Vector Store: {e}")
    

def get_vector_store() -> QdrantVectorStore:
    """초기화된 벡터 DB 객체 반환"""
    global vector_store

    if vector_store is None:
        raise RuntimeError("Vector Store is not initialized.")
    return vector_store