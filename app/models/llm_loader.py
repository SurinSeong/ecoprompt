from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from app.core.config import base_settings

# 전역 변수 정의
MODEL_NAME = base_settings.base_model + "/v_latest"

llm: Optional[AutoModelForCausalLM] = None
llm_tokenizer: Optional[AutoTokenizer] = None

async def load_llm():
    """vLLM 엔진을 초기화하고 전역 변수에 할당"""
    global llm

    # 이미 있다면
    if llm is not None:
        print("LLM already loaded.")
        return

    # 없다면
    print("⏳ Starting LLM Load from loader...")
    try:

        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            device_map="auto"
        )
        print("✅ LLM loaded successfully.")

    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")       


def get_llm() -> AutoModelForCausalLM:
    """초기화된 LLM 객체를 반환"""
    global llm

    if llm is None:
        raise RuntimeError("LLM Engine is not initialized.")
    return llm


def load_tokenizer():
    """토크나이저 로드하기"""
    global llm_tokenizer
    
    # 이미 있다면
    if llm_tokenizer is not None:
        print("Tokenizer already loaded.")
        return

    # 없다면
    print("⏳ Starting Tokenizer Load from loader...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
        print("✅ Tokenizer loaded successfully.")

    except Exception as e:
        print(f"❌ Failed to load Tokenizer: {e}")


def get_tokenizer() -> AutoTokenizer:
    """초기화된 토크나이저 객체를 반환"""
    global llm_tokenizer

    if llm_tokenizer is None:
        raise RuntimeError("Tokenizer is not initialized.")
    return llm_tokenizer
