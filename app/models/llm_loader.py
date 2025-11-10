from typing import Optional

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from transformers import AutoTokenizer

from app.core.config import base_settings

# 전역 변수 정의
MODEL_NAME = base_settings.base_model + "/qwen"
# MODEL_NAME = "../quantization/Midm-2.0-Mini-Instruct"
# MODEL_NAME = "./local-models/Qwen2.5-Coder-7B-Instruct"

# llm: Optional[AutoModelForCausalLM] = None
llm_tokenizer: Optional[AutoTokenizer] = None
llm_engine: Optional[AsyncLLM] = None

# async def load_llm():
#     """vLLM 엔진을 초기화하고 전역 변수에 할당"""
#     global llm

#     # 이미 있다면
#     if llm is not None:
#         print("LLM already loaded.")
#         return

#     # 없다면
#     print("⏳ Starting LLM Load from loader...")
#     try:

#         llm = AutoModelForCausalLM.from_pretrained(
#             pretrained_model_name_or_path=MODEL_NAME
#         )
#         print("✅ LLM loaded successfully.")

#     except Exception as e:
#         print(f"❌ Failed to load LLM: {e}")       


# def get_llm() -> AutoModelForCausalLM:
#     """초기화된 LLM 객체를 반환"""
#     global llm

#     if llm is None:
#         raise RuntimeError("LLM is not initialized.")
#     return llm


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
            pretrained_model_name_or_path=MODEL_NAME,
            use_fast=True
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


async def load_llm_engine():
    """vllm으로 llm 엔진 초기화 하기"""
    global llm_engine

    # 이미 있다면
    if llm_engine is not None:
        print("LLM Engine already loaded.")
        return

    # 없다면
    print("⏳ Starting LLM Engine Load from loader...")
    
    try:
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            quantization="bitsandbytes",
            max_model_len=8192
        )

        llm_engine = AsyncLLM.from_engine_args(engine_args)

        print("✅ LLM Engine loaded successfully.")

    except Exception as e:
        print(f"❌ Failed to load LLM Engine: {e}")

    
def get_llm_engine() -> AsyncLLM:
    """초기화된 LLM 엔진 객체를 반환"""
    global llm_engine

    if llm_engine is None:
        raise RuntimeError("LLM Engine is not initialized.")
    return llm_engine

