from typing import Optional

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from transformers import AutoTokenizer

from app.core.config import base_settings

# 전역 변수 정의
MODEL_NAME = base_settings.base_model + "/qwen"
# MODEL_NAME = "../quantization/Midm-2.0-Mini-Instruct"
MODEL_NAME_1 = "./local-models/Qwen2.5-Coder-0.5B-Instruct"
MODEL_NAME_2 = "./local-models/Llama-3.2-1B-Instruct"
ROUTER_MODEL = "./local-models/router"


llm_tokenizer_1: Optional[AutoTokenizer] = None
llm_tokenizer_2: Optional[AutoTokenizer] = None
llm_engine_1: Optional[AsyncLLM] = None
llm_engine_2: Optional[AsyncLLM] = None
router_engine: Optional[AsyncLLM] = None


def load_tokenizers():
    """
    토크나이저 로드하기
    1: qwen, 2: llama
    """
    global llm_tokenizer_1, llm_tokenizer_2
    
    # 1. Qwen2.5-Coder 토크나이저 로드
    if llm_tokenizer_1 is None:
        try:
            llm_tokenizer_1 = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=MODEL_NAME_1,
            )
            print(f"✅ Tokenizer 1 {MODEL_NAME_1} loaded successfully.")

        except Exception as e:
            print(f"❌ Failed to load Tokenizer 1: {e}")
        
    # 2. Llama-Korean-3.1-8B-Instruct 토크나이저 로드
    if llm_tokenizer_2 is None:
        try:
            llm_tokenizer_2 = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=MODEL_NAME_2
            )
            print(f"✅ Tokenizer 2 {MODEL_NAME_2} loaded successfully.")

        except Exception as e:
            print(f"❌ Failed to load Tokenizer 2: {e}")


def get_tokenizer_1() -> AutoTokenizer:
    """초기화된 토크나이저 객체를 반환"""
    global llm_tokenizer_1

    if llm_tokenizer_1 is None:
        raise RuntimeError("Tokenizer 1 is not initialized.")
    
    return llm_tokenizer_1


def get_tokenizer_2() -> AutoTokenizer:
    """초기화된 토크나이저 객체를 반환"""
    global llm_tokenizer_2
    
    if llm_tokenizer_2 is None:
        raise RuntimeError("Tokenizer 2 is not initialized.")
    
    llm_tokenizer_2.pos_token = llm_tokenizer_2.eos_token
    llm_tokenizer_2.padding_side = "left"
    
    return llm_tokenizer_2


async def load_llm_engines():
    """vllm으로 llm 엔진 초기화 하기"""
    global llm_engine_1, llm_engine_2

    # 이미 있다면
    if llm_engine_1 is not None and llm_engine_2 is not None:
        print("Both LLM Engines already loaded.")
        return
    
    # 1. Qwen2.5-Coder 엔진 설정
    if llm_engine_1 is None:
        print(f"⏳ Starting LLM Engine 1 Load ({MODEL_NAME_1})...")

        try:
            engine_args_1 = AsyncEngineArgs(
                model=MODEL_NAME_1,
                enforce_eager=True,
                gpu_memory_utilization=0.35,
                quantization="bitsandbytes",
                max_model_len=8192,
                tensor_parallel_size=1
            )

            llm_engine_1 = AsyncLLM.from_engine_args(engine_args_1)

            print("✅ LLM Engine 1 loaded successfully.")

        except Exception as e:
            print(f"❌ Failed to load LLM Engine 1: {e}")

    # 2. Llama-Korean-3.1-8B-Instruct
    if llm_engine_2 is None:
        print(f"⏳ Starting LLM Engine 2 Load ({MODEL_NAME_2})...")

        try:
            engine_args_2 = AsyncEngineArgs(
                model=MODEL_NAME_2,
                enforce_eager=True,
                gpu_memory_utilization=0.5,
                quantization="bitsandbytes",
                max_model_len=8192,
                tensor_parallel_size=1
            )

            llm_engine_2 = AsyncLLM.from_engine_args(engine_args_2)

            print("✅ LLM Engine 2 loaded successfully.")

        except Exception as e:
            print(f"❌ Failed to load LLM Engine 2: {e}")

    
def get_llm_engine_1() -> AsyncLLM:
    """초기화된 LLM 엔진 객체를 반환"""
    global llm_engine_1

    if llm_engine_1 is None:
        raise RuntimeError("LLM Engine 1 is not initialized.")
    return llm_engine_1


def get_llm_engine_2() -> AsyncLLM:
    """초기화된 LLM 엔진 객체를 반환"""
    global llm_engine_2

    if llm_engine_2 is None:
        raise RuntimeError("LLM Engine 2 is not initialized.")
    return llm_engine_2


# async def load_router_engine():
#     """vllm으로 router 엔진 초기화"""
#     global router_engine

#     if router_engine is not None:
#         print("Router Engines already loaded.")
#         return
    
#     if router_engine is None:
#         print(f"⏳ Starting Router Engine Load ({ROUTER_MODEL})...")

#         try:
#             engine_args = AsyncEngineArgs(
#                 model=ROUTER_MODEL,
#                 enforce_eager=True,
#                 gpu_memory_utilization=0.1,
#                 quantization="bitsandbytes",
#                 max_model_len=8192,
#                 tensor_parallel_size=1
#             )

#             router_engine = AsyncLLM.from_engine_args(engine_args)

#             print("✅ Router Engine loaded successfully.")

#         except Exception as e:
#             print(f"❌ Failed to load Router Engine: {e}")