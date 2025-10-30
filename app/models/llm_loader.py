from typing import Optional

from vllm import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind

from transformers import AutoTokenizer

from app.core.config import base_settings

# 전역 변수 정의
llm_engine: Optional[AsyncLLM] = None
llm_tokenizer: Optional[AutoTokenizer] = None

async def load_llm_engine():
    """vLLM 엔진을 초기화하고 전역 변수에 할당"""
    global llm_engine

    # 이미 있다면
    if llm_engine is not None:
        print("LLM Engine already loaded.")
        return

    # 없다면
    print("⏳ Starting LLM Engine Load from loader...")
    try:
        # AsyncLLM 설정
        engine_args = AsyncEngineArgs(
            model="./local-models/Llama-3.1-Korean-8B-Instruct",
            enforce_eager=True,
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
            "meta-llama/Llama-3.1-8B-Instruct",
            trust_remote_code=True
        )
        print("✅ LLM Engine loaded successfully.")

    except Exception as e:
        print(f"❌ Failed to load Tokenizer: {e}")


def get_tokenizer() -> AutoTokenizer:
    """초기화된 토크나이저 객체를 반환"""
    global llm_tokenizer

    if llm_tokenizer is None:
        raise RuntimeError("Tokenizer is not initialized.")
    return llm_tokenizer
            


def load_sampling_params():
    """Sampling Params 로드"""
    sampling_params = SamplingParams(
        temperature=base_settings.temperature,    # 답변의 다양성
        top_p=base_settings.top_p,    # 핵심 단어 생성 기준
        max_tokens=base_settings.max_tokens,    # 제한 토큰 수
        seed=42,
        output_kind=RequestOutputKind.DELTA,
        n=2
    )
    return sampling_params
