import asyncio
from vllm import LLM, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind

from app.core.config import base_settings

_sync_llm = None

def load_sync_llm():
    """sync llm 로드"""
    
    pass

# LLM 설정
llm = LLM(
    model="./llm-models/Llama-3.1-Korean-8B-Instruct",    # "sh2orc/Llama-3.1-Korean-8B-Instruct",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    trust_remote_code=True,
    dtype="auto",
    gpu_memory_utilization=base_settings.gpu_memory_utilization,
    max_model_len=base_settings.max_model_len,
)

# AsyncLLM 설정
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True,
)

async_llm = AsyncLLM.from_engine_args(engine_args)

sampling_params = SamplingParams(
    temperature=base_settings.temperature,    # 답변의 다양성
    top_p=base_settings.top_p,    # 핵심 단어 생성 기준
    max_tokens=base_settings.max_tokens,    # 제한 토큰 수
    seed=42,
    output_kind=RequestOutputKind.DELTA,
)