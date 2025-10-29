from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from transformers import AutoTokenizer

from app.core.config import base_settings

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(
    "sh2orc/Llama-3.1-Korean-8B-Instruct",
    trust_remote_code=True
)

# LLM 엔진 설정
engine_args = AsyncEngineArgs(
    model="./llm-models/Llama-3.1-Korean-8B-Instruct",    # "sh2orc/Llama-3.1-Korean-8B-Instruct",
    gpu_memory_utilization=base_settings.gpu_memory_utilization,
    tensor_parallel_size=base_settings.tensor_parallel_size,
    max_model_len=base_settings.max_model_len,
    max_num_seqs=base_settings.max_num_seqs
)

llm = AsyncLLMEngine.from_engine_args(engine_args)

sampling_params = SamplingParams(
    temperature=base_settings.temperature,    # 답변의 다양성
    top_p=base_settings.top_p,    # 핵심 단어 생성 기준
    max_tokens=base_settings.max_tokens,    # 제한 토큰 수
    stop=["<eot_id>"] 
)