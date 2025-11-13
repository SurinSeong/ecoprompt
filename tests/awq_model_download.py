import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams

model_name = "LGAI-EXAONE/EXAONE-4.0-32B-AWQ"

llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    max_model_len=2000
)

sampling_params = SamplingParams(
    temperature=0.5,
    top_p=0.7,
    repetition_penalty=1.1,
    max_tokens=1024
)

query = "해리포터 줄거리를 한글로 간략히 설명해주세요."
