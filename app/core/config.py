from pydantic_settings import BaseSettings

SEED = 42

class TotalSettings(BaseSettings):
    # DPO 파인튜닝한 모델 저장 폴더
    base_model: str = "./llm-models/Llama-SSAFY-8B"
    # 모델 엔진 실행 관련 파라미터
    gpu_memory_utilization: float = 0.82
    tensor_parallel_size: int = 2
    max_model_len: int = 4096
    max_num_seqs: int = 128
    # sampling params
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 2048


# wandb 관련
class WandbSettings(BaseSettings):
    project: str = "ecoprompt"
    entity: str = "surinseong-ai"
    

# 학습 관련
class TrainSettings(BaseSettings):
    # training arguments
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    learning_rate: float = 5e-6
    optimizer: str = "adamw_torch"
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"

    # DPO
    dpo_beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 2048
    

# 평가 관련
class EvaluateSettings(BaseSettings):
    eval_model: str = "hf"
    eval_tasks: list = ["haerae"]    # 직접 만든 태스크 넣어도 좋음. => ex) SSAFY 관련..
    eval_batch_size: int = 4
    eval_device: str = "cuda:1"
    eval_seed: int = SEED
    eval_num_fewshot: int = 0    # 기본값: 0
    eval_output_path: str = "./output/eval"
    eval_log_samples: bool = True
    eval_auto_after_train: bool = True
    

base_settings = TotalSettings()
wandb_settings = WandbSettings()
train_settings = TrainSettings()
evaluate_settings = EvaluateSettings()