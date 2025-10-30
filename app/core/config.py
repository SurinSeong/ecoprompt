from pydantic_settings import BaseSettings
from typing import Optional

SEED = 42

class TotalSettings(BaseSettings):
    # DPO 파인튜닝한 모델 저장 폴더
    base_model: str = "./local-models/Llama-SSAFY-8B"

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

    class Config:
        env_file = ".env"    # 환경변수 파일 경로

base_settings = TotalSettings()
wandb_settings = WandbSettings()
train_settings = TrainSettings()
evaluate_settings = EvaluateSettings()