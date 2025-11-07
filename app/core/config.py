import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

SEED = 42

class TotalSettings(BaseSettings):
    # DPO 파인튜닝한 모델 저장 폴더
    base_model: str = "./local-models/Llama-SSAFY-8B"
    # MongoDB 관련
    mongo_url: str = os.getenv("MONGO_URL")


# wandb 관련
class WandbSettings(BaseSettings):
    project: str = "ecoprompt"
    entity: str = "surinseong-ai"
    

# 재학습 관련
class PostTrainSettings(BaseSettings):
    # dpo training arguments
    output_dir: str = "./local-models/dpo_train"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    optimizer: str = "adamw_torch"
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    loss_type: str = "sigmoid"
    beta: float = 0.1
    

# 평가 관련
class EvaluateSettings(BaseSettings):
    eval_model: str = "hf"
    eval_tasks: list = ["haerae", "humaneval"]    # 직접 만든 태스크 넣어도 좋음. => ex) SSAFY 관련..
    eval_batch_size: int = 4
    eval_device: str = "cuda:1"
    eval_seed: int = SEED
    eval_num_fewshot: int = 0    # 기본값: 0
    eval_output_path: str = "./eval_output"
    eval_log_samples: bool = True
    eval_auto_after_train: bool = True



base_settings = TotalSettings()
wandb_settings = WandbSettings()
train_settings = PostTrainSettings()
evaluate_settings = EvaluateSettings()