from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

SEED = 42

class TotalSettings(BaseSettings):
    # DPO 파인튜닝한 모델 저장 폴더
    model_path: str = "./models/Llama-SSAFY-8B"

# 학습 관련
class TrainSettings(BaseSettings):
    output_dir: str = ""
    learning_rate: float = 1e-6
    optimizer: str = "adamw"
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    dpo_beta: float = 0.25
    


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
train_settings = TrainSettings()
evaluate_settings = EvaluateSettings()