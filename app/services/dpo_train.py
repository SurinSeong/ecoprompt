from app.core.config import train_settings

import wandb
import torch
from transformers import AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

from app.core.config import wandb_settings, train_settings, base_settings

# 모델 DPO 학습하기
def train_model(dpo_dataset, tokenizer):

    # 학습할 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_settings.base_model + "/v_latest",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_settings.base_model + "/v_latest",
        device_map="auto",
        # load_in_8bit=True    # 8bit로 메모리 절감
    )    

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Training Arguments 설정
    dpo_args = DPOConfig(
        output_dir=train_settings.output_dir,    # 파인튜닝 모델 저장할 경로
        num_train_epochs=train_settings.num_train_epochs,     # 학습 에포크
        per_device_train_batch_size=train_settings.per_device_train_batch_size,     # 학습 배치 사이즈
        gradient_accumulation_steps=train_settings.gradient_accumulation_steps,     # 그라디언트 스텝
        gradient_checkpointing=train_settings.gradient_checkpointing,
        bf16=True,
        optim=train_settings.optimizer,     # 옵티마이저
        logging_steps=10,       # 로깅 스텝
        learning_rate=train_settings.learning_rate,     # 학습률
        warmup_ratio=train_settings.warmup_ratio,       # 웜업 비율
        lr_scheduler_type=train_settings.lr_scheduler_type,     # 학습률 스케줄러 타입
        loss_type=train_settings.loss_type,     # 손실 타입
        beta=train_settings.beta,       # DPO Loss의 온도, 일반적 0.1 ~ 0.5 => beta가 작을수록 레퍼런스 모델을 무시한다.
        save_total_limit=3,
        save_steps=200,
        remove_unused_columns=False,
        eval_strategy='steps',
        eval_steps=100,
        report_to="wandb"
    )

    # DPO trainer 설정
    trainer = DPOTrainer(
        model,
        model_ref,
        args=dpo_args,    
        train_dataset=dpo_dataset["train"],
        eval_dataset=dpo_dataset["test"],
        processing_class=tokenizer,
    )

    wandb_config = {
        "model": base_settings.base_model.split("/")[-1],
        "learning_rate": train_settings.learning_rate,
        "epochs": train_settings.num_train_epochs,
        "batch_size": train_settings.per_device_train_batch_size,
        "dataset": "User-QA"
    }

    wandb.init(
        project=wandb_settings.project,
        entity=wandb_settings.entity,
        name=f"dpo_train",
        config=wandb_config
    )

    print("[START] 모델 학습 시작")
    trainer.train()

    # 최종 모델 저장
    trainer.save_model(base_settings.base_model + "/dpo_model")

    wandb.finish()
