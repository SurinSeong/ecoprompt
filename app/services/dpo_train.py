from app.core.config import train_settings

import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

from app.core.config import wandb_settings, train_settings, base_settings

# 모델 DPO 학습하기
def train_model(model_path: str, dpo_dataset):

    # 학습할 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)    # 8bit로 메모리 절감

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Training Arguments 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=train_settings.per_device_train_batch_size,
        gradient_accumulation_steps=train_settings.gradient_accumulation_steps,
        gradient_checkpointing=train_settings.gradient_checkpointing,
        bf16=True,
        optim=train_settings.optimizer,
        logging_steps=10,
        learning_rate=train_settings.learning_rate,
        num_train_epochs=1,
        warmup_ratio=train_settings.warmup_ratio,
        lr_scheduler_type=train_settings.lr_scheduler_type,
        output_dir="./dpo_output",
        save_total_limit=3,
        save_steps=100,
        remove_unused_columns=False,
        eval_strategy='steps',
        eval_steps=50,
        report_to="wandb"
    )

    # DPO trainer 설정
    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=train_settings.dpo_beta,    # DPO Loss의 온도, 일반적 0.1 ~ 0.5 => beta가 작을수록 레퍼런스 모델을 무시한다.
        train_dataset=dpo_dataset["train"],
        eval_dataset=dpo_dataset["test"],
        processing_class=tokenizer,
        max_prompt_length=train_settings.max_prompt_length,
        max_length=train_settings.max_length
    )

    wandb_config = {
        "model": base_settings.model_path.split("/")[-1],
        "learning_rate": train_settings.learning_rate,
        "epochs": 5,
        "batch_size": train_settings.per_device_train_batch_size,
        "dataset": "user-QA"
    }

    wandb.init(
        project=wandb_settings.project,
        entity=wandb_settings.entity,
        name=f"dpo_self_train",
        config=wandb_config
    )

    print("[START] 모델 학습 시작")
    trainer.train()

    # 모델 저장
    trainer.save_model(base_settings.base_model + "/dpo_model")

    wandb.finish()
