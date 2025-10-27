from app.core.config import train_settings
from app.services.load_dpo_datasets import final_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

BASE_MODEL = "./models/Llama-SSAFY-8B"

# 모델 DPO 학습하기
def train_model(model_path: str):

    # 학습할 모델 로드
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Training Arguments 설정
    training_args = TrainingArguments(
        output_dir=train_settings.output_dir,
        evaluation_strategy="steps",
        do_eval=True,
        optim="adamw",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=2,
        logging_steps=100,
        learning_rate=1e-6,
        eval_steps=100,
        num_train_epochs=1,
        save_steps=500,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine"
    )

    # DPO trainer 설정
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.25,    # DPO Loss의 온도, 일반적 0.1 ~ 0.5 => beta가 작을수록 레퍼런스 모델을 무시한다.
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048
    )

    history = trainer.train()


    return history