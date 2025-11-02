import os
import wandb
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

from load_training_datasets import load_sft_datasets

# GPU 메모리 관리
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42

# ===============
# 학습 데이터 로드
# ===============

# Llama 3.1 chat template 적용하기
def format_example(row):
    return {
        'text': f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
        당신은 친절하고 똑똑한 AI assistant 입니다. 사용자의 질문에 알맞은 답변을 반환해주세요.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
 
        {row['instruction']}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
 
        {row['output']}<|eot_id|"""
    }

# 데이터셋 로드
train_dataset, valid_dataset = load_sft_datasets()

# 포맷팅 함수 적용
train_dataset = train_dataset.map(format_example)
valid_dataset = valid_dataset.map(format_example)

# =========
# 모델 로드
# =========

BASE_MODEL = "./local-models/Llama-3.1-Korean-8B-Instruct"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token    # 시퀸스 패딩에 eos 토큰 사용
tokenizer.padding_side = "right"    # 패딩을 오른쪽에 추가

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,    # BF16
    device_map="auto"      # GPU 자동 분배
)

# LoRA 설정
from peft import LoraConfig, get_peft_model

print("LoRA 설정")
lora_config = LoraConfig(
    r=8,    # LoRA 차원
    lora_alpha=16,    # LoRA Scaling Factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],    # LoRA 적용 대상 모듈
    lora_dropout=0.05,   # 드롭아웃 비율
    bias="none",     # LoRA에서 bias 사용 여부
    task_type="CAUSAL_LM"    # LLM 파인튜닝을 위한 Causal Language Model 설정 (작업 유형)
)

# PEFT 어댑터 설정을 모델에 적용한다.
model = get_peft_model(model, lora_config)

# MAX SEQ LENGTH
MAX_SEQ_LENGTH = 4096
if hasattr(tokenizer, "model_max_length"):
    MAX_SEQ_LENGTH = min(MAX_SEQ_LENGTH, tokenizer.model_max_length)

# 학습 Argument 설정
print("Training Argument 설정")

train_args = TrainingArguments(
    per_device_train_batch_size=8,    # 배치 크기 (GPU 당 샘플 개수)
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,    # 메모리 최적화 gradient accumulation 누적 스텝
    gradient_checkpointing=True,    # 활성화하면, GPU 메모리 사용감소 가능, 수행시간은 더 걸린다.
    num_train_epochs=3,    # 전체 데이터셋을 몇 번 반복해서 학습할 것인가
    # warmup_steps=100,    # 학습률을 서서히 증가시키는 단계 (0 ~ 100)
    max_steps=3000,    # 최대 학습 스텝 (-1: 조기종료 막기)
    learning_rate=2e-4,    # 학습률
    lr_scheduler_type="cosine",    # 학습률 스케쥴러
    weight_decay=0.01,
    bf16=True,
    warmup_ratio=0.1,
    optim="adamw_torch",
    seed=SEED,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    report_to="wandb",
)

# Trainer Setup
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # dataset_text_field="text",
    args=train_args,
    # max_seq_length=MAX_SEQ_LENGTH,
    # tokenizer=tokenizer
)

# ======
# 저장
# ======

lora_adapter_dir = "./local-models/train/llama-lora-adapter"
os.makedirs(lora_adapter_dir, exist_ok=True)

# 혹시 모를 캐시 관련 메모리 문제 예방
model.config.use_cache = False

# wandb 설정
wandb_config = {
    "model": BASE_MODEL.split("/")[-1],
    "learning_rate": 2e-4,
    "max_steps": 3000,
    "batch_size": 8,
    "lora_r": 8,
    "dataset": "OpenCoder-LLM/opc-sft-stage2"
}

wandb.init(
    project="ecoprompt",
    entity="surinseong-ai",
    name="code_sft_lora_v3",
    config=wandb_config,
    # resume=True    # 재시작
)

# ======
# 학습 시작
# ======

print("Start Training..")
trainer.train()

# 학습 재시작 하기
# print("Resume Training..")
# resume_checkpoint = "./trainer_output/checkpoint-2000"
# trainer.train(resume_from_checkpoint=resume_checkpoint)    # 

# model.eval()    # 모델의 가중치는 변경하지 않고, forward 연산만 수행한다.
# model.config.use_cache = True    # 이전 계산 결과를 저장하고 사용한다. => 추론속도 빨라짐, 메모리 사용 증가

wandb.finish()

# LoRA 어댑터 저장
print("LoRA 어댑터 저장")
trainer.save_model(lora_adapter_dir)    # LoRA 어댑터 가중치 저장
print(f"LoRA 어댑터가 '{lora_adapter_dir}'에 저장되었습니다.")