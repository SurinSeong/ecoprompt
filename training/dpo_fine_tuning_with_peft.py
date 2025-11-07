# QLora + DPO

import os
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer

from datasets import load_dataset

# GPU 메모리 관리
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42

MODEL_NAME = "../quantization/Midm-2.0-Mini-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

def return_prompt_and_responses(samples):
    prompt = f"{samples['instruction']}\n\n{samples['input']}"
    prompt = {"role": "user", "content": prompt},
    chosen_messages = [
        {"role": "assistant", "content": samples["accepted"]}
    ]
    rejected_messages = [
        {"role": "assistant", "content": samples["rejected"]}
    ]
    
    return {
        "prompt": tokenizer.apply_chat_template(prompt, tokenize=False),
        "chosen": "{}".format(tokenizer.apply_chat_template(chosen_messages, tokenize=False).replace('<|begin_of_text|>', '')),
        "rejected": "{}".format(tokenizer.apply_chat_template(rejected_messages, tokenize=False).replace('<|begin_of_text|>', '')),
    }

dpo_dataset = load_dataset("Vezora/Code-Preference-Pairs")
# print(dpo_dataset)
# print(dpo_dataset["train"])

dpo_dataset = dpo_dataset["train"].map(
    return_prompt_and_responses,
    # batched=True,
    remove_columns=dpo_dataset["train"].column_names
)

dpo_dataset = dpo_dataset.train_test_split(test_size=0.1, seed=SEED)


# 모델 학습
print("=== [Training Model] ===")

# GPU 메모리의 감소와 속도 향상을 위한 BitsAndBytesConfig, Flash Attention 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# flash attention 설정
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16

else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# model 로드
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    quantization_config=bnb_config
)

model.config.use_cache = False

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

# Training Arguments 설정
train_args = TrainingArguments(
    output_dir="./local-models/dpo/test",
    num_train_epochs=4,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    tf32=True,
    optim="adamw_torch",
    logging_steps=10,
    learning_rate=1e-5,
    max_grad_norm=0.3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    save_total_limit=3,
    save_steps=200,
    eval_strategy='steps',
    eval_steps=100,
    report_to="wandb",
)

# DPO 손실 타입 설정
dpo_args = {
    "beta": 0.1,
    "loss_type": "sigmoid"
}

# DPO trainer 설정
trainer = DPOTrainer(
    model,
    ref_model=None,
    peft_config=peft_config,
    args=train_args,   
    train_dataset=dpo_dataset["train"],
    eval_dataset=dpo_dataset["test"],
    processing_class=tokenizer,
    beta=0.1,
)

wandb_config = {
    "model": "Midm-Mini",
    "learning_rate": 1e-5,
    "epochs": 1,
    "batch_size": 1,
    "dataset": "userQA"
}

wandb.init(
    project="ecoprompt",
    entity="surinseong-ai",
    name="dpo_test_v2",
    config=wandb_config
)

print("[START] 모델 학습 시작")
trainer.train()

wandb.finish()