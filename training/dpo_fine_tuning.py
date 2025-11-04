import os
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

from datasets import load_dataset

# GPU 메모리 관리
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="./local-models/Llama-SSAFY-8B/test",
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="./local-models/Llama-SSAFY-8B/test",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
)

model.config.use_cache = False

model_ref = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="./local-models/Llama-SSAFY-8B/test",
    device_map="auto",
    # load_in_8bit=True
)

# Training Arguments 설정
dpo_args = DPOConfig(
    output_dir="./local-models/dpo/test",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_torch",
    logging_steps=10,
    learning_rate=1e-5,
    max_grad_norm=0.3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    save_total_limit=3,
    save_steps=200,
    remove_unused_columns=False,
    eval_strategy='steps',
    eval_steps=100,
    report_to="wandb",
    loss_type="sigmoid",
    beta=0.1    # DPO Loss의 온도, 일반적 0.1 ~ 0.5 => beta가 작을수록 레퍼런스 모델을 무시한다.
)

# DPO trainer 설정
trainer = DPOTrainer(
    model,
    model_ref,
    args=dpo_args,   
    train_dataset=dpo_dataset["train"],
    eval_dataset=dpo_dataset["test"],
    processing_class=tokenizer,
    # max_prompt_length=512,
    # max_length=2048
)

wandb_config = {
    "model": "Llama-SSAFY-8B",
    "learning_rate": 1e-5,
    "epochs": 1,
    "batch_size": 1,
    "dataset": "userQA"
}

wandb.init(
    project="ecoprompt",
    entity="surinseong-ai",
    name="dpo_test",
    config=wandb_config
)

print("[START] 모델 학습 시작")
trainer.train()

wandb.finish()