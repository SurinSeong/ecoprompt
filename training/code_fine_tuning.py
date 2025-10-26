import random

SEED = 42

# 학습 데이터셋 로드하기
from datasets import load_dataset, concatenate_datasets

data_path = "./data/code_data"

alpaca_dataset = load_dataset("json", data_files=data_path + "/alpaca_total.jsonl")
searchnet_dataset = load_dataset("json", data_files=data_path + "/code_search_net_total.jsonl")

# code search net 데이터셋 줄이기
num_samples = 20000
indices = random.sample(range(len(searchnet_dataset["train"])), num_samples)
searchnet_dataset = searchnet_dataset["train"].select(indices)

# train / valid로 분리하기
alpaca_dataset = alpaca_dataset["train"].train_test_split(test_size=0.2, seed=SEED)
alpaca_train = alpaca_dataset["train"]
alpaca_valid = alpaca_dataset["test"]

searchnet_dataset = searchnet_dataset.train_test_split(test_size=0.2, seed=SEED)
searchnet_train = searchnet_dataset["train"]
searchnet_valid = searchnet_dataset["test"]

print("Alpaca dataset split:")
print(f"  Train size: {len(alpaca_train)}")
print(f"  Valid size: {len(alpaca_valid)}")

print("\nCode Search Net dataset split:")
print(f"  Train size: {len(searchnet_train)}")
print(f"  Valid size: {len(searchnet_valid)}")

# 두 종류의 데이터 합치기
combined_train_dataset = concatenate_datasets([alpaca_train, searchnet_train])
combined_valid_dataset = concatenate_datasets([alpaca_valid, searchnet_valid])

print("\nCombined dataset split:")
print("   Train size:", len(combined_train_dataset))
print("   Valid size:", len(combined_valid_dataset))

print("[COMPLETED] 학습 데이터셋 로드 완료")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# =========
# 모델 로드
# =========

BASE_MODEL = "./models/Llama-3.1-Korean-8B-Instruct"

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,    # BF16
    device_map="auto"      # GPU 자동 분배
)

# Llama 3.1 chat template 적용하기 => mLA 파일 확인하기
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>
{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{OUTPUT}<|eot_id|>"""

def formatting_prompts(examples):
    texts = []

    for messages in examples["messages"]:

    #     prompt = tokenizer.apply_chat_template(
    #         messages[:2], tokenizer=False, add_generation_prompt=True
    #     )
    #     response = messages[-1]["content"].strip() + tokenizer.eos_token
    #     texts.append(prompt + response)

    # return {"texts": texts}
        prompt = chat_template.format(
            SYSTEM = messages[0]["content"],
            INPUT = messages[1]["content"],
            OUTPUT = messages[2]["content"]
        )
        
        texts.append(prompt)
    
    return {"text": texts}

# # TRL이 기대하는 키로 매핑하기
# def to_trl_schema(dataset):
#     return dataset.map(formatting_prompts, batched=True, remove_columns=dataset.column_names)

train_dataset = combined_train_dataset.map(formatting_prompts, batched=True)
valid_dataset = combined_valid_dataset.map(formatting_prompts, batched=True)

train_data = train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
valid_data = valid_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)



# LoRA 설정
from peft import LoraConfig

print("LoRA 설정")
lora_config = LoraConfig(
    r=16,    # LoRA 차원
    lora_alpha=32,    # LoRA Scaling Factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],    # LoRA 적용 대상 모듈
    lora_dropout=0.05,   # 드롭아웃 비율
    bias="none",     # LoRA에서 bias 사용 여부
    task_type="CASUAL_LM"    # LLM 파인튜닝을 위한 Casual Language Model 설정
)

# MAX SEQ LENGTH
MAX_SEQ_LENGTH = 4096
if hasattr(tokenizer, "model_max_length"):
    MAX_SEQ_LENGTH = min(MAX_SEQ_LENGTH, tokenizer.model_max_length)

# 학습 Argument 설정
print("Training Argument 설정")

# 결과 저장 폴더
import os

output_dir = "./models/Llama-3.1-Korean-8B-Instruct-LoRA"
lora_adapter_dir = f"{output_dir}/lora-adapter"
os.makedirs(output_dir, exist_ok=True)

train_args = TrainingArguments(
    per_device_train_batch_size=8,    # 배치 크기 (GPU 당 샘플 개수)
    gradient_accumulation_steps=4,    # 메모리 최적화 gradient accumulation 누적 스텝
    gradient_checkpointing=True,    # 활성화하면, GPU 메모리 사용감소 가능, 수행시간은 더 걸린다.
    num_train_epochs=5,    # 전체 데이터셋을 몇 번 반복해서 학습할 것인가
    warmup_steps=30,    # 학습률을 서서히 증가시키는 단계 (0 ~ 100)
    max_steps=300,    # 최대 학습 스텝
    learning_rate=1e-4,    # 학습률
    lr_scheduler_type="linear",    # 학습률 스케쥴러
    weight_decay=0.01,
    bf16=True,
    warmup_ratio=0.1,
    optim="adamw_torch",
    seed=SEED,
    output_dir="",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    report_to="wandb"
)


# Trainer Setup
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=valid_data,
    args=train_args,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False
)

model.config.use_cache = False

# 학습 시작
print("Start Training..")
trainer.train()

model.eval()    # 모델의 가중치는 변경하지 않고, forward 연산만 수행한다.
model.config.use_cache = True    # 이전 계산 결과를 저장하고 사용한다. => 추론속도 빨라짐, 메모리 사용 증가

# LoRA 어댑터/토크나이저 저장
print("LoRA 어댑터 저장")
os.makedirs(lora_adapter_dir, exist_ok=True)
trainer.save_model(lora_adapter_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA 어댑터가 '{lora_adapter_dir}'에 저장되었습니다.")
print(f"토크나이저가  '{output_dir}'에 저장되었습니다.")