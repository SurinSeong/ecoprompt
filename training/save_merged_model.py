import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_dir = "./local-models/Llama-3.1-Korean-8B-Instruct"
# lora_dir = "./local-models/train/llama-lora-adapter"
lora_dir = "./trainer_output/checkpoint-500"    # 학습이 중간에 끊겨 체크포인트를 통해 저장한다.
out_dir = "./local-models/Llama-SSAFY-8B/test"

dtype = torch.bfloat16
base = AutoModelForCausalLM.from_pretrained(
    base_dir, torch_dtype=dtype, device_map="auto", trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True, trust_remote_code=True)

# LoRA 로드 후 병합
model = PeftModel.from_pretrained(base, lora_dir)
merged = model.merge_and_unload()

# 저장
merged.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
print(f"병합 모델 저장: {out_dir}")