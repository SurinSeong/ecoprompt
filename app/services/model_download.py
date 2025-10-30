from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

def download_and_save(model_id: str, save_dir: str):
    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    device_map = "auto" if has_cuda else None

    print(f"\n[다운로드 시작] {model_id}")
    print(f" - GPU 사용: {has_cuda}")
    print(f" - 저장 경로: {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    print(f"[완료] 모델과 토크나이저가 '{save_dir}'에 저장되었습니다.")


# ===== Llama-3.1-Korean-8B-Instruct =====
download_and_save(
    model_id="sh2orc/Llama-3.1-Korean-8B-Instruct",
    save_dir="./local-models/Llama-3.1-Korean-8B-Instruct"
)

# # ===== Midm-2.0-Base-Instruct =====
# download_and_save(
#     model_id="K-intelligence/Midm-2.0-Base-Instruct",
#     save_dir="./Midm-2.0-Base-Instruct"
# )