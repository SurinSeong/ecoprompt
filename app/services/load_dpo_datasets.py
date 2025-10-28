# DPO 학습용 데이터셋 불러오기
# 필요한 요소: prompt, chosen, rejected
from datasets import load_dataset
from app.core.config import SEED

# 데이터셋 전처리
def return_prompt_and_responses(samples):
    return {
        "prompt": [
            f"### Input: ```{input}```\n ### Output: "
            for input in samples["input"]
        ],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }

# MongoDB에서 마스킹 처리된 데이터 가져오기
def get_masking_data(first_id: int):
    # MongoDB에서 데이터 가져오기


    # dataset으로 로드할 수 있도록 전처리


    # dataset으로 로드하기
    dataset = load_dataset()

    # 가져온 데이터를 학습용으로 전처리하기
    original_columns = dataset.columns_names

    dpo_dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns
    )

    final_dataset = dpo_dataset["train"].train_test_split(test_size=0.1, seed=SEED)

    return final_dataset



