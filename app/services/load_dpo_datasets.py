# DPO 학습용 데이터셋 불러오기
# 필요한 요소: prompt, chosen, rejected
from datasets import Dataset

SEED = 42

# 마스킹 처리된 데이터 가져오기
def process_training_data(tokenizer, datasets: list):
    dpo_pre_datasets = {}
    # datasets으로 로드할 수 있도록 전처리
    for dataset in datasets:
        message_uuid = dataset.get("messageUUID")
        if message_uuid not in dpo_pre_datasets:
            print("새로 생성")
            dpo_pre_datasets[message_uuid] = {}

        sender_type = dataset.get("sender_type")
        content = dataset.get("content")
        
        if sender_type == "USER":
            dpo_pre_datasets[message_uuid]["question"] = content
        if sender_type == "AI":
            dpo_pre_datasets[message_uuid]["chosen"] = content
        if sender_type == "TRAINING":
            dpo_pre_datasets[message_uuid]["rejected"] = content
    
    print("✅ 데이터 로드 완료")

    # 허깅페이스의 datasets 형태로 변환
    column_oriented_data = {
        "question": [],
        "chosen": [],
        "rejected": []
    }

    for uuid, data in dpo_pre_datasets.items():
        column_oriented_data["question"].append(data["question"])
        column_oriented_data["chosen"].append(data["chosen"])
        column_oriented_data["rejected"].append(data["rejected"])

    # 2. Dataset 객체 생성
    dpo_dataset = Dataset.from_dict(column_oriented_data)
    print("✅ 데이터 전처리 완료")

    # print(dpo_dataset)
    # print(dpo_dataset[0])

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 데이터셋 전처리를 위한 함수
    def return_prompt_and_responses(samples):

        prompt = f"{samples["question"]}"
        prompt = {"role": "user", "content": prompt},
        chosen_response = [
            {"role": "assistant", "content": samples["chosen"]}
        ]
        rejected_response = [
            {"role": "assistant", "content": samples["rejected"]}
        ]

        return {
            "prompt": tokenizer.apply_chat_template(prompt, tokenize=False),
            "chosen": "{}".format(tokenizer.apply_chat_template(chosen_response, tokenize=False).replace('<|begin_of_text|>', '')),
            "rejected": "{}".format(tokenizer.apply_chat_template(rejected_response, tokenize=False).replace('<|begin_of_text|>', ''))
        }

    # 가져온 데이터를 학습용으로 전처리하기
    original_columns = dpo_dataset.column_names

    print(dpo_dataset)

    dpo_dataset = dpo_dataset.map(
        return_prompt_and_responses,
        # batched=True,
        remove_columns=original_columns
    )

    final_dataset = dpo_dataset.train_test_split(test_size=0.1, seed=SEED)

    return final_dataset