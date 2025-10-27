# DPO 학습용 데이터셋 불러오기
# 필요한 요소: prompt, chosen, rejected
def generate_prompt(example):
    # 챗 template에 맞게 변형
    return example

dpo_dataset = dataset.map(generate_prompt)

final_dataset = dpo_dataset["train"].train_test_split(test_size=0.1)
