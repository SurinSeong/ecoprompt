# 학습 데이터셋 로드하기
from datasets import load_dataset, concatenate_datasets

data_path = "./data/code_data"

alpaca_dataset = load_dataset("json", data_files=data_path + "/alpaca_total.jsonl")
searchnet_dataset = load_dataset("json", data_files=data_path + "/code_search_net_total.jsonl")

# code search net 데이터셋 줄이기
num_samples = 30000 # total dataset의 데이터개수를 5만개로 설정하기 위해
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