from datasets import load_dataset, concatenate_datasets

SEED = 42

def load_sft_datasets():
    educational_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train[:10000]")
    educational_instruct = educational_instruct.select(columns=["instruction", "output"])

    evol_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "evol_instruct", split="train[:15000]")
    evol_instruct = evol_instruct.select(columns=["instruction", "output"])
    
    mceval_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "mceval_instruct", split="train[:15000]")
    mceval_instruct = mceval_instruct.select(columns=["instruction", "output"])
    
    package_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "package_instruct", split="train[:10000]")
    package_instruct = package_instruct.select(columns=["instruction", "output"])

    # train / valid 분리
    educational_instruct = educational_instruct["train"].train_test_split(test_size=0.1, seed=SEED)
    educational_instruct_train, educational_instruct_test = educational_instruct["train"], educational_instruct["test"]

    evol_instruct = evol_instruct["train"].train_test_split(test_size=0.1, seed=SEED)
    evol_instruct_train, evol_instruct_test = evol_instruct["train"], educational_instruct["test"]

    mceval_instruct = mceval_instruct["train"].train_test_split(test_size=0.1, seed=SEED)
    mceval_instruct_train, mceval_instruct_test = mceval_instruct["train"], mceval_instruct["test"]
    
    package_instruct = package_instruct["train"].train_test_split(test_size=0.1, seed=SEED)
    package_instruct_train, package_instruct_test = package_instruct["train"], package_instruct["test"]

    train_dataset = concatenate_datasets([educational_instruct_train, evol_instruct_train, mceval_instruct_train, package_instruct_train])
    valid_dataset = concatenate_datasets([educational_instruct_test, evol_instruct_test, mceval_instruct_test, package_instruct_test])

    print("Total Dataset:")
    print("     Train size:", len(train_dataset))
    print("     Valid size:", len(valid_dataset))

    return train_dataset, valid_dataset