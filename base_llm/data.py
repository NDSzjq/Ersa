import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from datasets import Dataset
import base_llm.config as config


def get_tokenizer():
    """加载分词器"""
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    print("分词器加载完成！")
    return tokenizer


def load_dataset_by_name(dataset_name):
    """
    根据数据集名称加载数据
    支持: sst2, mnli, qnli
    优先从本地 parquet 文件加载
    """
    # ========== 1. 优先从本地加载 ==========
    # Resolve data relative to this project so the Windows and WSL copies do
    # not accidentally cross-load each other's absolute paths.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_data_dir = os.path.join(project_root, "data", dataset_name)
    train_path = os.path.join(local_data_dir, "train-00000-of-00001.parquet")
    valid_path = os.path.join(local_data_dir, "validation-00000-of-00001.parquet")
    
    if os.path.exists(train_path) and os.path.exists(valid_path):
        print(f"从本地加载数据集: {local_data_dir}")
        import pandas as pd
        
        train_df = pd.read_parquet(train_path)
        valid_df = pd.read_parquet(valid_path)
        
        # ===== 保留原始列，不做手动拼接 =====
        # 对于不同数据集，保留不同的列名
        if dataset_name == "sst2":
            # SST-2: 只有 sentence 列
            train_df = train_df[["sentence", "label"]]
            valid_df = valid_df[["sentence", "label"]]
            
        elif dataset_name == "qnli":
            # QNLI: 保留 question 和 sentence 两列
            # 确保列名正确
            if "question" in train_df.columns and "sentence" in train_df.columns:
                train_df = train_df[["question", "sentence", "label"]]
                valid_df = valid_df[["question", "sentence", "label"]]
            else:
                # 如果列名不对，尝试找文本列
                text_cols = [col for col in train_df.columns if col not in ["label", "idx", "id"]]
                if len(text_cols) >= 2:
                    train_df = train_df[text_cols[:2] + ["label"]]
                    valid_df = valid_df[text_cols[:2] + ["label"]]
                    train_df.columns = ["question", "sentence", "label"]
                    valid_df.columns = ["question", "sentence", "label"]
                    
        elif dataset_name == "mnli":
            # MNLI: 保留 premise 和 hypothesis 两列
            if "premise" in train_df.columns and "hypothesis" in train_df.columns:
                train_df = train_df[["premise", "hypothesis", "label"]]
                valid_df = valid_df[["premise", "hypothesis", "label"]]
            else:
                # 尝试找文本列
                text_cols = [col for col in train_df.columns if col not in ["label", "idx", "id"]]
                if len(text_cols) >= 2:
                    train_df = train_df[text_cols[:2] + ["label"]]
                    valid_df = valid_df[text_cols[:2] + ["label"]]
                    train_df.columns = ["premise", "hypothesis", "label"]
                    valid_df.columns = ["premise", "hypothesis", "label"]
        else:
            # 其他数据集：找第一列文本列作为 sentence
            for col in train_df.columns:
                if col not in ["label", "idx", "id"]:
                    train_df = train_df[[col, "label"]]
                    valid_df = valid_df[[col, "label"]]
                    train_df.columns = ["sentence", "label"]
                    valid_df.columns = ["sentence", "label"]
                    break
        
        # 确保 label 是整数
        train_df["label"] = train_df["label"].astype(int)
        valid_df["label"] = valid_df["label"].astype(int)
        
        train_ds = Dataset.from_pandas(train_df)
        valid_ds = Dataset.from_pandas(valid_df)
        
        print(f"训练集: {len(train_ds)} 条, 验证集: {len(valid_ds)} 条")
        print(f"列名: {train_ds.column_names}")
        return train_ds, valid_ds
    
    # ========== 2. 如果本地没有，从 Hugging Face 加载 ==========
    print(f"本地文件不存在，从 Hugging Face 加载...")
    return _load_from_huggingface(dataset_name)


def _load_from_huggingface(dataset_name):
    """从 HuggingFace 加载数据集"""
    from datasets import load_dataset
    
    dataset_map = {
        "sst2": ("glue", "sst2"),
        "mnli": ("glue", "mnli"),
        "qnli": ("glue", "qnli")
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"不支持的数据集: {dataset_name}，支持: {list(dataset_map.keys())}")
    
    path, name = dataset_map[dataset_name]
    print(f"加载数据集: {path}/{name}")
    
    dataset = load_dataset(path, name)
    
    if dataset_name == "sst2":
        train_ds = dataset["train"].rename_column("sentence", "sentence")
        valid_ds = dataset["validation"].rename_column("label", "label")
        return train_ds, valid_ds
    
    elif dataset_name == "mnli":
        # 使用 matched validation
        train_ds = dataset["train"]
        valid_ds = dataset["validation_matched"]
        # 保留 premise 和 hypothesis 列
        return train_ds, valid_ds
    
    elif dataset_name == "qnli":
        train_ds = dataset["train"]
        valid_ds = dataset["validation"]
        # 保留 question 和 sentence 列
        return train_ds, valid_ds


def preprocess(examples, tokenizer, dataset_name="sst2"):
    """
    分词预处理
    支持单句（sst2）和句子对（qnli, mnli）
    """
    if dataset_name == "sst2":
        # SST-2: 单句分类
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LEN
        )
    
    elif dataset_name == "qnli":
        # QNLI: question + sentence 句子对
        return tokenizer(
            examples["question"],
            examples["sentence"],  # text_pair
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LEN
        )
    
    elif dataset_name == "mnli":
        # MNLI: premise + hypothesis 句子对
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],  # text_pair
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LEN
        )
    
    else:
        # 默认：单句处理
        if "sentence" in examples:
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=config.MAX_LEN
            )
        else:
            # 尝试找第一列文本列
            for key in examples.keys():
                if key not in ["label", "idx", "id"]:
                    return tokenizer(
                        examples[key],
                        padding="max_length",
                        truncation=True,
                        max_length=config.MAX_LEN
                    )
            raise ValueError(f"无法找到文本列，可用列: {examples.keys()}")


def partition_non_iid(dataset, num_clients, alpha=0.5, seed=42):
    """Non-IID 划分"""
    np.random.seed(seed)
    labels = np.array(dataset["label"])
    from collections import defaultdict
    label_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_indices[lbl].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for lbl, indices in label_indices.items():
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        assigned = 0
        for c in range(num_clients):
            take = int(proportions[c] * len(indices))
            if take == 0 and assigned < len(indices):
                take = 1
            assigned += take
            if take > 0:
                chosen = np.random.choice(indices, take, replace=False)
                client_indices[c].extend(chosen)
                indices = list(set(indices) - set(chosen))
        while indices:
            for c in range(num_clients):
                if indices and len(client_indices[c]) < len(dataset) // num_clients * 2:
                    idx = indices.pop()
                    client_indices[c].append(idx)
    return client_indices


def get_dataloaders(dataset_name=None):
    """
    返回联邦训练所需的所有数据组件
    返回字典，方便扩展
    """
    if dataset_name is None:
        dataset_name = getattr(config, "DATASET_NAME", "sst2")
    
    tokenizer = get_tokenizer()
    raw_train_ds, raw_valid_ds = load_dataset_by_name(dataset_name)

    print("开始分词...")
    train_ds = raw_train_ds.map(
        lambda x: preprocess(x, tokenizer, dataset_name=dataset_name),
        batched=True
    )
    valid_ds = raw_valid_ds.map(
        lambda x: preprocess(x, tokenizer, dataset_name=dataset_name),
        batched=True
    )

    # 设置格式
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # token_type_ids 会自动被 tokenizer 生成，但 Set 格式默认不包含
    # 对于句子对任务，需要保留 token_type_ids
    if dataset_name in ["qnli", "mnli"]:
        # 重新设置格式，包含 token_type_ids
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

    print("Non-IID 划分...")
    client_indices = partition_non_iid(
        train_ds,
        num_clients=config.NUM_CLIENTS,
        alpha=0.5
    )

    client_loaders = []
    for idx, idx_list in enumerate(client_indices):
        subset = Subset(train_ds, idx_list)
        loader = DataLoader(
            subset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            drop_last=False
        )
        client_loaders.append(loader)
        print(f"  客户端 {idx+1}: {len(idx_list)} 条")

    test_loader = DataLoader(
        valid_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    print(f"完成！训练集: {len(train_ds)}，验证集: {len(valid_ds)}")
    print(f"数据集类型: {dataset_name}")

    return {
        "client_loaders": client_loaders,
        "test_loader": test_loader,
        "raw_train": raw_train_ds,
        "raw_test": raw_valid_ds,
        "tokenizer": tokenizer,
        "dataset_name": dataset_name,
    }


# ========== 测试函数 ==========
if __name__ == "__main__":
    # 测试 QNLI 数据加载
    print("=" * 50)
    print("测试 QNLI 数据加载")
    print("=" * 50)
    
    dataloaders = get_dataloaders("qnli")
    test_loader = dataloaders["test_loader"]
    
    # 取一个 batch 验证
    batch = next(iter(test_loader))
    print("\n=== 验证 batch 数据 ===")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    if "token_type_ids" in batch:
        print(f"token_type_ids shape: {batch['token_type_ids'].shape}")
        print(f"token_type_ids[0][:30]: {batch['token_type_ids'][0][:30]}")
        # 检查是否有 0/1 分段
        tti = batch['token_type_ids'][0]
        if (tti[:50] == 0).all() and (tti[50:100] == 1).all():
            print("✅ token_type_ids 正确分段（前段为0，后段为1）")
        elif (tti == 0).all():
            print("❌ token_type_ids 全是0（问题未修复）")
        else:
            print(f"⚠️ token_type_ids 分布: min={tti.min()}, max={tti.max()}")
    print(f"labels[:10]: {batch['labels'][:10]}")
