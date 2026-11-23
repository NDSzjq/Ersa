"""Experiment 4 client: local SST-2 training and LoRA parameter export.

This module is deliberately independent from the legacy experiment clients.  It
keeps the experiment-four rank/model settings local to the process and returns
only CPU tensors that can be supplied to CrypTen as party-owned inputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

import base_llm.config as config
from base_llm.data import load_dataset_by_name, partition_non_iid, preprocess
from base_llm.model import Net


@dataclass
class Exp4Settings:
    dataset: str = "sst2"
    model: str = "bert-base"
    rank: int = 4
    num_clients: int = 2
    batch_size: int = 16
    max_train_samples: int = 64
    local_epochs: int = 1
    seed: int = 2026


def model_path(model_name: str) -> str:
    names = {
        "bert-base": "bert-base-uncased",
        "bert-large": "bert-large-uncased",
    }
    try:
        directory = names[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported experiment-four model: {model_name}") from exc
    return os.path.join(config.BASE_DIR, "models", directory)


class Exp4Client:
    def __init__(self, rank: int, settings: Exp4Settings):
        if settings.dataset not in {"sst2", "mnli", "qnli"}:
            raise ValueError("Experiment-four supports sst2, mnli, and qnli")
        if settings.num_clients != 2:
            raise ValueError("The experiment-four CrypTen smoke test requires 2 clients")

        self.rank = rank
        self.settings = settings
        self.device = torch.device(
            "cuda", rank % max(torch.cuda.device_count(), 1)
        ) if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        # Net reads these values during construction.  They are process-local and
        # do not affect experiments one through three.
        config.DATASET_NAME = settings.dataset
        config.MODEL_NAME = settings.model
        config.MODEL_PATH = model_path(settings.model)
        config.LORA_R = settings.rank
        config.BATCH_SIZE = settings.batch_size
        config.LOCAL_EPOCHS = settings.local_epochs
        config.MAX_LEN = 64

        # Federated clients must begin from the same global model.  The prior
        # rank-dependent model seed gave each party a different LoRA
        # initialization, which makes a one-round aggregation meaningless.
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)
        self.model = Net().to(self.device)
        self.loader_generator = torch.Generator().manual_seed(settings.seed + rank)
        self.train_loader = self._build_loader()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _build_loader(self) -> DataLoader:
        raw, _ = load_dataset_by_name(self.settings.dataset)
        limit = min(len(raw), self.settings.max_train_samples)
        raw = raw.select(range(limit))
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_PATH, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        tokenized = raw.map(
            lambda batch: preprocess(
                batch, tokenizer, dataset_name=self.settings.dataset
            ),
            batched=True,
        )
        columns = ["input_ids", "attention_mask", "label"]
        if self.settings.dataset in {"mnli", "qnli"}:
            columns.insert(2, "token_type_ids")
        tokenized.set_format(type="torch", columns=columns)
        indices = partition_non_iid(
            tokenized,
            num_clients=self.settings.num_clients,
            alpha=0.5,
            seed=self.settings.seed,
        )[self.rank]
        if not indices:
            raise RuntimeError(f"Client {self.rank} received no samples")
        return DataLoader(
            Subset(tokenized, indices),
            batch_size=self.settings.batch_size,
            shuffle=True,
            drop_last=False,
            generator=self.loader_generator,
        )

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for _ in range(self.settings.local_epochs):
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
        return total_loss / max(batches, 1)

    def export_lora(self) -> dict[str, torch.Tensor]:
        result = {}
        for name, parameter in self.model.base.named_parameters():
            if "lora_" in name:
                result[name] = parameter.detach().cpu().float().clone()
        if not result:
            raise RuntimeError("No LoRA parameters were found")
        return result


