"""Isolated malicious and benign clients for Experiment 3."""

import torch
import torch.nn as nn
import hashlib

import base_llm.config as config
from base_llm.model import Net


class Exp3Client:
    def __init__(self, train_loader, rank=0, attack="none", num_labels=2, noise_scale=0.05, defer_model=False, max_batches=None, random_seed=42):
        self.rank = rank
        self.attack = attack
        self.num_labels = int(num_labels)
        self.noise_scale = float(noise_scale)
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self._round_base_lora = None
        self._defer_model = bool(defer_model)
        self.max_batches = max_batches
        self.last_noise_norms = {}
        self.random_seed = int(random_seed)
        self.attack_round = None
        if not self._defer_model:
            self._build_model()

    def _build_model(self):
        if self.model is not None:
            return
        self.model = Net().to(self.device)
        lora_parameters = [
            parameter for name, parameter in self.model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": lora_parameters, "lr": config.LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY},
                {"params": self.model.classifier.parameters(), "lr": config.LEARNING_RATE, "weight_decay": 0.0},
            ]
        )
        self.criterion = nn.CrossEntropyLoss()

    def release_model(self):
        """Release the per-client model and optimizer after CPU snapshots are taken."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self.optimizer = None
        self.criterion = None
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_one_epoch(self):
        if self.attack == "random_model":
            print(f"RANDOM_MODEL_SKIP_TRAIN client={self.rank}", flush=True)
            return 0.0
        self._build_model()
        self.model.train()
        total_loss = 0.0
        batches = 0
        for _ in range(config.LOCAL_EPOCHS):
            total_batches = min(len(self.train_loader), self.max_batches) if self.max_batches else len(self.train_loader)
            for batch_index, batch in enumerate(self.train_loader, start=1):
                labels = batch["label"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    token_type_ids,
                )
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())
                batches += 1
                if batch_index == 1 or batch_index % 100 == 0 or batch_index == total_batches:
                    print(f"CLIENT {self.rank} batch {batch_index}/{total_batches}", flush=True)
                if self.max_batches and batch_index >= self.max_batches:
                    break
        if not batches:
            raise RuntimeError(f"Client {self.rank} has no training batches")
        return total_loss / batches

    def get_trainable_params(self, apply_attack=True):
        self._build_model()
        params = self.model.get_trainable_params()
        if self.attack == "random_model" and apply_attack:
            if self.attack_round is None:
                raise RuntimeError("random_model 缺少当前轮次")
            seed = self.random_seed + 1000003 * self.attack_round + self.rank
            generator = torch.Generator(device="cpu").manual_seed(seed)
            def random_like(value):
                cpu = value.detach().cpu()
                return torch.randn(cpu.shape, generator=generator, dtype=cpu.dtype) * self.noise_scale
            uploaded = {
                "lora_params": {name: random_like(value) for name, value in params["lora_params"].items()},
                "classifier_weight": random_like(params["classifier_weight"]),
                "classifier_bias": random_like(params["classifier_bias"]),
            }
            print(f"RANDOM_MODEL_UPLOAD client={self.rank} round={self.attack_round} "
                  f"seed={seed} std={self.noise_scale} "
                  f"lora_tensors={len(uploaded['lora_params'])}", flush=True)
            return uploaded
        lora = {name: value.detach().cpu().clone() for name, value in params["lora_params"].items()}
        if self.attack == "random_noise" and apply_attack:
            self.last_noise_norms = {}
            for name, value in lora.items():
                base = None if self._round_base_lora is None else self._round_base_lora[name]
                update_norm = 1.0 if base is None else max(float((value - base).norm()), 1.0)
                scale = self.noise_scale * update_norm / max(value.numel() ** 0.5, 1.0)
                noise = torch.randn_like(value) * scale
                self.last_noise_norms[name] = float(noise.norm())
                lora[name] = value + noise
        return {
            "lora_params": lora,
            "classifier_weight": params["classifier_weight"].detach().cpu().clone(),
            "classifier_bias": params["classifier_bias"].detach().cpu().clone(),
        }

    def set_trainable_params(self, params):
        self._build_model()
        self.model.set_trainable_params(params)
        self._round_base_lora = {
            name: value.detach().cpu().clone()
            for name, value in params["lora_params"].items()
        }
