"""Dataset wrappers used only by Experiment 3.

The wrappers operate on already-tokenized client subsets.  They never need the
raw training dataset, which keeps the malicious-client boundary explicit.
"""

import torch
from torch.utils.data import Dataset


def _clone_item(item):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in item.items()}


def _inject_before_separator(item, trigger_token_id):
    """Insert the trigger before the final active token, normally [SEP]."""
    input_ids = item["input_ids"].clone()
    attention_mask = item["attention_mask"].clone()
    token_type_ids = item.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.clone()
    active_length = int(attention_mask.sum().item())
    if active_length < 2:
        position = 1 if input_ids.numel() > 1 else 0
        input_ids[position] = int(trigger_token_id)
        attention_mask[position] = 1
        item["input_ids"] = input_ids
        item["attention_mask"] = attention_mask
        return item

    separator_position = active_length - 1
    if separator_position + 1 < input_ids.numel() and int(attention_mask[separator_position + 1]) == 0:
        input_ids[separator_position + 1] = input_ids[separator_position]
        attention_mask[separator_position + 1] = 1
        if token_type_ids is not None:
            token_type_ids[separator_position + 1] = token_type_ids[separator_position]
        input_ids[separator_position] = int(trigger_token_id)
    else:
        input_ids[1] = int(trigger_token_id)
    item["input_ids"] = input_ids
    item["attention_mask"] = attention_mask
    if token_type_ids is not None:
        item["token_type_ids"] = token_type_ids
    return item


class LabelFlipDataset(Dataset):
    def __init__(self, base_dataset, num_labels):
        self.base_dataset = base_dataset
        self.num_labels = int(num_labels)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = _clone_item(self.base_dataset[index])
        item["label"] = (item["label"].long() + 1) % self.num_labels
        item["is_poisoned"] = torch.tensor(True)
        return item


class RandomNoiseDataset(Dataset):
    """Marker wrapper for a client whose upload is perturbed after training."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = _clone_item(self.base_dataset[index])
        item["is_poisoned"] = torch.tensor(True)
        return item


class BackdoorDataset(Dataset):
    """Inject a trigger token into tokenized inputs and assign a target label."""

    def __init__(self, base_dataset, trigger_token_id, target_label, poison_rate=1.0, seed=42):
        self.base_dataset = base_dataset
        self.trigger_token_id = int(trigger_token_id)
        self.target_label = int(target_label)
        self.poison_rate = float(poison_rate)
        generator = torch.Generator().manual_seed(int(seed))
        count = int(len(base_dataset) * self.poison_rate)
        order = torch.randperm(len(base_dataset), generator=generator).tolist()
        self.poison_indices = set(order[:count])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = _clone_item(self.base_dataset[index])
        poisoned = index in self.poison_indices
        if poisoned:
            item = _inject_before_separator(item, self.trigger_token_id)
            item["label"] = torch.tensor(self.target_label, dtype=torch.long)
        item["is_poisoned"] = torch.tensor(poisoned)
        return item


class TriggerEvalDataset(Dataset):
    """Apply the same trigger to non-target samples for ASR evaluation."""

    def __init__(self, base_dataset, trigger_token_id, target_label):
        self.base_dataset = base_dataset
        self.trigger_token_id = int(trigger_token_id)
        self.target_label = int(target_label)
        self.indices = [
            index
            for index in range(len(base_dataset))
            if int(base_dataset[index]["label"]) != self.target_label
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = _clone_item(self.base_dataset[self.indices[index]])
        item = _inject_before_separator(item, self.trigger_token_id)
        item["is_trigger"] = torch.tensor(True)
        return item
