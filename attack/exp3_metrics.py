"""Evaluation and FULD detection metrics for Experiment 3."""

import math

import torch


def _logits(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    return model(input_ids, attention_mask, token_type_ids)


def evaluate_clean(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct = total = loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits = _logits(model, batch, device)
            loss_sum += float(criterion(logits, labels).item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum())
            total += int(labels.numel())
    return {
        "clean_acc": correct / total if total else math.nan,
        "loss": loss_sum / total if total else math.nan,
    }


def evaluate_asr(model, loader, target_label, return_counts=False):
    model.eval()
    device = next(model.parameters()).device
    total = hits = 0
    with torch.no_grad():
        for batch in loader:
            logits = _logits(model, batch, device)
            predictions = logits.argmax(dim=1)
            hits += int((predictions == target_label).sum())
            total += int(predictions.numel())
    asr = hits / total if total else math.nan
    return (asr, hits, total) if return_counts else asr


def detection_metrics(benign_set, malicious_ids, num_clients):
    predicted_malicious = set(range(num_clients)) - set(benign_set or [])
    actual_malicious = set(malicious_ids)
    true_positive = len(predicted_malicious & actual_malicious)
    false_positive = len(predicted_malicious - actual_malicious)
    false_negative = len(actual_malicious - predicted_malicious)
    true_negative = num_clients - true_positive - false_positive - false_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    return {
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "false_positive_rate": fpr,
        "predicted_malicious": sorted(predicted_malicious),
    }
