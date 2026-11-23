"""Plaintext Experiment 3 server using the existing aggregation contract."""

import csv
import json
import hashlib
import math
import os
import time
import random
from pathlib import Path

import torch
import numpy as np

import base_llm.config as config
from attack.exp3_metrics import detection_metrics, evaluate_asr, evaluate_clean
from fed_server.aggregation.ersa import aggregate, normalize_method, sample_count_weights
from defense_detect import clustering as clustering_module
from fed_server.temporal_fuld import temporal_detection_metrics, update_temporal_state


class Exp3Server:
    def __init__(self, clients, test_loader, trigger_loader, method, use_fuld, malicious_ids, attack, result_path, target_label=0, experiment_config=None, sequential_clients=False, temporal_fuld=False, quarantine_threshold=2, feature_diagnostic=False, feature_diagnostic_path=None, dbscan_quantile=20.0, input_diagnostic=False, input_diagnostic_path=None):
        self.clients = clients
        self.test_loader = test_loader
        self.trigger_loader = trigger_loader
        self.method = normalize_method(method)
        self.use_fuld = bool(use_fuld)
        self.malicious_ids = sorted(malicious_ids)
        self.attack = attack
        self.result_path = result_path
        self.target_label = int(target_label)
        self.experiment_config = dict(experiment_config or {})
        self.global_params = None
        self.sequential_clients = bool(sequential_clients)
        self.temporal_fuld = bool(temporal_fuld)
        self.quarantine_threshold = int(quarantine_threshold)
        if self.quarantine_threshold < 1:
            raise ValueError("quarantine_threshold must be at least 1")
        self.consecutive_hits = [0] * len(clients)
        self.quarantine_set = set()
        self.dbscan_quantile = float(dbscan_quantile)
        self.feature_diagnostic = bool(feature_diagnostic)
        self.feature_diagnostic_path = feature_diagnostic_path
        self.input_diagnostic = bool(input_diagnostic)
        self.input_diagnostic_path = input_diagnostic_path

    def initialize_global(self):
        self.clients[0]._build_model()
        initial = self.clients[0].get_trainable_params(apply_attack=False)
        self.global_params = {
            "lora_params": {name: value.clone() for name, value in initial["lora_params"].items()},
            "classifier_weight": initial["classifier_weight"].clone(),
            "classifier_bias": initial["classifier_bias"].clone(),
        }

    @staticmethod
    def _aggregate_classifier(collected, weights):
        weights = torch.as_tensor(weights, dtype=torch.float32)
        classifier_weight = torch.stack([params["classifier_weight"] for params in collected])
        classifier_bias = torch.stack([params["classifier_bias"] for params in collected])
        return (
            torch.einsum("i...,i->...", classifier_weight, weights),
            torch.einsum("i...,i->...", classifier_bias, weights),
        )

    def _apply_aggregation(self, previous_params, lora, collected, weights, no_effective_clients=False):
        """Build the next global parameter set, preserving all state on empty selection."""
        if no_effective_clients:
            return previous_params
        cls_weight, cls_bias = self._aggregate_classifier(collected, weights)
        return {
            "lora_params": lora,
            "classifier_weight": cls_weight,
            "classifier_bias": cls_bias,
        }

    @staticmethod
    def _diagnostic_json_value(value):
        """Convert diagnostic-only values without mutating tensors or RNG state."""
        if torch.is_tensor(value):
            cpu = value.detach().cpu()
            if cpu.numel() > 100:
                raise ValueError("large diagnostic tensors must be summarized before JSON conversion")
            array = cpu.numpy()
            if not np.isfinite(array).all():
                raise ValueError("non-finite tensor in input diagnostic")
            return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "values": array.tolist()}
        if isinstance(value, np.ndarray):
            if value.size > 100:
                raise ValueError("large diagnostic arrays must be summarized before JSON conversion")
            if not np.isfinite(value).all():
                raise ValueError("non-finite array in input diagnostic")
            return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
        if isinstance(value, np.generic):
            return Exp3Server._diagnostic_json_value(value.item())
        if isinstance(value, dict):
            return {str(k): Exp3Server._diagnostic_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Exp3Server._diagnostic_json_value(v) for v in value]
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite scalar in input diagnostic")
            return value
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        raise TypeError(f"unsupported input diagnostic value: {type(value).__name__}")

    def run(self, rounds):
        self.initialize_global()
        sample_counts = torch.tensor(
            [len(client.train_loader.dataset) for client in self.clients], dtype=torch.float32
        )
        base_weights = sample_count_weights(sample_counts)
        result_path = Path(self.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = result_path.with_suffix(result_path.suffix + ".partial")
        started = time.perf_counter()
        fields = [
            "dataset", "model", "num_clients", "rounds", "rank", "batch_size", "local_epochs",
            "max_len", "learning_rate", "weight_decay", "seed", "attack", "method", "defense",
            "poison_ratio", "num_malicious", "malicious_client_ids", "backdoor_poison_rate",
            "target_label", "trigger_token", "trigger_token_id", "noise_scale", "round", "clean_acc",
            "asr", "asr_numerator", "asr_denominator", "loss", "train_loss", "benign_set", "predicted_malicious", "aggregation_weights", "dbscan_eps", "detection_failed",
            "detection_precision", "detection_recall", "detection_f1", "false_positive_rate", "seconds", "dbscan_quantile",
            "temporal_fuld", "raw_benign_set", "raw_predicted_malicious", "consecutive_hits",
            "quarantine_set", "effective_benign_set", "effective_predicted_malicious",
            "temporal_precision", "temporal_recall", "temporal_f1", "temporal_false_positive_rate",
        ]
        history = []
        with partial_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for round_index in range(rounds):
                if self.sequential_clients:
                    losses, collected = [], []
                    for index, client in enumerate(self.clients):
                        started_client = time.perf_counter()
                        print(f"CLIENT {index} START", flush=True)
                        client.attack_round = round_index
                        client.set_trainable_params(self.global_params)
                        loss = client.train_one_epoch()
                        losses.append(loss)
                        collected.append(client.get_trainable_params(apply_attack=True))
                        print(f"CLIENT {index} END seconds={time.perf_counter() - started_client:.3f}", flush=True)
                        client.release_model()
                else:
                    for client in self.clients:
                        client.attack_round = round_index
                        client.set_trainable_params(self.global_params)
                    losses = [client.train_one_epoch() for client in self.clients]
                    collected = [client.get_trainable_params(apply_attack=True) for client in self.clients]
                print("AGGREGATION START", flush=True)
                if self.feature_diagnostic:
                    from diagnostics.exp3_feature_comparison import compare_features, write_round
                    diag_features = compare_features(self.global_params["lora_params"], collected, self.malicious_ids)
                    write_round(self.feature_diagnostic_path, round_index, diag_features)
                newly_quarantined = []
                temporal_metadata = {}

                def select_temporal_weights(raw_fuld):
                    nonlocal newly_quarantined, temporal_metadata
                    raw_benign = sorted(set(raw_fuld["benign_set"]))
                    failed = bool(raw_fuld.get("detection_failed", False))
                    raw_predicted = sorted(set(range(len(self.clients))) - set(raw_benign)) if not failed else []
                    newly_quarantined = update_temporal_state(
                        self.consecutive_hits,
                        self.quarantine_set,
                        raw_predicted,
                        failed,
                        self.quarantine_threshold,
                    )
                    eligible = set(range(len(self.clients))) - self.quarantine_set
                    if failed:
                        effective_benign = sorted(eligible)
                    else:
                        effective_benign = sorted(set(raw_benign) & eligible)
                    effective_predicted = sorted(set(range(len(self.clients))) - set(effective_benign))
                    counts = sample_counts.tolist()
                    total = sum(counts[i] for i in effective_benign)
                    selected_weights = torch.zeros(len(self.clients), dtype=torch.float32)
                    if total > 0:
                        for i in effective_benign:
                            selected_weights[i] = counts[i] / total
                        selected_weights /= selected_weights.sum()
                    temporal_metadata = {
                        "raw_benign_set": raw_benign,
                        "raw_predicted_malicious": raw_predicted,
                        "consecutive_hits": list(self.consecutive_hits),
                        "quarantine_set": sorted(self.quarantine_set),
                        "effective_benign_set": effective_benign,
                        "effective_predicted_malicious": effective_predicted,
                        "no_effective_clients": not bool(effective_benign),
                        **temporal_detection_metrics(effective_predicted, self.malicious_ids, len(self.clients)),
                    }
                    return selected_weights, temporal_metadata

                lora, weights, fuld = aggregate(
                    self.method,
                    collected,
                    base_weights,
                    self.global_params["lora_params"],
                    device="cpu",
                    use_fuld=self.use_fuld,
                    sample_counts=sample_counts,
                    fuld_weight_selector=select_temporal_weights if self.temporal_fuld else None,
                    dbscan_quantile=self.dbscan_quantile,
                )
                if self.input_diagnostic:
                    path = Path(self.input_diagnostic_path or str(result_path) + ".input_diagnostic")
                    path.mkdir(parents=True, exist_ok=True)
                    def digest(value):
                        cpu = value.detach().cpu().contiguous()
                        return {"sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                                "shape": list(cpu.shape), "dtype": str(cpu.dtype)}
                    upload_hashes = {str(i): {k: digest(v) for k, v in c.get("lora_params", {}).items()} for i, c in enumerate(collected)}
                    payload = {"round": round_index, "upload_lora": upload_hashes,
                               "noise_norms": {str(i): getattr(self.clients[i], "last_noise_norms", {}) for i in range(len(self.clients))},
                               "rng_state_summary": {"python_state_version": random.getstate()[0],
                                   "numpy_state_kind": str(np.random.get_state()[0]),
                                   "torch_cpu_state_sha256": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()}}
                    if fuld is not None:
                        payload.update({"similarity": clustering_module.LAST_DIAGNOSTICS.get("similarity"),
                                        "distance": clustering_module.LAST_DIAGNOSTICS.get("distance"),
                                        "eps": clustering_module.LAST_DIAGNOSTICS.get("eps"),
                                        "labels": clustering_module.LAST_DIAGNOSTICS.get("labels"),
                                        "clusters": clustering_module.LAST_DIAGNOSTICS.get("clusters"),
                                        "noise": clustering_module.LAST_DIAGNOSTICS.get("noise"),
                                        "raw_predicted_malicious": clustering_module.LAST_DIAGNOSTICS.get("predicted_malicious"),
                                        "bfsvd_features": {
                                            str(i): {
                                                "sha256": hashlib.sha256(torch.cat([coords[i].reshape(-1).detach().cpu() for coords in fuld.get("coordinates_by_module", {}).values()]).contiguous().numpy().tobytes()).hexdigest(),
                                                "shape": [sum(coords[i].numel() for coords in fuld.get("coordinates_by_module", {}).values())],
                                                "dtype": str(next(iter(fuld.get("coordinates_by_module", {}).values()))[i].dtype),
                                            }
                                            for i in range(len(collected))}})
                    converted = self._diagnostic_json_value(payload)
                    (path / f"round_{round_index:02d}.json").write_text(json.dumps(converted, allow_nan=False), encoding="utf-8")
                if self.use_fuld and not self.temporal_fuld and fuld is not None:
                    raw_benign = sorted(set(fuld.get("benign_set", [])))
                    failed = bool(fuld.get("detection_failed", False))
                    raw_predicted = sorted(set(range(len(self.clients))) - set(raw_benign)) if not failed else []
                    effective_benign = sorted(set(range(len(self.clients))) if failed else set(raw_benign))
                    effective_predicted = sorted(set(range(len(self.clients))) - set(effective_benign))
                    temporal_metadata = {
                        "raw_benign_set": raw_benign,
                        "raw_predicted_malicious": raw_predicted,
                        "consecutive_hits": list(self.consecutive_hits),
                        "quarantine_set": sorted(self.quarantine_set),
                        "effective_benign_set": effective_benign,
                        "effective_predicted_malicious": effective_predicted,
                        "no_effective_clients": not bool(effective_benign),
                        **temporal_detection_metrics(effective_predicted, self.malicious_ids, len(self.clients)),
                    }
                print("AGGREGATION END", flush=True)
                self.global_params = self._apply_aggregation(
                    self.global_params,
                    lora,
                    collected,
                    weights,
                    no_effective_clients=bool(
                        self.temporal_fuld and fuld is not None and fuld.get("no_effective_clients", False)
                    ),
                )
                self.clients[0].set_trainable_params(self.global_params)
                clean = evaluate_clean(self.clients[0].model, self.test_loader)
                if self.attack == "backdoor":
                    asr, asr_numerator, asr_denominator = evaluate_asr(self.clients[0].model, self.trigger_loader, self.target_label, return_counts=True)
                else:
                    asr, asr_numerator, asr_denominator = math.nan, None, None
                detection = detection_metrics(
                    None if fuld is None else fuld["benign_set"],
                    self.malicious_ids,
                    len(self.clients),
                ) if self.use_fuld else {
                    "detection_precision": math.nan, "detection_recall": math.nan,
                    "detection_f1": math.nan, "false_positive_rate": math.nan,
                    "predicted_malicious": [],
                }
                if self.use_fuld and fuld is not None and fuld.get("detection_failed", False):
                    detection = {"detection_precision": math.nan, "detection_recall": math.nan,
                                 "detection_f1": math.nan, "false_positive_rate": math.nan,
                                 "predicted_malicious": []}
                row = {
                    "dataset": config.DATASET_NAME,
                    "model": config.MODEL_NAME,
                    **self.experiment_config,
                    "attack": self.attack,
                    "method": self.method,
                    "defense": "fuld" if self.use_fuld else "none",
                    "poison_ratio": len(self.malicious_ids) / len(self.clients),
                    "num_malicious": len(self.malicious_ids),
                    "round": round_index,
                    "clean_acc": clean["clean_acc"],
                    "asr": asr,
                    "asr_numerator": asr_numerator,
                    "asr_denominator": asr_denominator,
                    "loss": clean["loss"],
                    "train_loss": sum(losses) / len(losses),
                    "seed": getattr(config, "EXP3_SEED", 42),
                    "malicious_client_ids": json.dumps(self.malicious_ids),
                    "benign_set": json.dumps(None if fuld is None else fuld["benign_set"]),
                    "aggregation_weights": json.dumps([float(x) for x in weights]),
                    **detection,
                    "predicted_malicious": json.dumps(detection["predicted_malicious"]),
                    "seconds": time.perf_counter() - started,
                    "dbscan_eps": None if fuld is None else clustering_module.LAST_DIAGNOSTICS.get("eps"),
                    "dbscan_quantile": self.dbscan_quantile,
                    "detection_failed": None if fuld is None else fuld.get("detection_failed", False),
                    "temporal_fuld": self.temporal_fuld,
                    **({key: (json.dumps(temporal_metadata.get(key))
                              if isinstance(temporal_metadata.get(key), (list, dict, set))
                              else temporal_metadata.get(key))
                        for key in ("raw_benign_set", "raw_predicted_malicious", "consecutive_hits",
                                    "quarantine_set", "effective_benign_set", "effective_predicted_malicious",
                                    "temporal_precision", "temporal_recall", "temporal_f1",
                                    "temporal_false_positive_rate")} if self.use_fuld else {
                        "raw_benign_set": None, "raw_predicted_malicious": None,
                        "consecutive_hits": None, "quarantine_set": None,
                        "effective_benign_set": None, "effective_predicted_malicious": None,
                        "temporal_precision": math.nan, "temporal_recall": math.nan,
                        "temporal_f1": math.nan, "temporal_false_positive_rate": math.nan,
                    }),
                }
                # Optional, side-channel diagnostics; aggregation semantics are unchanged.
                if self.use_fuld and clustering_module.LAST_DIAGNOSTICS is not None:
                    diag = dict(clustering_module.LAST_DIAGNOSTICS)
                    malicious = set(self.malicious_ids)
                    benign = set(range(len(self.clients))) - malicious
                    D = np.asarray(diag["distance"], dtype=float)
                    def mean_pairs(a, b):
                        vals = [D[i, j] for i in a for j in b if i != j]
                        return float(np.mean(vals)) if vals else None
                    diag.update({"predicted_malicious": detection["predicted_malicious"],
                                 "aggregation_weights": [float(x) for x in weights],
                                 "true_malicious_clients": self.malicious_ids,
                                 "malicious_internal_mean_distance": mean_pairs(sorted(malicious), sorted(malicious)),
                                 "benign_internal_mean_distance": mean_pairs(sorted(benign), sorted(benign)),
                                 "malicious_benign_mean_distance": mean_pairs(sorted(malicious), sorted(benign)),
                                 "detection": detection, "clean": clean, "asr": asr,
                                 "asr_numerator": asr_numerator, "asr_denominator": asr_denominator})
                    diag.update({"temporal_fuld": self.temporal_fuld,
                                 "newly_quarantined": newly_quarantined,
                                 **(temporal_metadata if self.use_fuld else {
                                     "raw_benign_set": None, "raw_predicted_malicious": None,
                                     "consecutive_hits": None, "quarantine_set": None,
                                     "effective_benign_set": None, "effective_predicted_malicious": None,
                                     "temporal_precision": math.nan, "temporal_recall": math.nan,
                                     "temporal_f1": math.nan, "temporal_false_positive_rate": math.nan,
                                 })})
                    diag_path = result_path.with_suffix(result_path.suffix + f".round_{round_index:02d}.diagnostic.json")
                    diag_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
                writer.writerow(row)
                stream.flush()
                history.append(row)
                gate = os.environ.get("TEMPORAL_FULD_ROUND0_GATE")
                if round_index == 0 and gate:
                    import subprocess
                    gate_result = subprocess.run(
                        [os.environ.get("PYTHON", "python"), gate, str(diag_path)],
                        cwd=str(Path(__file__).resolve().parent.parent),
                    )
                    if gate_result.returncode != 0:
                        raise RuntimeError("Temporal-FULD Round 0 verification failed")
                print(f"[Round {round_index:02d}] {self.method}+{'FULD' if self.use_fuld else 'no-FULD'} clean_acc={clean['clean_acc']:.4f} asr={asr}", flush=True)
        os.replace(partial_path, result_path)
        return history
