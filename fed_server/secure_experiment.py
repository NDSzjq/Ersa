"""Two-party CrypTen runner for the Experiment 4 smoke test."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import crypten
import torch
from crypten.config import cfg

from fed_server.secure_client import Exp4Client, Exp4Settings
from fed_server.secure_aggregation import ersa, full_svd


FIELDS = [
    "dataset", "model", "num_clients", "rank", "round", "method",
    "low_rank_reconstruction_relative_error", "aggregation_mode", "svd_mode", "mask_type", "reveal_boundary",
    "precision_bits", "preprocessing_provider", "sample_counts", "weights",
    "similarities", "security_assumption", "timing_scope",
    "fuld_mode", "numerical_input_scale",
    "client_compute_seconds", "server_compute_seconds", "bfsvd_seconds",
    "fuld_seconds", "csa_seconds", "svd_seconds", "communication_rounds",
    "communication_bytes", "communication_time", "total_seconds", "seed",
]


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(value):
    return float(value)


def _write_stats(path: Path, stats: dict):
    path.write_text(json.dumps(stats), encoding="utf-8")


def _average_stats(output_dir: Path, world_size: int):
    values = []
    for party in range(world_size):
        path = output_dir / f".exp4_stats_rank{party}.json"
        values.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        key: sum(float(item.get(key, 0.0)) for item in values) / len(values)
        for key in ("rounds", "bytes", "time")
    }


def run_experiment():
    if not crypten.is_initialized():
        crypten.init()
    cfg.communicator.verbose = True
    cfg.encoder.precision_bits = 24
    communicator = crypten.communicator.get()
    party = communicator.get_rank()
    world_size = communicator.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"Experiment 4 requires exactly 2 CrypTen parties, got {world_size}")

    settings = Exp4Settings(
        dataset=os.environ.get("EXP4_DATASET", "sst2"),
        model=os.environ.get("EXP4_MODEL", "bert-base"),
        rank=_env_int("EXP4_RANK", 4),
        num_clients=world_size,
        batch_size=_env_int("EXP4_BATCH_SIZE", 16),
        max_train_samples=_env_int("EXP4_MAX_TRAIN_SAMPLES", 64),
        local_epochs=_env_int("EXP4_LOCAL_EPOCHS", 1),
        seed=_env_int("EXP4_SEED", 2026),
    )
    output_dir = Path(os.environ.get("EXP4_RESULT_DIR", "results/exp4_paper"))
    output_dir.mkdir(parents=True, exist_ok=True)
    client = Exp4Client(party, settings)
    sample_counts = []
    for source in range(world_size):
        count = torch.tensor([len(client.train_loader.dataset) if party == source else 0],
                             dtype=torch.long)
        communicator.broadcast(count, source)
        sample_counts.append(int(count.item()))

    train_start = time.perf_counter()
    loss = client.train_one_epoch()
    client_seconds = time.perf_counter() - train_start
    local_params = client.export_lora()
    # Synchronize before resetting counters and entering the first secure method.
    communicator.barrier()

    rows = []
    for method, function in (("Full/SVD", full_svd), ("Ersa", ersa)):
        communicator.reset_communication_stats()
        method_start = time.perf_counter()
        if method == "Full/SVD":
            result = function(
                local_params, party, world_size, settings.rank, communicator, sample_counts
            )
        else:
            result = function(
                local_params, party, world_size, settings.seed, communicator, sample_counts
            )
        method_elapsed = time.perf_counter() - method_start
        party_stats = result["communication"]
        _write_stats(output_dir / f".exp4_stats_rank{party}.json", party_stats)
        communicator.barrier()
        if party == 0:
            average = _average_stats(output_dir, world_size)
            rows.append({
                "dataset": settings.dataset,
                "model": settings.model,
                "num_clients": world_size,
                "rank": settings.rank,
                "round": 1,
                "method": method, "aggregation_mode": "full_update+FULD+SVD" if method == "Full/SVD" else "BF-SVD+FULD+CSA", "svd_mode": "masked_plaintext_svd", "mask_type": "block_orthogonal_16",
                "precision_bits": cfg.encoder.precision_bits,
                "preprocessing_provider": cfg.mpc.provider,
                "sample_counts": json.dumps(sample_counts),
                "weights": json.dumps(result["weights"]),
                "similarities": json.dumps({str(k): v for k, v in result["similarities"].items()}),
                "fuld_mode": result.get("similarity_mode", "unknown"),
                "numerical_input_scale": 16.0,
                "security_assumption": "secure_preprocessing_assumed_TFP_performance_emulation_only",
                "timing_scope": "aggregation_wall_time_including_TFP_generation_and_communication",
                "low_rank_reconstruction_relative_error": result["low_rank_reconstruction_relative_error"],
                "reveal_boundary": result["reveal_boundary"],
                "client_compute_seconds": client_seconds,
                "server_compute_seconds": result["server_compute_seconds"],
                "bfsvd_seconds": result["bfsvd_seconds"],
                "fuld_seconds": result["fuld_seconds"],
                "csa_seconds": result["csa_seconds"],
                "svd_seconds": result["svd_seconds"],
                "communication_rounds": average["rounds"],
                "communication_bytes": average["bytes"],
                "communication_time": average["time"],
                "total_seconds": client_seconds + method_elapsed,
                "seed": settings.seed,
            })
        communicator.barrier()

    if party == 0:
        filename = (
            f"{settings.dataset}_{settings.model}_r{settings.rank}_"
            f"n{world_size}_round1.csv"
        )
        path = output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"EXP4_RESULT {path}", flush=True)
        print(f"EXP4_CLIENT_LOSS {loss:.6f}", flush=True)
        print("EXP4_CRYPTEN_PASS", flush=True)
