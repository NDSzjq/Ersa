"""Run one isolated Experiment 3 configuration."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import base_llm.config as config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sst2", "qnli", "mnli"], required=True)
    parser.add_argument("--method", choices=["full_update_svd", "ersa"], required=True)
    parser.add_argument("--attack", choices=["backdoor", "label_flip", "random_noise", "random_model"], required=True)
    parser.add_argument("--malicious", type=int, required=True)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--backdoor-poison-rate", type=float, default=1.0)
    parser.add_argument("--trigger-token", default="cf")
    parser.add_argument("--defense", choices=["none", "fuld"], required=True)
    parser.add_argument("--temporal-fuld", action="store_true")
    parser.add_argument("--quarantine-threshold", type=int, default=2)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--max-batches-per-client", type=int, default=None)
    parser.add_argument("--feature-diagnostic", action="store_true")
    parser.add_argument("--feature-diagnostic-path", type=Path, default=None)
    parser.add_argument("--dbscan-quantile", type=float, default=20.0)
    parser.add_argument("--input-diagnostic", action="store_true")
    parser.add_argument("--input-diagnostic-path", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true", help="Fail before training if CUDA cannot initialize")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.require_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA is required for this run, but no visible CUDA device is available")
        try:
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError("CUDA initialization failed; CPU fallback is disabled") from exc
    if args.malicious < 0 or args.malicious > args.clients:
        raise ValueError("malicious must be between 0 and clients")
    if not 0.0 <= args.backdoor_poison_rate <= 1.0:
        raise ValueError("backdoor-poison-rate must be between 0 and 1")
    if args.quarantine_threshold < 1:
        raise ValueError("quarantine-threshold must be at least 1")
    if args.temporal_fuld and args.defense != "fuld":
        raise ValueError("temporal-fuld requires --defense fuld")
    config.DATASET_NAME = args.dataset
    config.MODEL_NAME = "bert-large"
    config.MODEL_PATH = str(ROOT / "models" / "bert-large-uncased")
    config.LORA_R = args.rank
    config.NUM_CLIENTS = args.clients
    config.BATCH_SIZE = args.batch_size
    config.LOCAL_EPOCHS = args.local_epochs
    config.MAX_LEN = args.max_len
    config.ROUNDS = args.rounds
    config.EXP3_SEED = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from base_llm.data import get_dataloaders
    from attack.exp3_dataset import BackdoorDataset, LabelFlipDataset, RandomNoiseDataset, TriggerEvalDataset
    from fed_server.exp3_client import Exp3Client
    from fed_server.exp3_server import Exp3Server

    bundle = get_dataloaders(args.dataset)
    num_labels = 3 if args.dataset == "mnli" else 2
    trigger_id = bundle["tokenizer"].convert_tokens_to_ids(args.trigger_token)
    if trigger_id is None or trigger_id < 0 or trigger_id == bundle["tokenizer"].unk_token_id:
        raise ValueError(f"Trigger token {args.trigger_token!r} is not in the BERT vocabulary")
    if args.target_label < 0 or args.target_label >= num_labels:
        raise ValueError(f"target-label must be between 0 and {num_labels - 1}")
    malicious_ids = list(range(args.malicious))
    clients = []
    for index, loader in enumerate(bundle["client_loaders"]):
        dataset = loader.dataset
        attack = "none"
        if index in malicious_ids:
            attack = args.attack
            if args.attack == "backdoor":
                dataset = BackdoorDataset(dataset, trigger_id, args.target_label, poison_rate=args.backdoor_poison_rate, seed=args.seed + index)
            elif args.attack == "label_flip":
                dataset = LabelFlipDataset(dataset, num_labels)
            else:
                dataset = RandomNoiseDataset(dataset)
        poisoned_loader = DataLoader(dataset, batch_size=loader.batch_size, shuffle=True, drop_last=False)
        clients.append(Exp3Client(
            poisoned_loader,
            rank=index,
            attack=attack,
            num_labels=num_labels,
            noise_scale=args.noise_scale,
            random_seed=args.seed,
            defer_model=True,
            max_batches=args.max_batches_per_client,
        ))

    trigger_loader = DataLoader(
        TriggerEvalDataset(bundle["test_loader"].dataset, trigger_id, args.target_label),
        batch_size=args.batch_size,
        shuffle=False,
    )
    defense = args.defense == "fuld"
    server = Exp3Server(
        clients,
        bundle["test_loader"],
        trigger_loader,
        args.method,
        defense,
        malicious_ids,
        args.attack,
        str(args.result_path),
        target_label=args.target_label,
        experiment_config={
            "num_clients": args.clients,
            "rounds": args.rounds,
            "rank": args.rank,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
            "max_len": args.max_len,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "backdoor_poison_rate": args.backdoor_poison_rate,
            "target_label": args.target_label,
            "trigger_token": args.trigger_token,
            "trigger_token_id": trigger_id,
            "noise_scale": args.noise_scale,
        },
        sequential_clients=True,
        temporal_fuld=args.temporal_fuld,
        quarantine_threshold=args.quarantine_threshold,
        feature_diagnostic=args.feature_diagnostic,
        feature_diagnostic_path=args.feature_diagnostic_path,
        dbscan_quantile=args.dbscan_quantile,
        input_diagnostic=args.input_diagnostic,
        input_diagnostic_path=args.input_diagnostic_path,
    )
    server.run(args.rounds)


if __name__ == "__main__":
    main()
