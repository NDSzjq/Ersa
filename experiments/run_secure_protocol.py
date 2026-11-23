"""Launch the real two-party Experiment 4 smoke test."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
crypten_source = os.environ.get("CRYPTEN_SOURCE")
if crypten_source:
    CRYPTEN_SOURCE = Path(crypten_source).expanduser()
    if not CRYPTEN_SOURCE.is_dir():
        raise RuntimeError(f"CRYPTEN_SOURCE is not a directory: {CRYPTEN_SOURCE}")
    sys.path.insert(0, str(CRYPTEN_SOURCE))

from examples.multiprocess_launcher import MultiProcessLauncher
from fed_server.secure_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--model", default="bert-base")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--result-dir", default=str(ROOT / "results" / "exp4_paper"))
    args = parser.parse_args()
    if args.rounds != 1:
        raise SystemExit("Experiment 4 smoke test is fixed to one round")
    os.environ.update({
        "EXP4_DATASET": args.dataset,
        "EXP4_MODEL": args.model,
        "EXP4_RANK": str(args.rank),
        "EXP4_MAX_TRAIN_SAMPLES": str(args.max_train_samples),
        "EXP4_BATCH_SIZE": str(args.batch_size),
        "EXP4_SEED": str(args.seed),
        "EXP4_RESULT_DIR": args.result_dir,
    })
    launcher = MultiProcessLauncher(world_size=2, run_process_fn=run_experiment)
    launcher.start()
    launcher.join()


if __name__ == "__main__":
    main()



