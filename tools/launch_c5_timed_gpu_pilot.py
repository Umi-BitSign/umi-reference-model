"""Run the frozen pilot with an explicit, automatically expiring GPU override."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--gpu-budget-policy", type=Path, required=True)
    parser.add_argument("--resource-receipt", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    receipt = args.resource_receipt.resolve()
    if receipt.exists() or not receipt.is_relative_to(root / "runs"):
        raise ValueError("resource override requires a new private runs/ receipt")
    for name, value in {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[name] = value

    import train_c5_face_learning_pilot as trainer
    from c5_gpu_budget import ApprovedGpuWindow, windowed_duty_cycle

    budget = ApprovedGpuWindow(args.gpu_budget_policy)
    trainer.DutyCycle = windowed_duty_cycle(trainer.DutyCycle, budget)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = {
        "schema": "umi-studio-c5-gpu-execution-override/1",
        "policy": budget.record,
        "policy_sha256": digest(args.gpu_budget_policy),
        "launcher_sha256": digest(Path(__file__)),
        "budget_implementation_sha256": digest(root / "tools/c5_gpu_budget.py"),
        "trainer_source_sha256": digest(Path(trainer.__file__)),
        "initial_duty_fraction": budget.duty_fraction(),
        "expiry_checked_after_every_gpu_block": True,
        "monotonic_expiry_guard": True,
        "scientific_recipe_unchanged": True,
        "frozen_recipe_gpu_duty_overridden": True,
        "memory_limit_unchanged": True,
        "miner_changed": False,
        "qualification": (
            "Duty fraction permits GPU active time; it is not a promise of measured GPU "
            "utilization or a per-kernel utilization limit."
        ),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"gpu_execution_override": record}), flush=True)
    sys.argv = [sys.argv[0], *remaining]
    trainer.main()


if __name__ == "__main__":
    main()
