"""Run the single frozen baseline/candidate confirmation and decide once."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
RUNTIME = BASELINE / ".runtime/bin/python"
CHECKPOINT = BASELINE / "models/checkpoint-11625"
ADAPTER = (
    ROOT
    / "checkpoints/visual-lora-corrective-v1/visual-lora-after-000384.safetensors"
)
CACHE = ROOT / "data/temporal-stride-confirmation/stream-cache-25fps"
RUN_DIR = ROOT / "runs/temporal-stride-v1"
DOMAINS = {
    "how2sign": {
        "index": RUN_DIR / "confirmation-how2sign-stream-index.jsonl",
        "split": "validation",
        "examples": 42,
    },
    "fleurs": {
        "index": RUN_DIR / "confirmation-fleurs-stream-index.jsonl",
        "split": "confirmation",
        "examples": 46,
    },
}


def evaluation_command(domain: str, candidate: bool) -> tuple[list[str], Path]:
    config = DOMAINS[domain]
    label = "candidate-stride2" if candidate else "baseline-stride1"
    results = RUN_DIR / f"confirmation-{domain}-{label}.jsonl"
    command = [
        "nice",
        "-n",
        "20",
        str(RUNTIME),
        "tools/evaluate_fleurs_cache.py",
        "--index",
        str(config["index"]),
        "--cache-dir",
        str(CACHE),
        "--split",
        str(config["split"]),
        "--baseline",
        str(BASELINE),
        "--checkpoint",
        str(CHECKPOINT),
        "--results",
        str(results),
        "--miner-maintenance-mode",
        "--device",
        "cpu",
        "--use-stream-features",
        "--temporal-stride",
        "2" if candidate else "1",
        "--limit",
        str(config["examples"]),
    ]
    if candidate:
        insertion = command.index("--results")
        command[insertion:insertion] = ["--adapter", str(ADAPTER)]
    return command, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (
        args.output.exists()
        or plan.get("schema") != "umi-studio-temporal-stride-confirmation-plan/1"
        or sha256(args.plan) != sha256(RUN_DIR / "confirmation-plan.json")
        or sha256(ADAPTER) != plan.get("candidate_adapter_sha256")
        or sha256(CHECKPOINT / "model.safetensors") != plan.get("checkpoint_sha256")
    ):
        parser.error("confirmation plan, model, adapter or output differs")
    results = {}
    for domain in ("how2sign", "fleurs"):
        index_rows = DOMAINS[domain]["index"].read_text().splitlines()
        if len(index_rows) != DOMAINS[domain]["examples"]:
            raise ValueError(f"{domain} confirmation stream index is incomplete")
        for candidate in (False, True):
            command, result_path = evaluation_command(domain, candidate)
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            results[(domain, candidate)] = result_path
    subprocess.run(
        [
            str(RUNTIME),
            "tools/decide_temporal_stride_confirmation.py",
            "--plan",
            str(args.plan),
            "--how2sign-baseline",
            str(results[("how2sign", False)]),
            "--how2sign-candidate",
            str(results[("how2sign", True)]),
            "--fleurs-baseline",
            str(results[("fleurs", False)]),
            "--fleurs-candidate",
            str(results[("fleurs", True)]),
            "--output",
            str(args.output),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
