"""Freeze a short signer-corrective continuation from the best visual LoRA blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_visual_lora_plan import SOURCE_LAYOUT, TUNING_GATE
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
SEED = 29
TARGETS = {
    "fleurs": 96,
    "1": 96,
    "2": 160,
    "3": 27,
    "4": 27,
    "5": 27,
    "8": 27,
    "9": 26,
    "11": 26,
}
RECIPE = {
    "id": "visual-lora-corrective-v1",
    "learning_rate": 0.00001,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "recovery_after_examples": 128,
    "train_examples": 512,
    "checkpoint_after_examples": [128, 256, 384, 512],
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_examples(parent: dict) -> list[dict]:
    rows = []
    seen_how2sign = set()
    for source, layout in SOURCE_LAYOUT.items():
        index = ROOT / layout["index"]
        if sha256(index) != parent["sources"][source]["index_sha256"]:
            raise ValueError(f"corrective source index changed: {source}")
        records = [json.loads(line) for line in index.read_text().splitlines()]
        if len(records) != layout["expected"]:
            raise ValueError(f"corrective source count changed: {source}")
        for record in records:
            identifier = record["example_id"] if source == "fleurs" else record["sentence_name"]
            if source != "fleurs":
                if identifier in seen_how2sign:
                    raise ValueError("corrective How2Sign source repeats an example")
                seen_how2sign.add(identifier)
            rows.append(
                {
                    "source": source,
                    "id": identifier,
                    "signer": record["signer"],
                }
            )
    return rows


def corrective_schedule(rows: list[dict], seed: int = SEED) -> list[dict]:
    groups = {key: [] for key in TARGETS}
    for row in rows:
        key = "fleurs" if row["source"] == "fleurs" else str(row["signer"])
        if key not in groups:
            raise ValueError(f"corrective source has unexpected signer: {key}")
        groups[key].append(row)
    selected = []
    for key, target in TARGETS.items():
        ordered = sorted(
            groups[key],
            key=lambda row: digest([seed, key, row["source"], row["id"]]),
        )
        if not ordered:
            raise ValueError(f"corrective schedule lacks group: {key}")
        selected.extend(ordered[index % len(ordered)] for index in range(target))
    random.Random(seed).shuffle(selected)
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("corrective schedule length differs")
    return selected


def build_plan(parent_path: Path, adapter: Path) -> dict:
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("schema") != "umi-studio-visual-lora-plan/1"
        or parent.get("tuning_gate") != TUNING_GATE
    ):
        raise ValueError("corrective parent plan differs")
    baseline_hash = sha256(BASELINE / "models/checkpoint-11625/model.safetensors")
    with safe_open(str(adapter), framework="pt", device="cpu") as source:
        metadata = source.metadata()
    if (
        metadata.get("schema") != "umi-studio-shubert-visual-lora/1"
        or metadata.get("rank") != "16"
        or metadata.get("alpha") != "32.0"
        or metadata.get("visual_last_layers") != "4"
        or metadata.get("baseline_checkpoint_sha256") != baseline_hash
    ):
        raise ValueError("corrective starting adapter differs")
    schedule = corrective_schedule(source_examples(parent))
    counts = Counter(
        "fleurs" if row["source"] == "fleurs" else str(row["signer"])
        for row in schedule
    )
    if dict(counts) != TARGETS:
        raise ValueError("corrective schedule mix differs")
    return {
        "schema": "umi-studio-visual-lora-corrective-plan/1",
        "parent_plan": str(parent_path.resolve().relative_to(ROOT)),
        "parent_plan_sha256": sha256(parent_path),
        "starting_adapter": str(adapter.resolve().relative_to(ROOT)),
        "starting_adapter_sha256": sha256(adapter),
        "baseline_checkpoint_sha256": baseline_hash,
        "rank": 16,
        "alpha": 32.0,
        "visual_last_layers": 4,
        "seed": SEED,
        "recipe": RECIPE,
        "target_counts": TARGETS,
        "schedule_sha256": digest(
            [(row["source"], row["id"]) for row in schedule]
        ),
        "tuning": parent["tuning"],
        "tuning_gate": TUNING_GATE,
        "confirmation_scored": False,
        "plan_source_sha256": sha256(Path(__file__)),
        "trainer_source_sha256": sha256(ROOT / "tools/train_visual_lora_corrective.py"),
        "miner_health_policy": "user_authorized_maintenance",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.parent_plan.resolve().is_relative_to(ROOT / "runs")
        or not args.starting_adapter.resolve().is_relative_to(ROOT / "checkpoints")
        or not args.output.resolve().is_relative_to(ROOT / "runs")
        or args.output.exists()
    ):
        parser.error("corrective plan needs workspace inputs and a new runs output")
    plan = build_plan(args.parent_plan, args.starting_adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"frozen corrective plan sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
