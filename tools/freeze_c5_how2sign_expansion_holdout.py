"""Freeze an unseen, signer-capped holdout for the fixed expansion candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_how2sign_expansion_plan import EXPANSION_INDEX
from freeze_c5_stem_scale_plan import ROOT

SEED = 53
PER_SIGNER_CAP = 20
PLAN = ROOT / "runs/c5-how2sign-precut-expansion-scale-v2/plan.json"
DECISION = ROOT / "runs/c5-how2sign-precut-expansion-scale-v2/tuning-decision.json"
CHECKPOINT_DIR = ROOT / "checkpoints/c5-how2sign-precut-expansion-scale-v2"
CANDIDATE_STEP = 512


def stable_key(row: dict) -> str:
    return hashlib.sha256(
        f"{SEED}\0{row['signer']}\0{row['sentence_name']}".encode()
    ).hexdigest()


def select_holdout(rows: list[dict], seen: set[str]) -> list[dict]:
    by_signer = defaultdict(list)
    for row in rows:
        if row["sentence_name"] not in seen:
            by_signer[row["signer"]].append(row)
    if set(by_signer) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("unseen expansion holdout lacks a signer")
    selected = []
    for signer, signer_rows in sorted(by_signer.items()):
        ordered = sorted(signer_rows, key=stable_key)
        selected.extend(ordered[:PER_SIGNER_CAP])
    return sorted(selected, key=lambda row: (row["signer"], stable_key(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("expansion holdout needs a new run-ledger directory")
    plan = json.loads(PLAN.read_text())
    decision = json.loads(DECISION.read_text())
    checkpoint_index = [
        json.loads(line)
        for line in (CHECKPOINT_DIR / "checkpoint-index.jsonl").read_text().splitlines()
    ]
    checkpoint_row = next(
        row for row in checkpoint_index if row.get("example_step") == CANDIDATE_STEP
    )
    candidate = CHECKPOINT_DIR / checkpoint_row["checkpoint_name"]
    if (
        plan.get("recipe", {}).get("id") != "c5-how2sign-precut-expansion-scale-v2"
        or decision.get("plan_sha256") != sha256(PLAN)
        or decision.get("decision") != "retain_confirmed_candidate"
        or decision.get("selected_step") is not None
        or decision["checkpoints"][str(CANDIDATE_STEP)]["how2sign"][
            "candidate_minus_baseline_mean_wer"
        ]
        > -0.02
        or checkpoint_row.get("checkpoint_sha256") != sha256(candidate)
    ):
        raise ValueError("expansion candidate is not fixed for internal holdout")
    seen = set()
    ledger = CHECKPOINT_DIR / "steps.jsonl"
    for row in map(json.loads, ledger.read_text().splitlines()):
        if row["example_step"] > CANDIDATE_STEP:
            continue
        seen.update(
            identifier
            for source, identifier in zip(
                row["sources"], row["example_ids"], strict=True
            )
            if source == "how2sign_precut_expansion"
        )
    source_rows = [json.loads(line) for line in EXPANSION_INDEX.read_text().splitlines()]
    selected = select_holdout(source_rows, seen)
    if not selected or {row["sentence_name"] for row in selected} & seen:
        raise ValueError("expansion holdout overlaps candidate training")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index = args.output_dir / "index.jsonl"
    with index.open("x", encoding="utf-8") as output:
        for row in selected:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    selection = {
        "schema": "umi-studio-c5-how2sign-expansion-holdout/1",
        "purpose": "internal unseen-training-split diagnostic; not promotion confirmation",
        "seed": SEED,
        "per_signer_cap": PER_SIGNER_CAP,
        "examples": len(selected),
        "signer_counts": dict(
            sorted(Counter(str(row["signer"]) for row in selected).items())
        ),
        "source_index_sha256": sha256(EXPANSION_INDEX),
        "training_plan_sha256": sha256(PLAN),
        "training_ledger_sha256": sha256(ledger),
        "candidate_step": CANDIDATE_STEP,
        "candidate_adapter": str(candidate.relative_to(ROOT)),
        "candidate_adapter_sha256": sha256(candidate),
        "excluded_seen_expansion_examples": len(seen),
        "index_sha256": sha256(index),
        "selection_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
    }
    selection_path = args.output_dir / "selection.json"
    with selection_path.open("x", encoding="utf-8") as output:
        json.dump(selection, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
