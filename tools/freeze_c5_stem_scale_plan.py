"""Freeze a full replay plan over both stride-2 ASL STEM tranches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import (
    BASELINE,
    CONFIRMATION,
    MERGED_EQUIVALENCE,
    ROOT,
    STARTING_ADAPTER,
    TUNING,
)
from freeze_visual_lora_plan import SOURCE_LAYOUT
from interpolate_visual_adapter import DIRECT_NAMES
from visual_training_data import load_visual_examples

SEED = 44
HOW2SIGN_PER_SIGNER = 94
FLEURS_EXAMPLES = 312
STEM_LAYOUT = {
    "asl_stem_original": {
        "index": ROOT / "runs/asl-stem-wiki-v1/stream-cache-stride2.jsonl",
        "cache": ROOT / "data/asl-stem-wiki/stream-cache-stride2",
    },
    "asl_stem_supplemental": {
        "index": ROOT / "runs/asl-stem-wiki-scale-v1/stream-cache-stride2.jsonl",
        "cache": ROOT / "data/asl-stem-wiki/supplemental-stream-cache-stride2",
    },
}
RECIPE = {
    "id": "c5-stem-scale-projection-frozen-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "learning_rate": 2e-6,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "cpu_threads": 4,
    "train_examples": 3_112,
    "recovery_after_examples": 512,
    "checkpoint_after_examples": [512, 1_024, 2_048, 3_112],
    "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
}


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def load_stem_index(source: str, index: Path, cache_dir: Path) -> list[dict]:
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if len(rows) != 1_024 or len({row["example_id"] for row in rows}) != 1_024:
        raise ValueError(f"{source} stream index is incomplete")
    examples = []
    for row in rows:
        cache = cache_dir / f"{row['example_id']}.npz"
        if (
            row.get("schema") != "umi-studio-asl-stem-wiki-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or row.get("frames") != row["encoder_shape"][0]
            or sha256(cache) != row["encoder_cache_sha256"]
        ):
            raise ValueError(f"{source} stream row differs: {row.get('example_id')}")
        examples.append(
            {
                "id": row["example_id"],
                "source": source,
                "signer": row["participant"],
                "participant": row["participant"],
                "article": row["article"],
                "reference": row["reference"],
                "cache": str(cache),
                "cache_sha256": row["encoder_cache_sha256"],
                "frames": row["frames"],
                "stored_temporal_stride": 2,
            }
        )
    return examples


def select_schedule(replay: list[dict], stem: list[dict]) -> list[dict]:
    by_signer_source = defaultdict(lambda: defaultdict(list))
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            by_signer_source[row["signer"]][row["source"]].append(row)
    if set(by_signer_source) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("scale replay lacks a How2Sign signer")
    selected = []
    for signer, source_rows in sorted(by_signer_source.items()):
        for source in source_rows:
            source_rows[source].sort(
                key=lambda row: stable_key("h", signer, source, row["id"])
            )
        sources = sorted(source_rows, key=lambda value: stable_key("source", signer, value))
        pointers = Counter()
        signer_selected = []
        while len(signer_selected) < HOW2SIGN_PER_SIGNER:
            advanced = False
            for source in sources:
                pointer = pointers[source]
                if pointer < len(source_rows[source]):
                    signer_selected.append(source_rows[source][pointer])
                    pointers[source] += 1
                    advanced = True
                    if len(signer_selected) == HOW2SIGN_PER_SIGNER:
                        break
            if not advanced:
                raise ValueError(f"scale replay signer {signer} lacks unique examples")
        selected.extend(signer_selected)
    if len(fleurs) < FLEURS_EXAMPLES:
        raise ValueError("scale replay lacks FLEURS examples")
    selected.extend(sorted(fleurs, key=lambda row: stable_key("f", row["id"]))[:FLEURS_EXAMPLES])
    if (
        len(stem) != 2_048
        or len({(row["source"], row["id"]) for row in stem}) != 2_048
    ):
        raise ValueError("scale replay needs both complete STEM tranches")
    selected.extend(stem)
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("scale schedule count differs")
    return sorted(selected, key=lambda row: stable_key("schedule", row["source"], row["id"]))


def load_examples() -> tuple[list[dict], list[dict], dict[str, str]]:
    replay = []
    source_hashes = {}
    for source, layout in SOURCE_LAYOUT.items():
        index = ROOT / layout["index"]
        source_hashes[f"{source}_index_sha256"] = sha256(index)
        replay.extend(
            load_visual_examples(
                index,
                ROOT / layout["cache_dir"],
                BASELINE,
                source,
                sidecar_index=(ROOT / layout["sidecar_index"])
                if "sidecar_index" in layout
                else None,
                sidecar_dir=(ROOT / layout["sidecar_dir"])
                if "sidecar_dir" in layout
                else None,
            )
        )
    stem = []
    for source, layout in STEM_LAYOUT.items():
        source_hashes[f"{source}_index_sha256"] = sha256(layout["index"])
        stem.extend(load_stem_index(source, layout["index"], layout["cache"]))
    return replay, stem, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("scale plan needs a new run-ledger directory")
    decision = json.loads(CONFIRMATION.read_text())
    equivalence = json.loads(MERGED_EQUIVALENCE.read_text())
    if (
        decision.get("decision") != "package_candidate"
        or decision.get("candidate_adapter_sha256") != sha256(STARTING_ADAPTER)
        or equivalence.get("all_hypotheses_equal") is not True
        or equivalence.get("examples") != 88
    ):
        raise ValueError("confirmed starting candidate is not frozen")
    replay, stem, source_hashes = load_examples()
    schedule = select_schedule(replay, stem)
    schedule_rows = [
        {
            "position": position,
            "source": row["source"],
            "example_id": row["id"],
            "signer": row["signer"],
            "stored_temporal_stride": row.get("stored_temporal_stride", 1),
        }
        for position, row in enumerate(schedule)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = args.output_dir / "schedule.jsonl"
    with schedule_path.open("x", encoding="utf-8") as output:
        for row in schedule_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    plan = {
        "schema": "umi-studio-c5-stem-scale-plan/1",
        "objective": "test all 2,048 diverse STEM clips with balanced public-data replay",
        "seed": SEED,
        "recipe": RECIPE,
        "starting_adapter": str(STARTING_ADAPTER.relative_to(ROOT)),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": decision["how2sign"]["checkpoint_sha256"],
        "confirmation_decision_sha256": sha256(CONFIRMATION),
        "merged_equivalence_sha256": sha256(MERGED_EQUIVALENCE),
        "confirmation_policy": (
            "The consumed confirmation results are final. Do not evaluate this scale "
            "experiment on either confirmation set."
        ),
        "schedule_sha256": sha256(schedule_path),
        "source_counts": dict(sorted(Counter(row["source"] for row in schedule).items())),
        "how2sign_signer_counts": dict(
            sorted(
                Counter(
                    str(row["signer"])
                    for row in schedule
                    if row["source"].startswith("how2sign")
                ).items()
            )
        ),
        "asl_stem_participants": len(
            {row["participant"] for row in schedule if row["source"].startswith("asl_stem")}
        ),
        "source_hashes": source_hashes,
        "open_tuning": {
            domain: {
                "results": str(path.relative_to(ROOT)),
                "results_sha256": sha256(path),
            }
            for domain, path in TUNING.items()
        },
        "screen": {
            "how2sign_max_mean_wer_change": 0.01,
            "fleurs_max_mean_wer_change": 0.01,
            "equal_domain_max_mean_wer_change": -0.005,
            "minimum_improved_how2sign_signers": 3,
            "next_action": (
                "Only a passing checkpoint may justify fresh held-out collection; it cannot "
                "replace the confirmed candidate from open tuning alone."
            ),
        },
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_stem_scale.py"),
        "plan_source_sha256": sha256(Path(__file__)),
        "miner_health_policy": "user_authorized_maintenance",
    }
    plan_path = args.output_dir / "plan.json"
    with plan_path.open("x", encoding="utf-8") as output:
        json.dump(plan, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(
        json.dumps(
            {
                "plan_sha256": sha256(plan_path),
                "source_counts": plan["source_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
