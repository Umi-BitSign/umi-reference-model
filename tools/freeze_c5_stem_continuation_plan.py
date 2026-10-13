"""Freeze a gentle stride-2 ASL STEM continuation with public-data replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_visual_lora_plan import SOURCE_LAYOUT
from visual_training_data import load_visual_examples

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
STEM_INDEX = ROOT / "runs/asl-stem-wiki-v1/stream-cache-stride2.jsonl"
STEM_CACHE = ROOT / "data/asl-stem-wiki/stream-cache-stride2"
STARTING_ADAPTER = (
    ROOT
    / "checkpoints/visual-lora-corrective-v1/visual-lora-after-000384.safetensors"
)
CONFIRMATION = ROOT / "runs/temporal-stride-v1/confirmation-decision.json"
MERGED_EQUIVALENCE = ROOT / "runs/temporal-stride-v1/merged-export-equivalence.json"
TUNING = {
    "how2sign": ROOT / "runs/temporal-stride-v1/corrective384-how2sign-stride2.jsonl",
    "fleurs": ROOT / "runs/temporal-stride-v1/corrective384-fleurs-stride2.jsonl",
}
SEED = 43
COUNTS = {"asl_stem": 256, "how2sign": 192, "fleurs": 64}
RECIPE = {
    "id": "c5-stem-continuation-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "learning_rate": 2e-6,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "train_examples": 512,
    "recovery_after_examples": 128,
    "checkpoint_after_examples": [128, 256, 384, 512],
}


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def select_schedule(replay: list[dict], stem: list[dict]) -> list[dict]:
    by_signer = defaultdict(list)
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            by_signer[row["signer"]].append(row)
    if set(by_signer) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("How2Sign replay lacks a training signer")
    selected = []
    for signer, rows in sorted(by_signer.items()):
        ordered = sorted(rows, key=lambda row: stable_key("h", signer, row["source"], row["id"]))
        if len(ordered) < 24:
            raise ValueError(f"How2Sign replay signer {signer} has fewer than 24 clips")
        selected.extend(ordered[:24])
    selected.extend(sorted(fleurs, key=lambda row: stable_key("f", row["id"]))[:64])
    stem_by_participant = defaultdict(list)
    for row in stem:
        stem_by_participant[row["participant"]].append(row)
    if len(stem_by_participant) != 36:
        raise ValueError("ASL STEM replay lacks a participant")
    for participant in stem_by_participant:
        stem_by_participant[participant].sort(
            key=lambda row: stable_key("s", participant, row["article"], row["id"])
        )
    pointers = Counter()
    participants = sorted(stem_by_participant, key=lambda value: stable_key("p", value))
    stem_selected = []
    while len(stem_selected) < COUNTS["asl_stem"]:
        for participant in participants:
            if len(stem_selected) >= COUNTS["asl_stem"]:
                break
            pointer = pointers[participant]
            if pointer >= len(stem_by_participant[participant]):
                continue
            stem_selected.append(stem_by_participant[participant][pointer])
            pointers[participant] += 1
    if len(stem_selected) != COUNTS["asl_stem"]:
        raise ValueError("ASL STEM selection cannot fill its target")
    selected.extend(stem_selected)
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("continuation schedule count differs")
    return sorted(
        selected,
        key=lambda row: stable_key("schedule", row["source"], row["id"]),
    )


def load_stem() -> list[dict]:
    rows = [json.loads(line) for line in STEM_INDEX.read_text().splitlines()]
    if len(rows) != 1024 or len({row["example_id"] for row in rows}) != 1024:
        raise ValueError("ASL STEM stream index is incomplete")
    examples = []
    for row in rows:
        cache = STEM_CACHE / f"{row['example_id']}.npz"
        if (
            row.get("schema") != "umi-studio-asl-stem-wiki-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or row.get("frames") != row["encoder_shape"][0]
            or sha256(cache) != row["encoder_cache_sha256"]
        ):
            raise ValueError(f"ASL STEM stream row differs: {row.get('example_id')}")
        examples.append(
            {
                "id": row["example_id"],
                "source": "asl_stem",
                "signer": row["participant"],
                "participant": row["participant"],
                "article": row["article"],
                "reference": row["reference"],
                "cache": str(cache),
                "cache_sha256": row["encoder_cache_sha256"],
                "sidecar": None,
                "sidecar_sha256": None,
                "frames": row["frames"],
                "stored_temporal_stride": 2,
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("continuation plan needs a new run-ledger directory")
    decision = json.loads(CONFIRMATION.read_text())
    equivalence = json.loads(MERGED_EQUIVALENCE.read_text())
    if (
        decision.get("decision") != "package_candidate"
        or decision.get("additional_confirmation_variants_evaluated") is not False
        or decision.get("candidate_adapter_sha256") != sha256(STARTING_ADAPTER)
        or equivalence.get("all_hypotheses_equal") is not True
        or equivalence.get("examples") != 88
    ):
        raise ValueError("confirmed starting candidate is not frozen")
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
    stem = load_stem()
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
    source_counts = Counter(row["source"] for row in schedule)
    plan = {
        "schema": "umi-studio-c5-stem-continuation-plan/1",
        "objective": "test ASL STEM supervision without sacrificing the confirmed C5 gains",
        "seed": SEED,
        "recipe": RECIPE,
        "starting_adapter": str(STARTING_ADAPTER.relative_to(ROOT)),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": decision["how2sign"]["checkpoint_sha256"],
        "confirmation_decision_sha256": sha256(CONFIRMATION),
        "merged_equivalence_sha256": sha256(MERGED_EQUIVALENCE),
        "confirmation_policy": (
            "The consumed 42-clip How2Sign and 46-clip FLEURS confirmation results are final. "
            "Do not evaluate this experimental continuation on either confirmation set."
        ),
        "schedule_sha256": sha256(schedule_path),
        "source_counts": dict(sorted(source_counts.items())),
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
            {row["participant"] for row in schedule if row["source"] == "asl_stem"}
        ),
        "source_hashes": {
            **source_hashes,
            "asl_stem_index_sha256": sha256(STEM_INDEX),
        },
        "open_tuning": {
            domain: {
                "results": str(path.relative_to(ROOT)),
                "results_sha256": sha256(path),
                "mean_wer": sum(
                    row["single_reference_wer"]
                    for row in map(json.loads, path.read_text().splitlines())
                )
                / len(path.read_text().splitlines()),
            }
            for domain, path in TUNING.items()
        },
        "screen": {
            "how2sign_max_mean_wer_change": 0.01,
            "fleurs_max_mean_wer_change": 0.01,
            "equal_domain_max_mean_wer_change": -0.005,
            "minimum_improved_how2sign_signers": 3,
            "next_action": (
                "Only a passing checkpoint may justify a larger continuation and fresh held-out "
                "collection; it cannot replace the confirmed candidate from open tuning alone."
            ),
        },
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_stem_continuation.py"),
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
            {"schedule": dict(source_counts), "plan_sha256": sha256(plan_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
