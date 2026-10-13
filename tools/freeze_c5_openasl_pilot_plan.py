"""Freeze a source-diverse OpenASL pilot with public-data replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import sha256
from freeze_c5_stem_continuation_plan import (
    CONFIRMATION,
    MERGED_EQUIVALENCE,
    STARTING_ADAPTER,
    TUNING,
)
from freeze_c5_stem_scale_plan import ROOT, load_examples
from interpolate_visual_adapter import DIRECT_NAMES

SEED = 49
OPENASL_INDEX = ROOT / "runs/openasl-v1/stream-cache-stride2.jsonl"
OPENASL_CACHE = ROOT / "data/openasl/stream-cache-stride2"
OPENASL_POLICY = "full_source_frame_preserve_aspect_ratio"
COUNTS = {
    "openasl": 120,
    "asl_stem_original": 64,
    "asl_stem_supplemental": 64,
    "how2sign": 192,
    "fleurs": 72,
}
RECIPE = {
    "id": "c5-openasl-diversity-projection-frozen-v1",
    "rank": 16,
    "alpha": 32.0,
    "visual_last_layers": 4,
    "learning_rate": 2e-6,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "cpu_threads": 4,
    "train_examples": 512,
    "checkpoint_after_examples": [128, 256, 512],
    "frozen_direct_parameter_names": sorted(DIRECT_NAMES),
}


def stable_key(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in (SEED, *values)).encode()
    ).hexdigest()


def load_openasl_examples(
    index: Path = OPENASL_INDEX,
    cache_dir: Path = OPENASL_CACHE,
    spatial_policy: str = OPENASL_POLICY,
) -> tuple[list[dict], dict[str, str]]:
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if len(rows) != 120 or len({row["clip_id"] for row in rows}) != 120:
        raise ValueError("OpenASL stride-2 stream index is incomplete")
    examples = []
    for row in rows:
        cache = cache_dir / f"{row['clip_id']}.npz"
        if (
            row.get("schema") != "umi-studio-openasl-stream-cache/1"
            or row.get("split") != "training"
            or row.get("temporal_stride") != 2
            or row.get("spatial_policy") != spatial_policy
            or row.get("frames") != row.get("encoder_shape", [None])[0]
            or sha256(cache) != row.get("encoder_cache_sha256")
        ):
            raise ValueError(f"OpenASL stream row differs: {row.get('clip_id')}")
        examples.append(
            {
                "id": row["clip_id"],
                "source": "openasl",
                "signer": row["source_group"],
                "source_group": row["source_group"],
                "youtube_id": row["youtube_id"],
                "reference": row["reference"],
                "cache": str(cache),
                "cache_sha256": row["encoder_cache_sha256"],
                "frames": row["frames"],
                "stored_temporal_stride": 2,
            }
        )
    if set(Counter(row["youtube_id"] for row in examples).values()) != {2}:
        raise ValueError("OpenASL pilot no longer has two clips per source")
    return examples, {"openasl_index_sha256": sha256(index)}


def balanced_stem(rows: list[dict], count: int, source: str) -> list[dict]:
    by_participant = defaultdict(list)
    for row in rows:
        by_participant[row["participant"]].append(row)
    for participant in by_participant:
        by_participant[participant].sort(
            key=lambda row: stable_key("stem", source, participant, row["id"])
        )
    participants = sorted(
        by_participant, key=lambda value: stable_key("participant", source, value)
    )
    selected = []
    position = 0
    while len(selected) < count:
        advanced = False
        for participant in participants:
            if position < len(by_participant[participant]):
                selected.append(by_participant[participant][position])
                advanced = True
                if len(selected) == count:
                    break
        if not advanced:
            raise ValueError(f"{source} cannot fill its OpenASL replay target")
        position += 1
    return selected


def select_schedule(
    replay: list[dict], stem: list[dict], openasl: list[dict]
) -> list[dict]:
    by_signer = defaultdict(list)
    fleurs = []
    for row in replay:
        if row["source"] == "fleurs":
            fleurs.append(row)
        else:
            by_signer[row["signer"]].append(row)
    if set(by_signer) != {1, 2, 3, 4, 5, 8, 9, 11}:
        raise ValueError("OpenASL replay lacks a How2Sign training signer")
    selected = []
    for signer, rows in sorted(by_signer.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_key("h", signer, row["source"], row["id"]),
        )
        if len(ordered) < 24:
            raise ValueError(f"OpenASL replay lacks How2Sign signer {signer}")
        selected.extend(ordered[:24])
    if len(fleurs) < COUNTS["fleurs"]:
        raise ValueError("OpenASL replay lacks FLEURS examples")
    selected.extend(
        sorted(fleurs, key=lambda row: stable_key("f", row["id"]))[
            : COUNTS["fleurs"]
        ]
    )
    for source in ("asl_stem_original", "asl_stem_supplemental"):
        source_rows = [row for row in stem if row["source"] == source]
        selected.extend(balanced_stem(source_rows, COUNTS[source], source))
    if (
        len(openasl) != COUNTS["openasl"]
        or len({row["source_group"] for row in openasl}) != 60
    ):
        raise ValueError("OpenASL pilot examples differ")
    selected.extend(openasl)
    if len(selected) != RECIPE["train_examples"]:
        raise ValueError("OpenASL pilot schedule count differs")
    return sorted(
        selected,
        key=lambda row: stable_key("schedule", row["source"], row["id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        (ROOT / "runs").resolve()
    ):
        parser.error("OpenASL pilot needs a new run-ledger directory")
    decision = json.loads(CONFIRMATION.read_text())
    equivalence = json.loads(MERGED_EQUIVALENCE.read_text())
    if (
        decision.get("decision") != "package_candidate"
        or decision.get("candidate_adapter_sha256") != sha256(STARTING_ADAPTER)
        or equivalence.get("all_hypotheses_equal") is not True
        or equivalence.get("examples") != 88
    ):
        raise ValueError("confirmed OpenASL starting candidate is not frozen")
    replay, stem, source_hashes = load_examples()
    openasl, openasl_hashes = load_openasl_examples()
    source_hashes.update(openasl_hashes)
    schedule = select_schedule(replay, stem, openasl)
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
        "schema": "umi-studio-c5-frozen-replay-plan/1",
        "objective": "test source-diverse OpenASL supervision with balanced public replay",
        "seed": SEED,
        "recipe": RECIPE,
        "starting_adapter": str(STARTING_ADAPTER.relative_to(ROOT)),
        "starting_adapter_sha256": sha256(STARTING_ADAPTER),
        "baseline_checkpoint_sha256": decision["how2sign"]["checkpoint_sha256"],
        "confirmation_decision_sha256": sha256(CONFIRMATION),
        "merged_equivalence_sha256": sha256(MERGED_EQUIVALENCE),
        "confirmation_policy": (
            "The consumed confirmation results are final. Do not evaluate this OpenASL "
            "pilot on either confirmation set."
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
        "openasl_source_videos": len(
            {row["source_group"] for row in schedule if row["source"] == "openasl"}
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
                "Only a passing checkpoint may justify a larger OpenASL tranche and fresh "
                "held-out collection; open tuning cannot replace the confirmed candidate."
            ),
        },
        "trainer_source_sha256": sha256(ROOT / "tools/train_c5_frozen_replay.py"),
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
