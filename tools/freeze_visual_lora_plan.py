"""Freeze complete public training streams and a first visual-adapter schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from visual_training_data import FLEURS_REPEAT, SIGNER_REPEATS, balanced_schedule, load_visual_examples

SOURCE_LAYOUT = {
    "fleurs": {
        "index": "runs/fleurs-dev-v1/training-stream-encoder-cache.jsonl",
        "cache_dir": "data/fleurs-asl/encoder-cache-25fps/training-stream",
        "expected": 313,
    },
    "how2sign_original": {
        "index": "runs/how2sign-expanded-v1/encoder-cache-rebound.jsonl",
        "cache_dir": "data/how2sign/aligned-encoder-cache-25fps",
        "sidecar_index": "runs/how2sign-expanded-v1/visual-stream-sidecars-rebound.jsonl",
        "sidecar_dir": "data/how2sign/aligned-visual-stream-sidecars-25fps",
        "expected": 1024,
    },
    "how2sign_new": {
        "index": "runs/how2sign-expanded-v1/encoder-cache-new.jsonl",
        "cache_dir": "data/how2sign/aligned-encoder-cache-25fps",
        "expected": 2335,
    },
    "how2sign_raw": {
        "index": "runs/how2sign-raw-v1/expansion-training-index-v2.jsonl",
        "cache_dir": "data/how2sign/raw-encoder-cache-25fps",
        "sidecar_index": "runs/how2sign-raw-v1/expansion-visual-stream-sidecars-v2.jsonl",
        "sidecar_dir": "data/how2sign/raw-visual-stream-sidecars-25fps",
        "expected": 882,
    },
}
SEED = 17
RECIPE = {
    "id": "visual-lora-stage1-v1",
    "rank": 8,
    "alpha": 16,
    "visual_last_layers": 4,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "weight_decay": 0.1,
    "label_smoothing": 0.1,
    "gradient_accumulation": 4,
    "recovery_after_examples": 128,
    "train_examples": 4096,
    "checkpoint_after_examples": [512, 1024, 2048, 4096],
}
TUNING_GATE = {
    "how2sign_max_mean_wer_change": -0.05,
    "how2sign_min_improved_signers": 3,
    "fleurs_max_mean_wer_change": 0.03,
    "equal_domain_max_mean_wer_change": -0.025,
    "checkpoint_selection": "lowest passing equal-domain mean WER change, earlier step on ties",
}
MIX_GUARD = {
    "min_fleurs_fraction": 0.15,
    "max_fleurs_fraction": 0.22,
    "min_raw_fraction": 0.30,
    "min_how2sign_signer_fraction": 0.08,
    "max_how2sign_signer_fraction": 0.25,
}
TUNING_LAYOUT = {
    "fleurs": {
        "index": "runs/fleurs-dev-v1/tuning-stream-encoder-cache.jsonl",
        "baseline": "runs/fleurs-dev-v1/tuning-baseline-stream-encoder.jsonl",
        "partition": "runs/fleurs-dev-v1/evaluation-partition.json",
        "schema": "umi-studio-fleurs-cached-evaluation-example/1",
        "split": "tuning",
        "examples": 7,
    },
    "how2sign": {
        "index": "runs/how2sign-validation-v1/stream-encoder-cache.jsonl",
        "baseline": "runs/how2sign-validation-v1/baseline-stream-pilot16.jsonl",
        "partition": "runs/how2sign-validation-v1/partition.json",
        "schema": "umi-studio-how2sign-cached-evaluation-example/1",
        "split": "validation",
        "examples": 16,
    },
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def checked_stage_one_mix(scheduled: list[dict]) -> tuple[dict, dict]:
    source_counts = Counter(row["source"] for row in scheduled)
    signer_counts = Counter(row["signer"] for row in scheduled if row["source"] != "fleurs")
    total = len(scheduled)
    how2sign_total = sum(signer_counts.values())
    if (
        total != RECIPE["train_examples"]
        or set(source_counts) != set(SOURCE_LAYOUT)
        or set(signer_counts) != set(SIGNER_REPEATS)
        or not MIX_GUARD["min_fleurs_fraction"]
        <= source_counts["fleurs"] / total
        <= MIX_GUARD["max_fleurs_fraction"]
        or source_counts["how2sign_raw"] / total < MIX_GUARD["min_raw_fraction"]
        or any(
            not MIX_GUARD["min_how2sign_signer_fraction"]
            <= count / how2sign_total
            <= MIX_GUARD["max_how2sign_signer_fraction"]
            for count in signer_counts.values()
        )
    ):
        raise ValueError("visual stage-one source or signer mix is outside the frozen guard")
    return dict(sorted(source_counts.items())), dict(sorted(signer_counts.items()))


def checked_tuning(root: Path, checkpoint_hash: str) -> dict:
    tuning = {}
    for domain, layout in TUNING_LAYOUT.items():
        index = root / layout["index"]
        baseline = root / layout["baseline"]
        partition_path = root / layout["partition"]
        partition = json.loads(partition_path.read_text())
        index_rows = [json.loads(line) for line in index.read_text().splitlines()]
        result_rows = [json.loads(line) for line in baseline.read_text().splitlines()]
        identifiers = [
            row["example_id"] if domain == "fleurs" else row["sentence_name"]
            for row in index_rows
        ]
        baseline_ids = [row["example_id"] for row in result_rows]
        if domain == "fleurs":
            selection = json.loads((root / "runs/fleurs-dev-v1/selection.json").read_text())
            expected_ids = [
                row["example_id"]
                for row in selection["validation"]
                if row["document_id"] in partition["tuning_documents"]
            ]
            partition_valid = (
                partition.get("schema") == "umi-studio-fleurs-dev-evaluation-partition/1"
                and partition.get("selection_sha256")
                == sha256(root / "runs/fleurs-dev-v1/selection.json")
                and partition.get("tuning_examples") == layout["examples"]
                and identifiers == expected_ids
            )
        else:
            partition_valid = (
                partition.get("schema") == "umi-studio-how2sign-validation-partition/1"
                and partition.get("selection_sha256")
                == sha256(root / "runs/how2sign-validation-v1/selection.json")
                and identifiers == partition.get("tuning_sentence_names")
            )
        if (
            not partition_valid
            or any(
                row.get("schema")
                != (
                    "umi-studio-fleurs-dev-encoder-cache-example/1"
                    if domain == "fleurs"
                    else "umi-studio-how2sign-encoder-cache-example/1"
                )
                or row.get("split") != layout["split"]
                for row in index_rows
            )
            or len(index_rows) != layout["examples"]
            or len(result_rows) != layout["examples"]
            or len(set(identifiers)) != len(identifiers)
            or identifiers != baseline_ids
            or any(
                row.get("schema") != layout["schema"]
                or row.get("split") != layout["split"]
                or row.get("evaluation_input") != "visual_streams"
                or row.get("adapter_sha256") is not None
                or row.get("checkpoint_sha256") != checkpoint_hash
                or row.get("index_sha256") != sha256(index)
                or row.get("reference") != source["reference"]
                or row.get("signer") != source["signer"]
                or row.get("selection_sha256") != source["selection_sha256"]
                or not math.isfinite(row.get("single_reference_wer", float("nan")))
                or row["single_reference_wer"] < 0
                for row, source in zip(result_rows, index_rows, strict=True)
            )
        ):
            raise ValueError(f"visual {domain} tuning baseline is incomplete or changed")
        tuning[domain] = {
            "index": layout["index"],
            "index_sha256": sha256(index),
            "baseline": layout["baseline"],
            "baseline_sha256": sha256(baseline),
            "partition": layout["partition"],
            "partition_sha256": sha256(partition_path),
            "examples": len(result_rows),
            "baseline_mean_single_reference_wer": statistics.fmean(
                row["single_reference_wer"] for row in result_rows
            ),
        }
    return tuning


def checked_fleurs_fidelity(root: Path) -> dict:
    path = root / "runs/fleurs-dev-v1/training-stream-fidelity.json"
    record = json.loads(path.read_text())
    if (
        record.get("schema") != "umi-studio-fleurs-training-stream-fidelity/1"
        or record.get("examples_verified") != 313
        or record.get("complete") is not True
        or record.get("selection_sha256")
        != sha256(root / "runs/fleurs-dev-v1/selection.json")
        or record.get("original_index_sha256")
        != sha256(root / "runs/fleurs-dev-v1/training-encoder-cache.jsonl")
        or record.get("stream_index_sha256")
        != sha256(root / SOURCE_LAYOUT["fleurs"]["index"])
        or record.get("verifier_source_sha256")
        != sha256(root / "tools/verify_fleurs_training_stream_fidelity.py")
        or record.get("original_cache_dir")
        != str((root / "data/fleurs-asl/encoder-cache-25fps/training").resolve())
        or record.get("stream_cache_dir")
        != str((root / SOURCE_LAYOUT["fleurs"]["cache_dir"]).resolve())
    ):
        raise ValueError("FLEURS training streams lack complete frozen-state fidelity proof")
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def checked_partition(
    root: Path,
    examples: dict[str, list[dict]],
    index_hashes: dict[str, str],
) -> dict[str, str]:
    fleurs = json.loads((root / "runs/fleurs-dev-v1/selection.json").read_text())
    expanded = json.loads((root / "runs/how2sign-expanded-v1/selection.json").read_text())
    raw = json.loads((root / "runs/how2sign-raw-v1/expansion-selection-v2.json").read_text())
    validation = json.loads((root / "runs/how2sign-validation-v1/selection.json").read_text())
    receipt_path = root / "runs/how2sign-expanded-v1/clip-receipts.jsonl"
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    fleurs_rows = {row["example_id"]: row for row in fleurs["training"]}
    expanded_rows = {row["sentence_name"]: row for row in expanded["rows"]}
    raw_rows = {
        clip["sentence_name"]: (video, clip)
        for video in raw["videos"]
        for clip in video["clips"]
    }
    fleurs_ids = {row["example_id"] for row in fleurs["training"]}
    h_ids = {row["id"] for key in ("how2sign_original", "how2sign_new") for row in examples[key]}
    receipt_ids = {row["sentence_name"] for row in receipts}
    raw_ids = {clip["sentence_name"] for video in raw["videos"] for clip in video["clips"]}
    if (
        fleurs.get("schema") != "umi-studio-fleurs-dev-split/1"
        or expanded.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or raw.get("schema") != "umi-studio-how2sign-raw-training-expansion/2"
        or raw.get("training_only") is not True
        or validation.get("split") != "validation"
        or len(fleurs_ids) != 313
        or len(fleurs["training"]) != 313
        or len(receipt_ids) != 3359
        or len(receipts) != 3359
        or len(expanded_rows) != 3389
        or len(expanded["rows"]) != 3389
        or len(raw_ids) != 882
        or raw["selected_clip_count"] != 882
        or not receipt_ids.issubset(expanded_rows)
        or {row["id"] for row in examples["fleurs"]} != fleurs_ids
        or h_ids != receipt_ids
        or {row["id"] for row in examples["how2sign_raw"]} != raw_ids
    ):
        raise ValueError("visual training indexes do not cover the frozen public selections")
    if any(
        row["reference"] != fleurs_rows[row["id"]]["reference"]
        or row["signer"] != fleurs_rows[row["id"]]["signer"]
        for row in examples["fleurs"]
    ) or any(
        row["reference"] != expanded_rows[row["id"]]["reference"]
        or row["signer"] != expanded_rows[row["id"]]["signer"]
        for source in ("how2sign_original", "how2sign_new")
        for row in examples[source]
    ) or any(
        row["reference"] != raw_rows[row["id"]][1]["reference"]
        or row["signer"] != raw_rows[row["id"]][0]["signer"]
        for row in examples["how2sign_raw"]
    ):
        raise ValueError("visual training transcript or signer differs from frozen selection")
    expanded_hash = sha256(root / "runs/how2sign-expanded-v1/selection.json")
    fleurs_hash = sha256(root / "runs/fleurs-dev-v1/selection.json")
    if any(receipt.get("selection_sha256") != expanded_hash for receipt in receipts):
        raise ValueError("visual How2Sign receipts use another frozen selection")
    for source, selection_hash in (
        ("fleurs", fleurs_hash),
        ("how2sign_original", expanded_hash),
        ("how2sign_new", expanded_hash),
    ):
        path = root / SOURCE_LAYOUT[source]["index"]
        if sha256(path) != index_hashes[source] or any(
            json.loads(line)["selection_sha256"] != selection_hash
            for line in path.read_text().splitlines()
        ):
            raise ValueError(f"visual {source} uses another frozen selection")
    training_videos = {row["video_id"] for row in expanded["rows"]}
    raw_videos = {video["video_id"] for video in raw["videos"]}
    validation_videos = {row["video_id"] for row in validation["rows"]}
    if (
        training_videos & raw_videos
        or training_videos & validation_videos
        or raw_videos & validation_videos
    ):
        raise ValueError("visual training and validation source videos overlap")
    return {
        "fleurs_selection_sha256": fleurs_hash,
        "how2sign_selection_sha256": expanded_hash,
        "how2sign_clip_receipts_sha256": sha256(receipt_path),
        "how2sign_raw_selection_sha256": sha256(
            root / "runs/how2sign-raw-v1/expansion-selection-v2.json"
        ),
        "how2sign_validation_selection_sha256": sha256(
            root / "runs/how2sign-validation-v1/selection.json"
        ),
    }


def freeze_plan(root: Path, baseline: Path, health_url: str) -> dict:
    revision = miner_revision(health_url, None)
    examples = {}
    sources = {}
    index_hashes = {}
    for source, layout in SOURCE_LAYOUT.items():
        index = root / layout["index"]
        cache_dir = root / layout["cache_dir"]
        sidecar_index = root / layout["sidecar_index"] if "sidecar_index" in layout else None
        sidecar_dir = root / layout["sidecar_dir"] if "sidecar_dir" in layout else None
        examples[source] = load_visual_examples(
            index,
            cache_dir,
            baseline,
            source,
            sidecar_index=sidecar_index,
            sidecar_dir=sidecar_dir,
            miner_health_url=health_url,
        )
        if len(examples[source]) != layout["expected"]:
            raise ValueError(f"visual {source} training cache coverage is incomplete")
        index_hashes[source] = sha256(index)
        sources[source] = {
            "index": layout["index"],
            "index_sha256": index_hashes[source],
            "cache_dir": layout["cache_dir"],
            "sidecar_index": layout.get("sidecar_index"),
            "sidecar_index_sha256": sha256(sidecar_index) if sidecar_index else None,
            "sidecar_dir": layout.get("sidecar_dir"),
            "examples": len(examples[source]),
            "signer_counts": dict(sorted(Counter(row["signer"] for row in examples[source]).items())),
        }
        revision = miner_revision(health_url, revision)
    partition = checked_partition(root, examples, index_hashes)
    fleurs_fidelity = checked_fleurs_fidelity(root)
    ordered = balanced_schedule([row for part in examples.values() for row in part], SEED)
    if len(ordered) < RECIPE["train_examples"]:
        raise ValueError("visual balanced schedule is shorter than the stage-one budget")
    scheduled = ordered[: RECIPE["train_examples"]]
    source_counts, scheduled_counts = checked_stage_one_mix(scheduled)
    revision = miner_revision(health_url, revision)
    checkpoint_hash = sha256(baseline / "models/checkpoint-11625/model.safetensors")
    tuning = checked_tuning(root, checkpoint_hash)
    return {
        "schema": "umi-studio-visual-lora-plan/1",
        "objective": "first source-disjoint, signer-balanced visual-adapter tuning gate",
        "sources": sources,
        "partition": partition,
        "fleurs_training_stream_fidelity": fleurs_fidelity,
        "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
        "baseline_runtime_sha256": sha256(baseline / "runtime.py"),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "tuning": tuning,
        "fleurs_tuning_baseline_sha256": tuning["fleurs"]["baseline_sha256"],
        "how2sign_tuning_baseline_sha256": tuning["how2sign"]["baseline_sha256"],
        "baseline_mean_single_reference_wer": {
            domain: record["baseline_mean_single_reference_wer"]
            for domain, record in tuning.items()
        },
        "visual_adapter_source_sha256": sha256(root / "tools/shubert_visual_lora.py"),
        "loader_source_sha256": sha256(root / "tools/visual_training_data.py"),
        "plan_source_sha256": sha256(Path(__file__)),
        "trainer_source_sha256": sha256(root / "tools/train_visual_lora.py"),
        "tuning_gate": TUNING_GATE,
        "tuning_gate_source_sha256": sha256(root / "tools/decide_aligned_mixed.py"),
        "checkpoint_decision_source_sha256": sha256(root / "tools/decide_visual_lora.py"),
        "seed": SEED,
        "fleurs_repeat": FLEURS_REPEAT,
        "how2sign_signer_repeats": SIGNER_REPEATS,
        "stage_one_mix_guard": MIX_GUARD,
        "full_schedule_examples": len(ordered),
        "stage_one_schedule_sha256": _digest([(row["source"], row["id"]) for row in scheduled]),
        "stage_one_source_counts": source_counts,
        "stage_one_how2sign_signer_counts": scheduled_counts,
        "recipe": RECIPE,
        "miner_model_revision": revision,
        "confirmation_scored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if (
        baseline != root / "artifacts/baseline-6fe8df59"
        or args.output.exists()
        or not args.output.resolve().is_relative_to(root / "runs")
    ):
        parser.error("plan needs the independent baseline and a new run-ledger output")
    plan = freeze_plan(root, baseline, args.miner_health_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    if temporary.exists():
        parser.error("unfinished visual plan needs inspection")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(plan, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"frozen visual-adapter plan sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
