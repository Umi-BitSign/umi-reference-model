"""One-shot source-held-out OpenASL confirmation of the frozen face policy."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

import verify_c5_face_recovery_runtime as engine
from benchmark_fleurs_dev import normalize, sha256
from c5_experiment_sources import archive_sources

ROOT = engine.ROOT
PARTITION = ROOT / "runs/openasl-scale-v1/partition.json"
CLAIM = PARTITION.parent / "face-recovery-confirmation-claim.json"
GATES = {"maximum_mean_wer_delta": 0., "maximum_word_weighted_wer_delta": 0.,
         "minimum_improved_minus_regressed": 0, "maximum_source_bootstrap_95_upper_delta": .02,
         "bootstrap_samples": 5000, "bootstrap_seed": 1729, "empty_hypotheses_allowed": 0}


def normalized(text: str) -> str:
    return " ".join(normalize(text))


def freeze(output: Path) -> None:
    output = output.resolve()
    if output.exists() or CLAIM.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("confirmation needs a new output and an unclaimed cohort")
    runtime_dir = ROOT / "runs/c5-face-runtime-validation-parallel-v1"
    runtime = json.loads((runtime_dir / "decision.json").read_text())
    parent = json.loads((runtime_dir / "plan.json").read_text())
    if not all(runtime[key] for key in ("runtime_parity_passes", "open_cross_domain_passes", "speed_keep_passes")):
        raise ValueError("the frozen candidate has not passed every development/runtime gate")
    if parent["source_sha256"] != engine.sources():
        # The runner's progress denominator was generalized after this completed
        # run. Its archived source remains authoritative; inference sources must
        # still match every component pinned by that run.
        for name, value in parent["source_sha256"].items():
            if name != "verify_c5_face_recovery_runtime.py" and sha256(ROOT / "tools" / name) != value:
                raise ValueError("confirmed inference component changed")
    partition = json.loads(PARTITION.read_text())
    sealed = partition["sealed_confirmation_records"]
    if len(sealed) != 60 or len({r["youtube_id"] for r in sealed}) != 60:
        raise ValueError("sealed source cohort differs")
    training_path = ROOT / "runs/c5-all-source-corpus-v1/training-streams.jsonl"
    training = engine.read_rows(training_path)
    texts = {normalized(row["reference"]) for row in training}
    training_sources = {row.get("source_group") for row in training}
    if any(row["source_group"] in training_sources or normalized(row["reference"]) in texts for row in sealed):
        raise ValueError("confirmation overlaps prepared training references or sources")
    sealed_ids = {row["example_id"] for row in sealed}
    # Existing feature extraction does not consume confirmation. Prior decoded
    # hypotheses anywhere in the public experiment ledger do consume it.
    for path in (ROOT / "runs").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if '"hypothesis"' not in line and '"hypotheses"' not in line: continue
            row = json.loads(line)
            if row.get("example_id") in sealed_ids or row.get("id") in sealed_ids:
                raise ValueError(f"sealed example was previously decoded: {path}")
    index_path = ROOT / "runs/openasl-scale-v1/stream-cache-stride2.jsonl"
    receipts_path = ROOT / "runs/openasl-scale-v1/landmark-receipts.jsonl"
    index = {row["example_id"]: row for row in engine.read_rows(index_path)}
    receipts = {row["example_id"]: row for row in engine.read_rows(receipts_path)}
    examples = []
    for row in sealed:
        cached = index[row["example_id"]]; receipt = receipts[row["example_id"]]
        clip = ROOT / "data/openasl-scale/clips-25fps" / f"{cached['clip_id']}.mp4"
        landmarks = ROOT / "data/openasl-scale/landmarks" / f"{cached['clip_id']}.npz"
        if cached["spatial_policy"] != "full_source_frame_preserve_aspect_ratio" or receipt["clip_fps"] != 25. or cached["source_group"] != row["source_group"] or cached["reference"] != row["reference"] or sha256(clip) != cached["clip_sha256"] or sha256(landmarks) != cached["landmarks_sha256"] or receipt["clip_sha256"] != cached["clip_sha256"]:
            raise ValueError("sealed clip identity, framing or landmark receipt differs")
        examples.append({"example_id": row["example_id"], "domain": "openasl", "source_group": row["source_group"],
            "youtube_id": row["youtube_id"], "reference": row["reference"], "clip": str(clip.relative_to(ROOT)),
            "clip_sha256": cached["clip_sha256"], "expected_hypotheses": {}, "expected_features": {},
            "expected_legacy_landmarks": {"path": str(landmarks.relative_to(ROOT)), "sha256": cached["landmarks_sha256"]}})
    plan = {**parent, "schema": "umi-studio-c5-face-openasl-confirmation-plan/1", "examples": examples,
        "repeats": 1, "gates": GATES, "source_sha256": engine.sources(), "confirmation_evaluator_sha256": sha256(Path(__file__)),
        "data_dir": str((ROOT / "data" / output.name).relative_to(ROOT)), "sealed_partition_sha256": sha256(PARTITION),
        "training_inventory_sha256": sha256(training_path), "confirmation_index_sha256": sha256(index_path),
        "confirmation_landmark_receipts_sha256": sha256(receipts_path), "runtime_decision_sha256": sha256(runtime_dir / "decision.json"),
        "candidate_frozen_before_confirmation": True, "training_allowed": False,
        "confirmation_policy": "Exactly one frozen candidate and its selected crop168 control, once each on all 60 prepartitioned source-held-out clips. No reserve clips. This cohort is consumed regardless of outcome; never tune or select another variant on it.",
        "upstream_overlap_qualification": "Source/reference disjoint from all locally prepared training and earlier local decoded hypotheses. The upstream baseline checkpoint training inventory is unknown; no upstream novelty or signer-disjointness claim."}
    output.mkdir(); path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    archive_sources(path, output / "source-snapshot")
    print(json.dumps({"confirmation_sources": len(examples), "requests": len(examples) * 2, "plan_sha256": sha256(path)}))


def summarize(records: list[dict], plan: dict) -> dict:
    if plan["gates"] != GATES or plan["repeats"] != 1 or len(records) != len(plan["examples"]) * 2:
        raise ValueError("confirmation coverage or frozen gates differ")
    indexed = {(row["example_id"], row["arm"]): row for row in records}
    if len(indexed) != len(records): raise ValueError("a sealed variant was decoded more than once")
    pairs = []; timings = []
    for example in plan["examples"]:
        before, after = [indexed[(example["example_id"], arm)] for arm in engine.ARMS]
        if before["legacy_fingerprint"] != after["legacy_fingerprint"] or before["nonface_fingerprint"] != after["nonface_fingerprint"]:
            raise ValueError("face policy changed an unrelated confirmation input")
        changed = set(after["recovered_source_indices"])
        if len(before["face_frame_sha256"]) != len(after["face_frame_sha256"]) or any(a != b for i, (a, b) in enumerate(zip(before["face_frame_sha256"], after["face_frame_sha256"])) if i * 2 not in changed):
            raise ValueError("face policy changed an unrecovered confirmation frame")
        pairs.append((before, after)); timings.append(after["end_to_end_seconds"] / before["end_to_end_seconds"])
    summaries = {side: {"examples": len(pairs), "mean_wer": statistics.fmean(pair[i]["wer"] for pair in pairs),
        "word_weighted_wer": sum(pair[i]["edits"] for pair in pairs) / sum(pair[i]["reference_words"] for pair in pairs),
        "empty_hypotheses": sum(not pair[i]["hypothesis"].strip() for pair in pairs)} for i, side in enumerate(("control", "candidate"))}
    deltas = np.asarray([after["wer"] - before["wer"] for before, after in pairs])
    rng = np.random.default_rng(GATES["bootstrap_seed"])
    sample = rng.integers(0, len(pairs), size=(GATES["bootstrap_samples"], len(pairs)))
    interval = np.quantile(deltas[sample].mean(axis=1), [.025, .975]).tolist()
    mean_delta = summaries["candidate"]["mean_wer"] - summaries["control"]["mean_wer"]
    weighted_delta = summaries["candidate"]["word_weighted_wer"] - summaries["control"]["word_weighted_wer"]
    parity = all(row["expected_legacy_parity"] for row in records)
    passes = parity and mean_delta <= 0 and weighted_delta <= 0 and int((deltas < 0).sum()) >= int((deltas > 0).sum()) and interval[1] <= .02 and summaries["candidate"]["empty_hypotheses"] == 0
    return {"schema": "umi-studio-c5-face-openasl-confirmation-decision/1", "summaries": summaries,
        "mean_wer_delta": mean_delta, "word_weighted_wer_delta": weighted_delta, "source_bootstrap_95_interval": interval,
        "paired_improved": int((deltas < 0).sum()), "paired_regressed": int((deltas > 0).sum()), "paired_tied": int((deltas == 0).sum()),
        "geometric_candidate_to_control_latency_ratio": math.exp(statistics.fmean(math.log(r) for r in timings)),
        "passes": passes, "decision": "package_research_candidate_for_broader_quality_work" if passes else "reject_confirmation_regression",
        "confirmation_decoded": True, "cohort_consumed": True, "candidate_frozen_before_confirmation": True,
        "training_allowed": False, "miner_changed": False, "promoted": False,
        "qualification": plan["upstream_overlap_qualification"], "runtime_qualification": plan["runtime_qualification"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path); group.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    else:
        plan = json.loads(args.run.read_text())
        if plan["confirmation_evaluator_sha256"] != sha256(Path(__file__)) or plan["sealed_partition_sha256"] != sha256(PARTITION) or plan["gates"] != GATES:
            raise ValueError("sealed confirmation implementation or partition changed")
        # Exclusive, durable claim prevents any later candidate from reusing this
        # source cohort, including after an interrupted or failed decoding run.
        claim = {"plan": str(args.run.resolve().relative_to(ROOT)), "plan_sha256": sha256(args.run),
                 "sources": 60, "requests": 120, "cohort_consumed": True, "retry_other_variants_allowed": False}
        with CLAIM.open("x") as output:
            output.write(json.dumps(claim, indent=2, sort_keys=True) + "\n"); output.flush()
            import os
            os.fsync(output.fileno())
        engine.summarize = summarize
        engine.run(args.run.resolve())


if __name__ == "__main__": main()
