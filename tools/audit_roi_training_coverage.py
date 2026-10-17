"""Metadata-only coverage of existing authorized data; not training readiness or model quality."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256
from launch_terra_visual_job import write
from terra_training_storage import storage

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "runs/tiny-rgb-visual-teacher-corpus-v1/plan.json"
FIT = ROOT / "runs/roi-short-corpus-training-v1/plan.json"
DATA = ROOT / "checkpoints/roi-short-corpus-data-v1/data.json"
CURRICULUM = ROOT / "runs/terra-short-curriculum-v1/plan.json"


def eligible_pairs(pairs, reserved):
    eligible, excluded, known = [], Counter(), set()
    for pair in pairs:
        row, teacher = pair["raw"], pair["teacher"]
        key = (row["source"], row["example_id"])
        if (
            row["role"] != "training"
            or row["training_allowed"] is not True
            or teacher["role"] != "training"
            or teacher["training_allowed"] is not True
            or pair["feature_pretraining_allowed"] is not True
            or teacher["source"] != row["source"]
            or teacher["id"] != row["example_id"]
            or normalize(teacher["reference"]) != normalize(row["reference"])
            or key in known
        ):
            raise ValueError(
                "coverage audit requires distinct authorized aligned training metadata"
            )
        known.add(key)
        if row["source"].startswith("asl_stem") and row["signer"] in reserved:
            excluded[row["source"]] += 1
        else:
            eligible.append(pair)
    return eligible, dict(excluded)


def word_band(words):
    if words < 1:
        raise ValueError("coverage audit found an empty normalized reference")
    if words == 1:
        return "1"
    if words <= 3:
        return "2-3"
    if words <= 6:
        return "4-6"
    if words <= 10:
        return "7-10"
    return "11+"


def lexical_coverage(training, development):
    words = {word for row in training for word in normalize(row["reference"])}
    dev_words = {word for row in development for word in normalize(row["reference"])}
    references = {tuple(normalize(row["reference"])) for row in training}
    return {
        "training_word_types": len(words),
        "development_word_types": len(dev_words),
        "development_word_types_absent_from_training_references": len(dev_words - words),
        "development_clips_with_at_least_one_absent_word": sum(
            bool(set(normalize(row["reference"])) - words) for row in development
        ),
        "development_exact_normalized_references_seen": sum(
            tuple(normalize(row["reference"])) in references for row in development
        ),
        "not_ASL_sign_coverage_or_semantic_generalization": True,
    }


def audit():
    corpus, fit, data, curriculum = [
        json.loads(p.read_text()) for p in (CORPUS, FIT, DATA, CURRICULUM)
    ]
    manifest = ROOT / corpus["native_pairs"]
    if sha256(manifest) != corpus["native_pairs_sha256"]:
        raise ValueError("original approved training manifest changed")
    pairs = [json.loads(s) for s in manifest.read_text().splitlines()]
    if len(pairs) != 7523:
        raise ValueError("coverage audit cannot substitute a different native corpus")
    reserved = fit["reserved_STEM_participants"] + fit["ROI_diagnostic_participants"]
    eligible, excluded = eligible_pairs(pairs, reserved)
    training = [r["original_row"] for r in data["records"] if r["roi_training_allowed"]]
    development = [r for r in curriculum["records"] if r["role"] == "development"]
    keys = {(p["raw"]["source"], p["raw"]["example_id"]) for p in eligible}
    if (
        len(training) != 88
        or len(development) != 34
        or any((r["source"], r["example_id"]) not in keys for r in training)
    ):
        raise ValueError("coverage audit lost the original88/34 identity scope")
    bands, bounded, sources = Counter(), Counter(), Counter()
    for pair in eligible:
        row = pair["raw"]
        band = word_band(len(normalize(row["reference"])))
        bands[band] += 1
        if pair["header"]["sampled_frames"] <= 149:
            bounded[band] += 1
        sources[row["source"]] += 1
    return {
        "schema": "umi-studio-roi-training-coverage/1",
        "metadata_only": True,
        "inputs": {
            str(p.relative_to(ROOT)): sha256(p) for p in (CORPUS, manifest, FIT, DATA, CURRICULUM)
        },
        "implementation_sha256": sha256(Path(__file__)),
        "original_pairs": len(pairs),
        "eligible_after_ROI_reserved_signers": len(eligible),
        "reserved_STEM_participants": reserved,
        "excluded_reserved_by_source": excluded,
        "eligible_by_source": dict(sources),
        "eligible_by_word_band": dict(bands),
        "eligible_complete_clips_at_most149_frames_by_word_band": dict(bounded),
        "current88_reference_coverage": lexical_coverage(training, development),
        "all_eligible_reference_coverage": lexical_coverage(
            [p["raw"] for p in eligible], development
        ),
        "reference_coverage_does_not_select_training_clips": True,
        "media_or_landmark_bytes_revalidated": False,
        "training_readiness_claimed": False,
        "new_downloads": 0,
        "new_Terra_requests": 0,
        "optimizer_updates": 0,
        "model_win_claimed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("coverage audit requires a fresh owned output directory")
    storage()
    result = audit()
    output.parent.mkdir(parents=True)
    write(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
