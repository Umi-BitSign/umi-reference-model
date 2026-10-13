"""Bind protected training identities to original RGB clips, never learned feature arrays."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256
from c5_face_learning_pilot import ROOT, read_rows
from c5_full_coverage_mps import data_path, load_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("raw RGB audit requires a new private runs/ directory")
    corpus = ROOT / "runs/c5-all-source-corpus-v2"
    rows = load_training(corpus / "audit.json", corpus / "training-streams.jsonl")
    fresh_path = ROOT / "runs/fresh-lowres-v4/training-inventory.jsonl"
    fresh = {(r["source"], r["example_id"]): r for r in read_rows(fresh_path)}
    layouts = {
        "asl_stem_original": (
            "runs/asl-stem-wiki-v1/stream-cache-stride2.jsonl",
            "data/asl-stem-wiki/clips-25fps",
        ),
        "asl_stem_supplemental": (
            "runs/asl-stem-wiki-scale-v1/stream-cache-stride2.jsonl",
            "data/asl-stem-wiki/supplemental-clips-25fps",
        ),
        "openasl_scale": (
            "runs/openasl-scale-v1/stream-cache-stride2.jsonl",
            "data/openasl-scale/clips-25fps",
        ),
    }
    indexes = {
        source: {r["example_id"]: r for r in read_rows(ROOT / index)}
        for source, (index, _) in layouts.items()
    }
    hashes = {
        str(fresh_path.relative_to(ROOT)): sha256(fresh_path),
        "runs/c5-all-source-corpus-v2/audit.json": sha256(corpus / "audit.json"),
        "runs/c5-all-source-corpus-v2/training-streams.jsonl": sha256(
            corpus / "training-streams.jsonl"
        ),
        **{index: sha256(ROOT / index) for index, _ in layouts.values()},
    }
    eligible, unavailable = [], []
    for row in rows:
        source, identity = row["source"], row["id"]
        if source in layouts:
            metadata = indexes[source][identity]
            clip_id = metadata.get("clip_id", identity)
            relative = f"{layouts[source][1]}/{clip_id}.mp4"
            digest, fps = metadata["clip_sha256"], metadata["clip_fps"]
        else:
            alias = (
                "how2sign_aligned" if source in ("how2sign_new", "how2sign_original") else source
            )
            metadata = fresh.get((alias, identity))
            if metadata is None:
                unavailable.append({"source": source, "id": identity, "reason": "no_raw_binding"})
                continue
            relative, digest, fps = (
                metadata["video"],
                metadata["video_sha256"],
                metadata["video_fps"],
            )
        if tuple(normalize(metadata["reference"])) != tuple(normalize(row["reference"])):
            raise ValueError(
                f"raw RGB caption does not match protected training identity: {source}:{identity}"
            )
        clip = data_path(relative)
        if not clip.is_file():
            unavailable.append(
                {"source": source, "id": identity, "reason": "raw_clip_missing", "video": relative}
            )
            continue
        if sha256(clip) != digest:
            raise ValueError(f"raw RGB clip bytes changed: {source}:{identity}")
        if source.startswith("how2sign"):
            group = "how2sign:" + identity[:11]
        elif source == "fleurs":
            group = "fleurs:" + identity.split("_sentence_")[0]
        elif source.startswith("asl_stem"):
            group = f"asl_stem:{row['participant']}:{row['article']}"
        else:
            group = row["source_group"]
        eligible.append(
            {
                "schema": "umi-studio-tiny-rgb-training-example/1",
                "source": source,
                "example_id": identity,
                "signer": row["signer"],
                "source_group": group,
                "reference": row["reference"],
                "video": relative,
                "video_sha256": digest,
                "video_fps": fps,
                "role": "training",
                "training_allowed": True,
            }
        )
    output.mkdir(parents=True)
    inventory = output / "training-inventory.jsonl"
    with inventory.open("x") as stream:
        for row in eligible:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    byte_counts = Counter(r["video_sha256"] for r in eligible)
    report = {
        "schema": "umi-studio-tiny-rgb-corpus-audit/1",
        "approved_training_identities": len(rows),
        "available_raw_rgb_clips": len(eligible),
        "available_by_source": dict(sorted(Counter(r["source"] for r in eligible).items())),
        "unavailable": unavailable,
        "exact_duplicate_clip_hash_groups": sum(v > 1 for v in byte_counts.values()),
        "training_inventory_sha256": sha256(inventory),
        "source_inventory_sha256": hashes,
        "audit_source_sha256": sha256(Path(__file__)),
        "pretrained_features_loaded": False,
        "pretrained_weights_loaded": False,
        "pixels_decoded": False,
        "quality_verified": False,
        "qualification": (
            "Clip identity, protected source/reference exclusions and current bytes only. "
            "Eligibility metadata comes from the source audit; no learned features or "
            "teacher targets are loaded. Decoder/frame-rate validation, real-ASL fitting, "
            "perceptual deduplication, "
            "held-out generalization and independent confirmation remain required."
        ),
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "unavailable"}
            | {"unavailable_count": len(unavailable)}
        )
    )


if __name__ == "__main__":
    main()
