"""Audit every prepared continuous-sign stream with protected-source exclusions."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256
from bind_c5_openasl_scale_plan import load_scale_examples
from freeze_c5_how2sign_expansion_plan import load_expansion_examples
from freeze_c5_stem_scale_plan import ROOT, load_examples

PROTECTED = {
    "how2sign_validation": ROOT / "runs/how2sign-validation-v1/selection.json",
    "how2sign_internal_holdout": ROOT / "runs/c5-how2sign-precut-expansion-holdout-v1/index.jsonl",
    "fleurs_validation": ROOT / "runs/fleurs-dev-v1/selection.json",
    "openasl_partition": ROOT / "runs/openasl-scale-v1/partition.json",
    "openasl_consumed_confirmation_sources": ROOT / "runs/openasl-v1/pilot-selection.json",
}

fresh_face_plan = ROOT / "runs/c5-fresh-face-benchmark-v1/plan.json"
if fresh_face_plan.exists():
    PROTECTED["fresh_face_development"] = fresh_face_plan
for pilot_plan in sorted((ROOT / "runs").glob("c5-face-learning-pilot-*/plan.json")):
    PROTECTED[f"face_learning_holdout:{pilot_plan.parent.name}"] = pilot_plan


def add_declared_face_protection(path: Path, groups: dict, references: set) -> None:
    plan = json.loads(path.read_text())
    snapshot = json.loads((path.parent / "source-snapshot/manifest.json").read_text())
    if snapshot["plan_sha256"] != sha256(path):
        raise ValueError("protected face plan differs from its frozen source snapshot")
    if plan["schema"] == "umi-studio-c5-fresh-face-benchmark-plan/1":
        if plan.get("training_allowed") is not False:
            raise ValueError("fresh face development must remain evaluation-only")
        rows = plan["examples"]
        content_ids = [row["video_id"] for row in rows]
    elif plan["schema"] == "umi-studio-c5-face-learning-pilot-plan/1":
        rows = [row for row in plan["examples"] if row["role"] == "pilot_source_holdout"]
        if not rows or any(row["training_allowed"] is not False for row in rows):
            raise ValueError("pilot diagnostic holdout must forbid training")
        if any(not row["source_group"].startswith("how2sign:") for row in rows):
            raise ValueError("unknown diagnostic holdout source")
        content_ids = [row["source_group"].split(":", 1)[1] for row in rows]
    else:
        raise ValueError("unknown protected face-plan schema")
    if any(re.fullmatch(r"[A-Za-z0-9_-]{11}", value) is None for value in content_ids):
        raise ValueError("protected How2Sign content identity is invalid")
    groups["how2sign"].update(content_ids)
    references.update(tuple(normalize(row["reference"])) for row in rows)


def protected_sets() -> tuple[dict[str, set[str]], set[tuple[str, ...]]]:
    how2sign = json.loads(PROTECTED["how2sign_validation"].read_text())["rows"]
    internal = [json.loads(line) for line in PROTECTED["how2sign_internal_holdout"].read_text().splitlines()]
    fleurs = json.loads(PROTECTED["fleurs_validation"].read_text())["validation"]
    partition = json.loads(PROTECTED["openasl_partition"].read_text())
    openasl = partition["sealed_confirmation_records"] + partition["unused_reserve_records"]
    pilot = json.loads(PROTECTED["openasl_consumed_confirmation_sources"].read_text())["records"]
    groups = {
        "how2sign": {row["video_id"] for row in how2sign + internal},
        "fleurs": {row["document_id"] for row in fleurs},
        "openasl": {row["youtube_id"] for row in openasl + pilot},
    }
    references = {tuple(normalize(row["reference"])) for row in how2sign + internal + fleurs + openasl + pilot}
    for name, path in PROTECTED.items():
        if name == "fresh_face_development" or name.startswith("face_learning_holdout:"):
            add_declared_face_protection(path, groups, references)
    return groups, references


def exclusion_reason(row: dict, groups: dict, references: set) -> str | None:
    if row["source"].startswith("how2sign"):
        match = re.match(r"^([A-Za-z0-9_-]{11})_", row["id"])
        if match is None:
            raise ValueError("How2Sign example cannot be bound to a source video")
        if match.group(1) in groups["how2sign"]:
            return "protected_how2sign_source"
    elif row["source"] == "fleurs":
        match = re.match(r"^(dev_document_\d+)_sentence_", row["id"])
        if match is None:
            raise ValueError("FLEURS example cannot be bound to a source document")
        if match.group(1) in groups["fleurs"]:
            return "protected_fleurs_document"
    elif row["source"] == "openasl_scale" and row["youtube_id"] in groups["openasl"]:
        return "protected_openasl_source"
    if tuple(normalize(row["reference"])) in references:
        return "protected_reference"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to((ROOT / "runs").resolve()):
        parser.error("all-source audit needs a new runs/ output")
    replay, stem, hashes = load_examples()
    expansion, expansion_hashes = load_expansion_examples()
    partition = json.loads(PROTECTED["openasl_partition"].read_text())
    scale, scale_hashes = load_scale_examples(
        ROOT / "runs/openasl-scale-v1/stream-cache-stride2.jsonl",
        ROOT / "data/openasl-scale/stream-cache-stride2", partition,
    )
    all_rows = replay + stem + expansion + scale
    if len({(row["source"], row["id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("all-source corpus repeats a source example")
    groups, references = protected_sets()
    selected = []
    excluded = []
    for row in all_rows:
        reason = exclusion_reason(row, groups, references)
        if reason:
            excluded.append({"source": row["source"], "example_id": row["id"], "reason": reason})
        else:
            selected.append(row)
    by_cache = {}
    duplicates = []
    for row in selected:
        by_cache.setdefault(row["cache_sha256"], []).append((row["source"], row["id"]))
    for digest, keys in by_cache.items():
        if len(keys) > 1:
            duplicates.append({"cache_sha256": digest, "examples": keys})
    if duplicates:
        raise ValueError("all-source corpus has exact duplicate caches requiring review")
    args.output_dir.mkdir(parents=True)
    manifest = args.output_dir / "training-streams.jsonl"
    with manifest.open("x", encoding="utf-8") as output:
        for row in sorted(selected, key=lambda item: (item["source"], item["id"])):
            record = dict(row)
            for name in ("cache", "sidecar"):
                if record.get(name):
                    record[name] = str(Path(record[name]).resolve().relative_to(ROOT))
            output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    report = {
        "schema": "umi-studio-c5-all-source-corpus-audit/1",
        "prepared_candidates": len(all_rows), "eligible_training_examples": len(selected),
        "candidate_source_counts": dict(sorted(Counter(row["source"] for row in all_rows).items())),
        "eligible_source_counts": dict(sorted(Counter(row["source"] for row in selected).items())),
        "excluded_reasons": dict(sorted(Counter(row["reason"] for row in excluded).items())),
        "excluded": excluded,
        "protected_source_counts": {name: len(values) for name, values in groups.items()},
        "protected_normalized_references": len(references),
        "exact_duplicate_cache_groups": duplicates,
        "source_hashes": {**hashes, **expansion_hashes, **scale_hashes},
        "protected_source_hashes": {name: sha256(path) for name, path in PROTECTED.items()},
        "training_manifest_sha256": sha256(manifest), "audit_source_sha256": sha256(Path(__file__)),
        "confirmation_decoded": False,
        "qualification": (
            "Every source-specific loader verifies cached streams and training provenance. "
            "The additional source and reference exclusions protect known workspace evaluation "
            "groups. Cross-encoding perceptual deduplication and publisher pretraining overlap "
            "are not independently established by this audit."
        ),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "prepared_candidates", "eligible_training_examples", "eligible_source_counts", "excluded_reasons"
    )}))


if __name__ == "__main__":
    main()
