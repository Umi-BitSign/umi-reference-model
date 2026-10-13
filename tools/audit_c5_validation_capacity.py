"""Inventory unused, realigned public How2Sign validation clips without scoring."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from benchmark_fleurs_dev import normalize, sha256
from freeze_c5_stem_continuation_plan import ROOT


def candidate_rows(original: list[dict], aligned: list[dict], used: set[str], training_sources: set[str], seen_texts: set[tuple]) -> tuple[list[dict], Counter]:
    by_name = {row["SENTENCE_NAME"]: row for row in aligned}
    if len(by_name) != len(aligned) or {row["SENTENCE_NAME"] for row in original} != set(by_name):
        raise ValueError("validation timestamp inventories differ")
    counts = Counter()
    selected = []
    for row in original:
        if row["VIDEO_ID"] in training_sources:
            raise ValueError("official validation recording overlaps training")
        if row["VIDEO_ID"] in used:
            continue
        counts["unused_rows"] += 1
        realigned = by_name[row["SENTENCE_NAME"]]
        if any(row[key] != realigned[key] for key in ("VIDEO_ID", "VIDEO_NAME", "SENTENCE", "SENTENCE_ID")):
            raise ValueError("validation annotation identities differ")
        start, end = float(row["START"]), float(row["END"])
        a, b = float(realigned["START_REALIGNED"]), float(realigned["END_REALIGNED"])
        text = tuple(normalize(row["SENTENCE"]))
        if 2 <= end-start <= 12 and len(row["SENTENCE"].split()) >= 3 and b > a:
            counts["precut_duration_text_eligible"] += 1
            overlap = max(0., min(end,b)-max(start,a))
            if min(overlap/(end-start), overlap/(b-a)) >= .8:
                counts["precut_alignment_eligible"] += 1
        if a < 0 or not 2 <= b-a <= 12 or len(row["SENTENCE"].split()) < 3:
            continue
        counts["realigned_raw_timing_eligible"] += 1
        if text in seen_texts:
            counts["known_reference_excluded"] += 1
            continue
        match = re.search(r"-(\d+)-rgb_front$", row["VIDEO_NAME"])
        if match is None:
            raise ValueError("unexpected validation signer label")
        selected.append({"sentence_name":row["SENTENCE_NAME"], "video_name":row["VIDEO_NAME"],
                         "video_id":row["VIDEO_ID"], "signer":int(match.group(1)), "reference":row["SENTENCE"],
                         "start_sec":a, "end_sec":b, "duration_sec":b-a,
                         "requires_raw_video_recrop":True, "split":"new_public_development"})
    return selected, counts


def read_csv(path: Path) -> list[dict]:
    with path.open() as source:
        return list(csv.DictReader(source, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "runs"):
        parser.error("validation inventory needs a new runs/ output")
    inputs = {
        "original_metadata": ROOT / "data/how2sign/source/how2sign_val.csv",
        "realigned_metadata": ROOT / "data/how2sign/source/how2sign_realigned_val.csv",
        "training_metadata": ROOT / "data/how2sign/source/how2sign_train.csv",
        "used_validation_selection": ROOT / "runs/how2sign-validation-v1/selection.json",
        "training_manifest": ROOT / "runs/c5-all-source-corpus-v1/training-streams.jsonl",
        "license": ROOT / "data/how2sign/source/CC-BY-NC-4.0.txt",
    }
    original = read_csv(inputs["original_metadata"])
    previous = json.loads(inputs["used_validation_selection"].read_text())["rows"]
    used = {row["video_id"] for row in previous}
    texts = {tuple(normalize(row["reference"])) for row in previous}
    texts.update(tuple(normalize(json.loads(line)["reference"])) for line in inputs["training_manifest"].read_text().splitlines())
    training_sources = {row["VIDEO_ID"] for row in read_csv(inputs["training_metadata"])}
    candidates, counts = candidate_rows(original, read_csv(inputs["realigned_metadata"]), used, training_sources, texts)
    candidates.sort(key=lambda row: hashlib.sha256(f"c5-new-development-v1\0{row['sentence_name']}".encode()).hexdigest())
    report = {
        "schema":"umi-studio-c5-validation-capacity-audit/1",
        "input_sha256":{key:sha256(path) for key,path in inputs.items()}, "audit_source_sha256":sha256(Path(__file__)),
        "official_validation_rows":len(original), "official_validation_sources":len({r['VIDEO_ID'] for r in original}),
        "used_sources":len(used), "unused_sources":len({r['VIDEO_ID'] for r in original}-used),
        "filters":dict(counts), "fresh_reference_candidates":len(candidates),
        "candidate_source_videos":len({r['video_id'] for r in candidates}),
        "candidate_signer_counts":dict(sorted(Counter(r['signer'] for r in candidates).items())),
        "records":candidates, "predictions_inspected":False, "confirmation_decoded":False,
        "training_allowed":False,
        "qualification":"Metadata capacity only: raw source availability, member checksums, crop duration and semantic alignment still require validation before scoring. This inventory does not change existing tuning or sealed confirmation roles.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:report[key] for key in ('unused_sources','filters','fresh_reference_candidates','candidate_source_videos','candidate_signer_counts')}))


if __name__ == "__main__":
    main()
