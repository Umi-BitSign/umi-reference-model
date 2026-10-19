"""Reserve fresh confirmation metadata without media access or model predictions.

Official split names do not imply source-video independence. All local training
and earlier cohort identities/references are excluded before deterministic selection.
"""

# ruff: noqa: UP017 (pinned Python3.10 runtime)

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = "c5-broad-semantic-confirmation-20261019-v1"
SIGNERS = (1, 2, 3, 5)
GROUPS_PER_SIGNER = 3
CLIPS_PER_GROUP = 2
OPENASL_PRIMARY = 24
OPENASL_FALLBACK = 24
MAX_COMPRESSED = 256 * 1024**2
MAX_RAW_SOURCE = 512 * 1024**2
BUDGET = 8 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def normalized(text):
    return " ".join(normalize(text))


def key(*parts):
    return hashlib.sha256("\0".join((SEED, *map(str, parts))).encode()).hexdigest()


def protections(value, sources, texts):
    """Extract identity/reference fields only; never use any model-output fields."""
    if isinstance(value, dict):
        for name, item in value.items():
            if name in ("youtube_id", "yid", "video_id", "VIDEO_ID") and isinstance(item, str):
                sources.add(item)
            elif name == "source_group" and isinstance(item, str):
                sources.add(item.rsplit(":", 1)[-1])
            elif name in (
                "reference",
                "normalized_reference",
                "weak_clip_caption",
                "SENTENCE",
            ) and isinstance(item, str):
                text = normalized(item)
                if text:
                    texts.add(text)
            elif isinstance(item, dict | list):
                protections(item, sources, texts)
    elif isinstance(value, list):
        for item in value:
            protections(item, sources, texts)


def how2sign_pool(candidates, recordings, excluded_sources, excluded_texts):
    by_name = {r["sentence_name"]: r for r in candidates}
    require(len(by_name) == len(candidates), "duplicate How2Sign sentence identity")
    seen = set()
    pool = []
    for recording in recordings:
        name = recording["video_name"]
        require(name not in seen, "duplicate How2Sign recording")
        seen.add(name)
        if recording["video_id"] in excluded_sources or recording["signer"] not in SIGNERS:
            continue
        if not (
            0 < recording["compressed_size"] <= MAX_COMPRESSED
            and 0 < recording["file_size"] <= MAX_RAW_SOURCE
        ):
            continue
        clips = []
        for sentence in recording["candidate_clips"]:
            require(sentence in by_name, "How2Sign recording contains an unknown sentence")
            row = by_name[sentence]
            require(
                row["video_name"] == name
                and row["video_id"] == recording["video_id"]
                and row["signer"] == recording["signer"],
                "How2Sign recording mixes identities",
            )
            text = normalized(row["reference"])
            if text in excluded_texts or not 3 <= len(text.split()) <= 30:
                continue
            require(
                0 <= row["start_sec"] < row["end_sec"]
                and 2 <= row["end_sec"] - row["start_sec"] <= 12,
                "How2Sign realigned timing changed",
            )
            clips.append(row)
        clips.sort(key=lambda row: key("how2sign-clip", name, row["sentence_name"]))
        unique = []
        references = set()
        for row in clips:
            text = normalized(row["reference"])
            if text not in references:
                unique.append(row)
                references.add(text)
        if len(unique) >= CLIPS_PER_GROUP:
            pool.append({"recording": recording, "clips": unique})
    return sorted(pool, key=lambda row: key("how2sign-source", row["recording"]["video_name"]))


def choose_how2sign(pool):
    """Match signer slots to distinct content groups, including cross-signer duplicates."""
    choices = {
        signer: [r for r in pool if r["recording"]["signer"] == signer] for signer in SIGNERS
    }
    owners = {}
    assigned = {}

    def augment(slot, visited):
        for row in choices[slot[0]]:
            group = row["recording"]["video_id"]
            if group in visited:
                continue
            visited.add(group)
            prior = owners.get(group)
            if prior is None or augment(prior, visited):
                owners[group] = slot
                assigned[slot] = row
                return True
        return False

    slots = [(signer, i) for signer in SIGNERS for i in range(GROUPS_PER_SIGNER)]
    for slot in sorted(slots, key=lambda s: (len(choices[s[0]]), s)):
        require(augment(slot, set()), "fresh How2Sign cohort lacks all signer/content groups")
    selected = [assigned[slot] for slot in slots]
    references = set()
    clips = []
    for source in selected:
        chosen = []
        for row in source["clips"]:
            text = normalized(row["reference"])
            if text in references:
                continue
            chosen.append(row)
            references.add(text)
            if len(chosen) == CLIPS_PER_GROUP:
                break
        require(len(chosen) == CLIPS_PER_GROUP, "fresh How2Sign cohort repeats references")
        clips.extend(chosen)
    require(
        len({r["recording"]["video_id"] for r in selected}) == len(slots),
        "fresh How2Sign cohort repeats content groups",
    )
    return selected, clips


def seconds(value):
    require(isinstance(value, str), "invalid OpenASL timestamp")
    parts = value.split(":")
    require(len(parts) == 3, "invalid OpenASL timestamp")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    require(h >= 0 and 0 <= m < 60 and math.isfinite(s) and 0 <= s < 60, "invalid OpenASL clock")
    return h * 3600 + m * 60 + s


def openasl_pool(rows, bboxes, excluded_sources, excluded_texts):
    counts = Counter(row["vid"] for row in rows)
    result = defaultdict(list)
    for row in rows:
        if row["split"] != "valid" or counts[row["vid"]] != 1 or row["yid"] in excluded_sources:
            continue
        try:
            start, end = seconds(row["start"]), seconds(row["end"])
        except (ValueError, TypeError):
            continue
        text = normalized(row["raw-text"])
        bbox = bboxes.get(row["vid"])
        if (
            not 2 <= end - start <= 12
            or not 3 <= len(text.split()) <= 30
            or text in excluded_texts
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in bbox)
            or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])
        ):
            continue
        result[row["yid"]].append(
            {
                "example_id": row["vid"],
                "youtube_id": row["yid"],
                "reference": row["raw-text"],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "bbox": bbox,
                "official_split": "valid",
            }
        )
    return {
        source: sorted(records, key=lambda r: key("openasl-clip", r["example_id"]))
        for source, records in result.items()
    }


def choose_openasl(pool, excluded_texts):
    chosen = []
    seen = set(excluded_texts)
    for source in sorted(pool, key=lambda s: key("openasl-source", s)):
        row = next((r for r in pool[source] if normalized(r["reference"]) not in seen), None)
        if row is None:
            continue
        chosen.append(row)
        seen.add(normalized(row["reference"]))
        if len(chosen) == OPENASL_PRIMARY + OPENASL_FALLBACK:
            break
    require(
        len(chosen) == OPENASL_PRIMARY + OPENASL_FALLBACK,
        "fresh OpenASL cohort lacks source-disjoint primary/fallback coverage",
    )
    return chosen[:OPENASL_PRIMARY], chosen[OPENASL_PRIMARY:]


def freeze(output):
    output = output.absolute()
    require(
        output.resolve() == output and output.parent == ROOT / "runs" and not output.exists(),
        "confirmation reservation requires a fresh owned runs directory",
    )
    files = {
        "capacity": ROOT / "runs/c5-reassessment-v1/validation-capacity.json",
        "archive": ROOT / "runs/c5-reassessment-v1/raw-validation-archive-v2.json",
        "how2sign_train": ROOT / "data/how2sign/source/how2sign_train.csv",
        "how2sign_original": ROOT / "data/how2sign/source/how2sign_val.csv",
        "how2sign_aligned": ROOT / "data/how2sign/source/how2sign_realigned_val.csv",
        "openasl": ROOT / "data/openasl/repository/data/openasl-v1.0.tsv",
        "bboxes": ROOT / "data/openasl/repository/data/bbox-v1.0.json",
        "how2sign_license": ROOT / "data/how2sign/source/CC-BY-NC-4.0.txt",
        "openasl_license": ROOT / "data/openasl/repository/LICENSE.md",
    }
    exclusions = [
        ROOT / name
        for name in (
            "runs/how2sign-validation-v1/selection.json",
            "runs/c5-fresh-development-v1/plan.json",
            "runs/c5-fresh-development-v2/plan.json",
            "runs/openasl-v1/source-candidates-256.json",
            "runs/openasl-v1/pilot-selection.json",
            "runs/openasl-scale-v1/selection.json",
            "runs/fleurs-dev-v1/selection.json",
            "runs/c5-all-source-corpus-v1/training-streams.jsonl",
            "runs/c5-all-source-corpus-v2/training-streams.jsonl",
            "runs/c5-polarity-word-http502-recovery-v1/plan.json",
        )
    ]
    exclusions.extend(sorted((ROOT / "runs").glob("fresh-*/training-inventory*.jsonl")))
    exclusions.extend(sorted((ROOT / "runs").glob("fresh-*/tuning-inventory*.jsonl")))
    require(
        all(p.is_file() for p in (*files.values(), *exclusions)),
        "confirmation metadata input missing",
    )
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in (*files.values(), *exclusions)}
    capacity = json.loads(files["capacity"].read_text())
    archive = json.loads(files["archive"].read_text())
    require(
        archive["candidate_inventory_sha256"] == sha256(files["capacity"]),
        "How2Sign archive is bound to another candidate inventory",
    )
    for label, target in (
        ("training_metadata", "how2sign_train"),
        ("original_metadata", "how2sign_original"),
        ("realigned_metadata", "how2sign_aligned"),
        ("license", "how2sign_license"),
    ):
        require(
            capacity["input_sha256"][label] == sha256(files[target]),
            "How2Sign official metadata changed",
        )
    sources = set()
    texts = set()
    for p in exclusions:
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                protections(json.loads(line), sources, texts)
        else:
            protections(json.loads(p.read_text()), sources, texts)
    with files["how2sign_train"].open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sources.add(row["VIDEO_ID"])
            texts.add(normalized(row["SENTENCE"]))
    with files["openasl"].open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # Exclude every official training caption, but only locally used video IDs.
    # Official valid and train share videos; report that overlap explicitly.
    texts.update(normalized(r["raw-text"]) for r in rows if r["split"] == "train")
    pool = how2sign_pool(capacity["records"], archive["sources"], sources, texts)
    selected, how_clips = choose_how2sign(pool)
    all_how_sources = {r["recording"]["video_id"] for r in pool}
    all_how_texts = {normalized(c["reference"]) for r in pool for c in r["clips"]}
    open_pool = openasl_pool(
        rows,
        json.loads(files["bboxes"].read_text()),
        sources | all_how_sources,
        texts | all_how_texts,
    )
    primary, fallback = choose_openasl(open_pool, texts | all_how_texts)
    train_ids = {r["yid"] for r in rows if r["split"] == "train"}
    valid_ids = {r["yid"] for r in rows if r["split"] == "valid"}
    selected_groups = {r["recording"]["video_id"] for r in selected}
    raw_bytes = sum(r["recording"]["file_size"] for r in selected)
    require(raw_bytes <= 4 * 1024**3, "primary How2Sign raw media exceed prospective4GiB cap")
    records = [
        {
            "family": "how2sign",
            "source_group": "how2sign:" + r["video_id"],
            "metadata": r,
            "training_allowed": False,
        }
        for r in how_clips
    ]
    records.extend(
        {
            "family": "openasl",
            "source_group": "openasl:" + r["youtube_id"],
            "metadata": r,
            "training_allowed": False,
        }
        for r in primary
    )
    require(len(records) == 48, "confirmation reservation lost primary coverage")
    require(
        all(sha256(ROOT / p) == digest for p, digest in inputs.items()),
        "confirmation input changed",
    )
    require(
        shutil.disk_usage(ROOT).free >= 80 * 1024**3 + 4 * 1024**3 + 128 * 1024**2 + BUDGET,
        "confirmation reservation storage guard failed",
    )
    plan = {
        "schema": "umi-studio-broad-semantic-confirmation-metadata-reservation/1",
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": inputs,
        "source_sha256": {
            "tools/freeze_c5_broad_confirmation.py": sha256(Path(__file__)),
            "tools/benchmark_fleurs_dev.py": sha256(ROOT / "tools/benchmark_fleurs_dev.py"),
        },
        "primary_records": records,
        "how2sign_primary_recordings": [r["recording"] for r in selected],
        "how2sign_fallback_pool": [
            r for r in pool if r["recording"]["video_id"] not in selected_groups
        ],
        "openasl_fallback_records": fallback,
        "reserved_source_groups": sorted(
            ["how2sign:" + s for s in all_how_sources]
            + ["openasl:" + r["youtube_id"] for r in (*primary, *fallback)]
        ),
        "protected_source_count": len(sources),
        "protected_normalized_reference_count": len(texts),
        "protected_references_sha256": hashlib.sha256(
            "\n".join(sorted(texts)).encode()
        ).hexdigest(),
        "official_OpenASL_valid_sources": len(valid_ids),
        "official_OpenASL_valid_train_shared_sources": len(valid_ids & train_ids),
        "planned_How2Sign_raw_source_bytes": raw_bytes,
        "candidate_bound": False,
        "media_available_or_timing_validated": False,
        "model_predictions_inspected": False,
        "confirmation_consumed": False,
        "training_allowed": False,
        "media_policy": (
            "Verify all media and deterministic fallback assignments before predictions; "
            "preserve family/signer/group coverage. No post-prediction replacement."
        ),
        "candidate_policy": (
            "One fixed candidate after complete audited training, both full calibrated "
            "development gates and actual native compatibility checks. "
            "Consume on first confirmation decode; no tuning on this cohort."
        ),
        "scoring_policy": {
            "judge": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "passes": 2,
            "original_calibration_controls": "10/10/0/0-3 in each pass, raw scores unchanged",
            "weighting": "equal families, source groups within family, clips within source group",
            "minimum_absolute_gain": 0.1,
            "minimum_relative_gain": 0.05,
            "paired_source_group_bootstrap_95_lower_bound_positive": True,
            "no_family_mean_regression": True,
            "WER_is_diagnostic_only": True,
        },
        "qualifications": [
            "Local source/reference exclusions only; upstream C5 training inventory unknown.",
            "How2Sign covers four known signers; OpenASL signer identities unavailable.",
            "Metadata reservation only. No acquisition, model admission, scoring or serving claim.",
        ],
    }
    data = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(data) < BUDGET, "confirmation reservation metadata cap failed")
    output.mkdir()
    with (output / "plan.json").open("xb") as f:
        f.write(data)
    for name, digest in plan["source_sha256"].items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write((ROOT / name).read_bytes())
        require(sha256(target) == digest, "confirmation selector archive changed")
    print(
        json.dumps(
            {
                "plan": str((output / "plan.json").relative_to(ROOT)),
                "primary_clips": 48,
                "How2Sign_content_groups": 12,
                "OpenASL_primary_sources": 24,
                "OpenASL_fallback_sources": 24,
                "planned_How2Sign_raw_source_bytes": raw_bytes,
                "prediction_or_media_access": False,
            }
        ),
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    freeze(args.output)


if __name__ == "__main__":
    main()
