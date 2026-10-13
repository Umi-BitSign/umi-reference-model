"""The visual plan requires complete source identities and disjoint videos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from freeze_visual_lora_plan import (
    SOURCE_LAYOUT,
    TUNING_LAYOUT,
    checked_fleurs_fidelity,
    checked_partition,
    checked_stage_one_mix,
    checked_tuning,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_partition_requires_exact_coverage_and_source_disjointness(tmp_path):
    fleurs_ids = [f"f-{number}" for number in range(313)]
    h_ids = [f"h-{number}" for number in range(3359)]
    raw_ids = [f"r-{number}" for number in range(882)]
    _write_json(
        tmp_path / "runs/fleurs-dev-v1/selection.json",
        {
            "schema": "umi-studio-fleurs-dev-split/1",
            "training": [
                {"example_id": identifier, "reference": "fleurs text", "signer": 3}
                for identifier in fleurs_ids
            ],
        },
    )
    _write_json(
        tmp_path / "runs/how2sign-expanded-v1/selection.json",
        {
            "schema": "umi-studio-how2sign-training-aligned/1",
            "rows": [
                {
                    "video_id": "train-video",
                    "sentence_name": identifier,
                    "reference": "how2sign text",
                    "signer": 1,
                }
                for identifier in [*h_ids, *(f"unavailable-{number}" for number in range(30))]
            ],
        },
    )
    _write_json(
        tmp_path / "runs/how2sign-raw-v1/expansion-selection-v2.json",
        {
            "schema": "umi-studio-how2sign-raw-training-expansion/2",
            "training_only": True,
            "selected_clip_count": 882,
            "videos": [
                {
                    "video_id": "raw-video",
                    "signer": 5,
                    "clips": [
                        {"sentence_name": identifier, "reference": "raw text"}
                        for identifier in raw_ids
                    ],
                }
            ],
        },
    )
    _write_json(
        tmp_path / "runs/how2sign-validation-v1/selection.json",
        {"split": "validation", "rows": [{"video_id": "validation-video"}]},
    )
    fleurs_hash = sha256(tmp_path / "runs/fleurs-dev-v1/selection.json")
    h_hash = sha256(tmp_path / "runs/how2sign-expanded-v1/selection.json")
    _write_jsonl(
        tmp_path / "runs/how2sign-expanded-v1/clip-receipts.jsonl",
        [{"sentence_name": identifier, "selection_sha256": h_hash} for identifier in h_ids],
    )
    ids_by_source = {
        "fleurs": fleurs_ids,
        "how2sign_original": h_ids[:1024],
        "how2sign_new": h_ids[1024:],
        "how2sign_raw": raw_ids,
    }
    index_hashes = {}
    for source, identifiers in ids_by_source.items():
        index = tmp_path / SOURCE_LAYOUT[source]["index"]
        _write_jsonl(
            index,
            [
                {
                    "selection_sha256": fleurs_hash if source == "fleurs" else h_hash
                }
                for _ in identifiers
            ],
        )
        index_hashes[source] = sha256(index)
    examples = {
        source: [
            {
                "id": identifier,
                "reference": (
                    "fleurs text"
                    if source == "fleurs"
                    else "raw text" if source == "how2sign_raw" else "how2sign text"
                ),
                "signer": 3 if source == "fleurs" else 5 if source == "how2sign_raw" else 1,
            }
            for identifier in identifiers
        ]
        for source, identifiers in ids_by_source.items()
    }
    assert checked_partition(tmp_path, examples, index_hashes)[
        "how2sign_selection_sha256"
    ] == h_hash
    examples["how2sign_new"][0]["id"] = "another-sentence"
    with pytest.raises(ValueError, match="cover"):
        checked_partition(tmp_path, examples, index_hashes)
    examples["how2sign_new"][0]["id"] = h_ids[1024]
    validation = tmp_path / "runs/how2sign-validation-v1/selection.json"
    _write_json(
        validation,
        {"split": "validation", "rows": [{"video_id": "raw-video"}]},
    )
    with pytest.raises(ValueError, match="overlap"):
        checked_partition(tmp_path, examples, index_hashes)
    _write_json(
        validation,
        {"split": "validation", "rows": [{"video_id": "validation-video"}]},
    )
    examples["how2sign_raw"][0]["reference"] = "incorrect text"
    with pytest.raises(ValueError, match="transcript"):
        checked_partition(tmp_path, examples, index_hashes)


def test_tuning_baselines_are_frozen_to_visual_stream_indexes(tmp_path):
    checkpoint = "c" * 64
    fleurs_ids = [f"fleurs-{number}" for number in range(7)]
    _write_json(
        tmp_path / "runs/fleurs-dev-v1/selection.json",
        {
            "validation": [
                {"example_id": identifier, "document_id": "tuning-document"}
                for identifier in fleurs_ids
            ]
        },
    )
    _write_json(
        tmp_path / "runs/fleurs-dev-v1/evaluation-partition.json",
        {
            "schema": "umi-studio-fleurs-dev-evaluation-partition/1",
            "selection_sha256": sha256(tmp_path / "runs/fleurs-dev-v1/selection.json"),
            "tuning_documents": ["tuning-document"],
            "tuning_examples": 7,
        },
    )
    _write_json(tmp_path / "runs/how2sign-validation-v1/selection.json", {"split": "validation"})
    _write_json(
        tmp_path / "runs/how2sign-validation-v1/partition.json",
        {
            "schema": "umi-studio-how2sign-validation-partition/1",
            "selection_sha256": sha256(tmp_path / "runs/how2sign-validation-v1/selection.json"),
            "tuning_sentence_names": [f"how2sign-{number}" for number in range(16)],
        },
    )
    for domain, layout in TUNING_LAYOUT.items():
        index = tmp_path / layout["index"]
        results = tmp_path / layout["baseline"]
        source = [
            {
                "example_id": f"{domain}-{number}",
                "sentence_name": f"{domain}-{number}",
                "reference": "a sentence",
                "signer": 4 if domain == "fleurs" else 1,
                "selection_sha256": "s" * 64,
                "schema": (
                    "umi-studio-fleurs-dev-encoder-cache-example/1"
                    if domain == "fleurs"
                    else "umi-studio-how2sign-encoder-cache-example/1"
                ),
                "split": layout["split"],
            }
            for number in range(layout["examples"])
        ]
        _write_jsonl(index, source)
        _write_jsonl(
            results,
            [
                {
                    **row,
                    "schema": layout["schema"],
                    "split": layout["split"],
                    "evaluation_input": "visual_streams",
                    "adapter_sha256": None,
                    "checkpoint_sha256": checkpoint,
                    "index_sha256": sha256(index),
                    "single_reference_wer": 0.5,
                }
                for row in source
            ],
        )
    tuning = checked_tuning(tmp_path, checkpoint)
    assert tuning["fleurs"]["baseline_mean_single_reference_wer"] == 0.5
    candidate = tmp_path / TUNING_LAYOUT["how2sign"]["baseline"]
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    rows[0]["evaluation_input"] = "frozen_encoder_hidden_state"
    _write_jsonl(candidate, rows)
    with pytest.raises(ValueError, match="tuning baseline"):
        checked_tuning(tmp_path, checkpoint)


def test_plan_requires_complete_fleurs_training_state_fidelity(tmp_path):
    selection = tmp_path / "runs/fleurs-dev-v1/selection.json"
    original = tmp_path / "runs/fleurs-dev-v1/training-encoder-cache.jsonl"
    streamed = tmp_path / SOURCE_LAYOUT["fleurs"]["index"]
    verifier = tmp_path / "tools/verify_fleurs_training_stream_fidelity.py"
    output = tmp_path / "runs/fleurs-dev-v1/training-stream-fidelity.json"
    _write_json(selection, {"schema": "selection"})
    _write_jsonl(original, [{"id": "old"}])
    _write_jsonl(streamed, [{"id": "new"}])
    verifier.parent.mkdir(parents=True)
    verifier.write_text("verifier")
    record = {
        "schema": "umi-studio-fleurs-training-stream-fidelity/1",
        "examples_verified": 313,
        "complete": True,
        "selection_sha256": sha256(selection),
        "original_index_sha256": sha256(original),
        "stream_index_sha256": sha256(streamed),
        "verifier_source_sha256": sha256(verifier),
        "original_cache_dir": str(
            (tmp_path / "data/fleurs-asl/encoder-cache-25fps/training").resolve()
        ),
        "stream_cache_dir": str((tmp_path / SOURCE_LAYOUT["fleurs"]["cache_dir"]).resolve()),
    }
    _write_json(output, record)
    assert checked_fleurs_fidelity(tmp_path)["sha256"] == sha256(output)
    record["examples_verified"] = 312
    _write_json(output, record)
    with pytest.raises(ValueError, match="fidelity proof"):
        checked_fleurs_fidelity(tmp_path)


def test_stage_one_mix_guard_rejects_dominant_signer():
    signer_counts = {1: 395, 2: 377, 3: 351, 4: 367, 5: 400, 8: 756, 9: 379, 11: 367}
    scheduled = [{"source": "fleurs", "signer": 3} for _ in range(704)]
    how2sign = [
        {"source": "how2sign_original", "signer": signer}
        for signer, count in signer_counts.items()
        for _ in range(count)
    ]
    for row in how2sign[:1550]:
        row["source"] = "how2sign_raw"
    for row in how2sign[1550:2595]:
        row["source"] = "how2sign_new"
    scheduled.extend(how2sign)
    sources, signers = checked_stage_one_mix(scheduled)
    assert sources["how2sign_raw"] == 1550
    assert signers[8] == 756
    for row in how2sign[:500]:
        row["signer"] = 8
    with pytest.raises(ValueError, match="mix"):
        checked_stage_one_mix(scheduled)
