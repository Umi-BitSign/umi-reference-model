"""The replay trainer accepts only verified source-training encoder states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from freeze_aligned_mixed_plan import build_plan
from train_mixed_lora import check_source_plan, load_examples


def _fixture(tmp_path: Path, source: str):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "MODEL_IDENTITY.json").write_text("identity")
    (baseline / "runtime.py").write_text("runtime")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    identifier = {
        "fleurs": "train_sentence",
        "how2sign": "video_0-1-rgb_front",
        "flores2m": "flores2m_dev_row_0001",
    }[source]
    cache = cache_dir / f"{identifier}.npz"
    np.savez(cache, encoder_hidden_state=np.zeros((2, 1536), dtype=np.float32))
    row = {
        "split": "training",
        "signer": 3 if source == "fleurs" else 1,
        "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
        "baseline_runtime_sha256": sha256(baseline / "runtime.py"),
        "clip_fps": 25,
        "encoder_cache_sha256": sha256(cache),
        "encoder_shape": [2, 1536],
        "reference": "A signed sentence.",
    }
    if source == "fleurs":
        row["example_id"] = identifier
    elif source == "how2sign":
        row["schema"] = "umi-studio-how2sign-encoder-cache-example/1"
        row["sentence_name"] = identifier
    else:
        row["schema"] = "umi-studio-flores2m-encoder-cache-example/1"
        row["example_id"] = identifier
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    return baseline, cache_dir, index, row


@pytest.mark.parametrize("source", ["fleurs", "how2sign", "flores2m"])
def test_training_source_is_accepted(tmp_path: Path, source: str):
    baseline, cache_dir, index, _ = _fixture(tmp_path, source)
    examples = load_examples(index, cache_dir, source, baseline)
    assert len(examples) == 1
    assert examples[0]["source"] == source


@pytest.mark.parametrize("source", ["fleurs", "how2sign", "flores2m"])
def test_validation_source_is_rejected(tmp_path: Path, source: str):
    baseline, cache_dir, index, row = _fixture(tmp_path, source)
    row["split"] = "validation"
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="non-training"):
        load_examples(index, cache_dir, source, baseline)


def test_altered_cache_is_rejected(tmp_path: Path):
    baseline, cache_dir, index, _ = _fixture(tmp_path, "how2sign")
    cache = next(cache_dir.glob("*.npz"))
    cache.write_bytes(b"altered")
    with pytest.raises(ValueError, match="altered"):
        load_examples(index, cache_dir, "how2sign", baseline)


def test_aligned_plan_pins_cache_and_excludes_flores(tmp_path: Path):
    fleurs = tmp_path / "fleurs.jsonl"
    fleurs.write_text("frozen fleurs index\n")
    how2sign = tmp_path / "how2sign.jsonl"
    how2sign.write_text(json.dumps({"selection_sha256": "selection-hash"}) + "\n")
    flores = tmp_path / "flores.jsonl"
    flores.write_text(json.dumps({"selection_sha256": "flores-hash"}) + "\n")
    plan = {
        "schema": "umi-studio-aligned-mixed-plan/1",
        "fleurs_training_index_sha256": sha256(fleurs),
        "how2sign_training_index_sha256": sha256(how2sign),
        "how2sign_selection_sha256": "selection-hash",
        "training_examples": {"flores2m": 0},
    }
    check_source_plan(plan, fleurs, how2sign, None)
    with pytest.raises(ValueError, match="excludes"):
        check_source_plan(plan, fleurs, how2sign, flores)
    how2sign.write_text(json.dumps({"selection_sha256": "other-selection"}) + "\n")
    with pytest.raises(ValueError, match="index differs"):
        check_source_plan(plan, fleurs, how2sign, None)
    plan["how2sign_training_index_sha256"] = sha256(how2sign)
    with pytest.raises(ValueError, match="selection differs"):
        check_source_plan(plan, fleurs, how2sign, None)


def test_aligned_plan_seals_confirmation_and_rejects_validation_training():
    fleurs = [
        {
            "schema": "umi-studio-fleurs-dev-encoder-cache-example/1",
            "split": "training",
            "signer": 3,
            "example_id": f"fleurs-{number}",
        }
        for number in range(313)
    ]
    selected = [
        {"sentence_name": f"h2s-{number}", "reference": f"Sentence {number}."}
        for number in range(1024)
    ]
    how2sign = [
        {
            "schema": "umi-studio-how2sign-encoder-cache-example/1",
            "split": "training",
            "selection_sha256": "selection-hash",
            **row,
        }
        for row in selected[:168]
    ]
    partition = {
        "schema": "umi-studio-how2sign-validation-partition/1",
        "confirmation_predictions_inspected": False,
        "tuning_sentence_names": [f"tune-{number}" for number in range(16)],
        "confirmation_sentence_names": [f"confirm-{number}" for number in range(42)],
    }
    fleurs_baseline = [{"split": "tuning", "single_reference_wer": 1.0}] * 7
    how2sign_baseline = [
        {"split": "validation", "example_id": name, "single_reference_wer": 0.9}
        for name in partition["tuning_sentence_names"]
    ]
    hashes = {
        key: "selection-hash" if key == "how2sign_selection" else "hash"
        for key in (
            "fleurs_index",
            "how2sign_index",
            "how2sign_selection",
            "fleurs_partition",
            "how2sign_partition",
            "fleurs_baseline",
            "how2sign_baseline",
        )
    }
    plan = build_plan(
        fleurs,
        how2sign,
        {"schema": "umi-studio-how2sign-training-aligned/1", "rows": selected},
        fleurs_baseline,
        how2sign_baseline,
        partition,
        hashes,
        168,
    )
    assert plan["training_examples"] == {"fleurs": 313, "how2sign": 168, "flores2m": 0}
    assert plan["baseline_mean_single_reference_wer"]["how2sign"] == 0.9
    how2sign[0]["split"] = "validation"
    with pytest.raises(ValueError, match="incomplete"):
        build_plan(
            fleurs,
            how2sign,
            {"schema": "umi-studio-how2sign-training-aligned/1", "rows": selected},
            fleurs_baseline,
            how2sign_baseline,
            partition,
            hashes,
            168,
        )
