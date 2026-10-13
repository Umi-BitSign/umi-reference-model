"""Full aligned How2Sign index checks and canonical ordering."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_how2sign_training_index import canonical_rows


def _fixture(tmp_path: Path):
    source = tmp_path / "source"
    resampled = tmp_path / "resampled"
    cache = tmp_path / "cache"
    for directory in (source, resampled, cache):
        directory.mkdir()
    digests = {}
    for directory, suffix in ((source, ".mp4"), (resampled, ".mp4"), (cache, ".npz")):
        data = directory.name.encode()
        digests[directory.name] = hashlib.sha256(data).hexdigest()
        for number in range(1024):
            (directory / f"sentence{number}{suffix}").write_bytes(data)
    selected = [
        {
            "sentence_name": f"sentence{number}",
            "video_id": f"video{number}",
            "video_name": f"video{number}-1-rgb_front",
            "signer": 1,
            "reference": f"Reference {number}.",
            "original_interval_overlap_fraction": 1.0,
            "realigned_interval_overlap_fraction": 1.0,
        }
        for number in range(1024)
    ]
    receipts = {row["sentence_name"]: {"clip_sha256": digests["source"]} for row in selected}
    encoder = [
        {
            "schema": "umi-studio-how2sign-encoder-cache-example/1",
            "split": "training",
            "sentence_name": row["sentence_name"],
            "signer": row["signer"],
            "reference": row["reference"],
            "selection_sha256": "selection-hash",
            "clip_receipts_sha256": None,
            "source_clip_sha256": digests["source"],
            "resampled_clip_sha256": digests["resampled"],
            "encoder_cache_sha256": digests["cache"],
            "baseline_identity_sha256": "identity-hash",
            "baseline_runtime_sha256": "runtime-hash",
            "clip_fps": 25,
            "video_decode_strategy": "opencv_sequential_full_range",
            "encoder_dtype": "float32",
            "encoder_shape": [100, 1536],
        }
        for row in selected
    ]
    return (
        {
            "schema": "umi-studio-how2sign-training-aligned/1",
            "rows": selected,
            "max_clips_per_source_video": 2,
            "min_bidirectional_overlap": 0.8,
            "per_signer_quotas": {"1": 1024},
        },
        receipts,
        encoder,
        source,
        resampled,
        cache,
    )


def test_full_index_orders_reused_states_by_frozen_selection(tmp_path):
    selection, receipts, encoder, source, resampled, cache = _fixture(tmp_path)
    ordered = canonical_rows(
        selection,
        "selection-hash",
        receipts,
        list(reversed(encoder)),
        source,
        resampled,
        cache,
        "identity-hash",
        "runtime-hash",
    )
    assert [row["sentence_name"] for row in ordered] == [
        row["sentence_name"] for row in selection["rows"]
    ]


def test_full_index_rejects_incomplete_and_changed_cache(tmp_path):
    selection, receipts, encoder, source, resampled, cache = _fixture(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        canonical_rows(
            selection,
            "selection-hash",
            receipts,
            encoder[:-1],
            source,
            resampled,
            cache,
            "identity-hash",
            "runtime-hash",
        )
    (cache / "sentence0.npz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        canonical_rows(
            selection,
            "selection-hash",
            receipts,
            encoder,
            source,
            resampled,
            cache,
            "identity-hash",
            "runtime-hash",
        )


def test_index_uses_selection_count_and_signer_specific_video_cap(tmp_path):
    selection, receipts, encoder, source, resampled, cache = _fixture(tmp_path)
    selection["rows"] = selection["rows"][:3]
    selection["per_signer_quotas"] = {"1": 3}
    receipts = {row["sentence_name"]: receipts[row["sentence_name"]] for row in selection["rows"]}
    encoder = encoder[:3]
    for row in selection["rows"]:
        row["video_id"] = "shared-source-id"
    ordered = canonical_rows(
        selection,
        "selection-hash",
        receipts,
        encoder,
        source,
        resampled,
        cache,
        "identity-hash",
        "runtime-hash",
    )
    assert len(ordered) == 3
    for row in selection["rows"]:
        row["video_name"] = "same-signer-video"
    with pytest.raises(ValueError, match="incomplete"):
        canonical_rows(
            selection,
            "selection-hash",
            receipts,
            encoder,
            source,
            resampled,
            cache,
            "identity-hash",
            "runtime-hash",
        )


def test_index_rejects_noninteger_quota(tmp_path):
    selection, receipts, encoder, source, resampled, cache = _fixture(tmp_path)
    selection["per_signer_quotas"] = {"1": "1024"}
    with pytest.raises(ValueError, match="incomplete"):
        canonical_rows(
            selection,
            "selection-hash",
            receipts,
            encoder,
            source,
            resampled,
            cache,
            "identity-hash",
            "runtime-hash",
        )
