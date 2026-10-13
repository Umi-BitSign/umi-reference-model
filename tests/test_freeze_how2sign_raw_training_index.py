"""Raw-video encoder states must retain publisher and transform provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from freeze_how2sign_raw_training_index import (
    canonical_rows,
    checked_source_rows,
    receipt_sha256,
)


def _inputs(tmp_path: Path) -> tuple[dict, dict, dict, dict, tuple[Path, Path, Path, Path]]:
    video_dir = tmp_path / "videos"
    source_dir = tmp_path / "source"
    resampled_dir = tmp_path / "resampled"
    cache_dir = tmp_path / "cache"
    for path in (video_dir, source_dir, resampled_dir, cache_dir):
        path.mkdir()
    video = video_dir / "video-2-rgb_front.mp4"
    source = source_dir / "sentence.mp4"
    resampled = resampled_dir / "sentence.mp4"
    cache = cache_dir / "sentence.npz"
    for path, payload in (
        (video, b"video"),
        (source, b"crop"),
        (resampled, b"resampled"),
        (cache, b"state"),
    ):
        path.write_bytes(payload)
    video_receipt = {
        "schema": "umi-studio-how2sign-raw-training-video/1",
        "video_name": "video-2-rgb_front",
        "video_id": "video",
        "signer": 2,
        "video_sha256": sha256(video),
        "video_probe": {"duration_sec": 20.0},
    }
    selected = {
        "sentence_name": "sentence",
        "video_name": "video-2-rgb_front",
        "video_id": "video",
        "signer": 2,
        "start_sec": 2.0,
        "end_sec": 5.0,
        "reference": "three signed words",
        "source_video_sha256": sha256(video),
    }
    selection = {
        "schema": "umi-studio-how2sign-training-raw-realigned/1",
        "training_only": True,
        "video_name": "video-2-rgb_front",
        "video_id": "video",
        "signer": 2,
        "source_video_sha256": sha256(video),
        "source_video_receipt_sha256": receipt_sha256(video_receipt),
        "rows": [selected],
    }
    clip_receipt = {"clip_sha256": sha256(source)}
    encoder = {
        "schema": "umi-studio-how2sign-encoder-cache-example/1",
        "split": "training",
        "sentence_name": "sentence",
        "signer": 2,
        "reference": "three signed words",
        "selection_sha256": "selection-hash",
        "clip_receipts_sha256": None,
        "source_clip_sha256": sha256(source),
        "resampled_clip_sha256": sha256(resampled),
        "encoder_cache_sha256": sha256(cache),
        "baseline_identity_sha256": "identity-hash",
        "baseline_runtime_sha256": "runtime-hash",
        "clip_fps": 25,
        "video_decode_strategy": "opencv_sequential_full_range",
        "encoder_dtype": "float32",
        "encoder_shape": [75, 1536],
    }
    return (
        selection,
        video_receipt,
        clip_receipt,
        encoder,
        (
            video_dir,
            source_dir,
            resampled_dir,
            cache_dir,
        ),
    )


def _freeze(selection, video_receipt, clip_receipt, encoder, paths):
    return canonical_rows(
        selection,
        "selection-hash",
        video_receipt,
        {"sentence": clip_receipt},
        [encoder],
        *paths,
        "identity-hash",
        "runtime-hash",
    )


def test_freezes_verified_raw_encoder_state(tmp_path):
    inputs = _inputs(tmp_path)
    assert _freeze(*inputs) == [inputs[3]]
    (tmp_path / "cache/sentence.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        _freeze(*inputs)


def test_rejects_source_video_change_or_wrong_split(tmp_path):
    selection, video_receipt, clip_receipt, encoder, paths = _inputs(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        _freeze({**selection, "training_only": False}, video_receipt, clip_receipt, encoder, paths)
    (tmp_path / "videos/video-2-rgb_front.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="publisher bytes"):
        _freeze(selection, video_receipt, clip_receipt, encoder, paths)


def test_raw_rows_recompute_from_publisher_metadata(tmp_path):
    selection, video_receipt, _, _, _ = _inputs(tmp_path)
    aligned = [
        {
            "VIDEO_NAME": "video-2-rgb_front",
            "VIDEO_ID": "video",
            "SENTENCE_NAME": "sentence",
            "START_REALIGNED": "2.0",
            "END_REALIGNED": "5.0",
            "SENTENCE": "three signed words",
        }
    ]
    pre_cut = {"schema": "umi-studio-how2sign-training-aligned/1", "rows": []}
    checked_source_rows(selection, video_receipt, aligned, pre_cut)
    with pytest.raises(ValueError, match="publisher realignment"):
        checked_source_rows(
            {**selection, "rows": [{**selection["rows"][0], "reference": "wrong text"}]},
            video_receipt,
            aligned,
            pre_cut,
        )
    with pytest.raises(ValueError, match="already in pre-cut"):
        checked_source_rows(
            selection,
            video_receipt,
            aligned,
            {**pre_cut, "rows": [{"video_id": "video", "sentence_name": "another"}]},
        )


def test_source_duration_policy_excludes_uncuttable_publisher_row(tmp_path):
    selection, video_receipt, _, _, _ = _inputs(tmp_path)
    aligned = [
        {
            "VIDEO_NAME": "video-2-rgb_front",
            "VIDEO_ID": "video",
            "SENTENCE_NAME": name,
            "START_REALIGNED": start,
            "END_REALIGNED": end,
            "SENTENCE": "three signed words",
        }
        for name, start, end in (("sentence", "2.0", "5.0"), ("truncated", "18.0", "21.0"))
    ]
    pre_cut = {"schema": "umi-studio-how2sign-training-aligned/1", "rows": []}
    with pytest.raises(ValueError, match="outside"):
        checked_source_rows(selection, video_receipt, aligned, pre_cut)
    checked_source_rows(
        {**selection, "source_duration_policy": "drop_intervals_past_verified_video_end"},
        video_receipt,
        aligned,
        pre_cut,
    )
