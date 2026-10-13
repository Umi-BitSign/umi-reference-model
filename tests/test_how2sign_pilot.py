"""Selection and byte-range guards for the official How2Sign pilot."""

from __future__ import annotations

import json
import sys
import urllib.error
import zipfile
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import how2sign_remote_zip
from benchmark_fleurs_dev import sha256
from cache_how2sign_encoder import (
    checked_receipts,
    pending_rows,
    selected_window,
    verified_index,
)
from evaluate_fleurs_cache import evaluation_identifiers
from fetch_how2sign_pilot import SIGNERS, select_rows
from fetch_how2sign_validation import select_rows as select_validation_rows
from freeze_how2sign_validation import build_partition
from how2sign_remote_zip import ARCHIVE_SIZE, MAX_RANGE_BYTES, How2SignRemoteZip
from reuse_how2sign_encoder import reusable_record
from select_how2sign_full_video_confirmation import select_videos


def _fixture_rows():
    rows = []
    infos = {}
    for signer in SIGNERS:
        for number in range(3):
            video = f"video{number}-{signer}-rgb_front"
            sentence = f"video{number}_0-{signer}-rgb_front"
            rows.append(
                {
                    "VIDEO_ID": f"video{number}",
                    "VIDEO_NAME": video,
                    "SENTENCE_ID": f"sentence{number}",
                    "SENTENCE_NAME": sentence,
                    "START": "1.0",
                    "END": "5.0",
                    "SENTENCE": "A short example sentence.",
                }
            )
            info = zipfile.ZipInfo(f"raw_videos/{sentence}.mp4")
            info.file_size = 1000
            info.compress_size = 800
            info.CRC = number
            infos[info.filename] = info
    return rows, infos


def test_pilot_selection_is_balanced_and_deterministic():
    rows, infos = _fixture_rows()
    first = select_rows(rows, infos, per_signer=2)
    second = select_rows(list(reversed(rows)), infos, per_signer=2)
    assert first == second
    assert len(first) == 2 * len(SIGNERS)
    assert len({row["video_name"] for row in first}) == len(first)
    assert {row["signer"] for row in first} == set(SIGNERS)


def test_pilot_rejects_duplicate_sentence_name():
    rows, infos = _fixture_rows()
    with pytest.raises(ValueError, match="duplicate"):
        select_rows([*rows, rows[0]], infos, per_signer=2)


def test_scale_quotas_exclude_prior_source_videos():
    rows, infos = _fixture_rows()
    prior = select_rows(rows, infos, per_signer=1)
    excluded = {row["video_name"] for row in prior}
    quotas = {signer: (2 if signer in SIGNERS[:2] else 0) for signer in SIGNERS}
    additional = select_rows(rows, infos, quotas=quotas, excluded_videos=excluded)
    assert len(additional) == 4
    assert {row["video_name"] for row in additional}.isdisjoint(excluded)
    assert {row["signer"] for row in additional} == set(SIGNERS[:2])


def test_scale_rejects_invalid_quotas():
    rows, infos = _fixture_rows()
    with pytest.raises(ValueError, match="quotas"):
        select_rows(rows, infos, quotas={SIGNERS[0]: 1})
    with pytest.raises(ValueError, match="qualifying clips"):
        select_rows(rows, infos, quotas={signer: 4 for signer in SIGNERS})


def test_aligned_selection_filters_misaligned_clips_and_caps_source_videos():
    rows, infos = _fixture_rows()
    for signer in SIGNERS:
        first = next(row for row in rows if row["VIDEO_NAME"] == f"video0-{signer}-rgb_front")
        extra = {
            **first,
            "SENTENCE_ID": "extra",
            "SENTENCE_NAME": f"video0_extra-{signer}-rgb_front",
        }
        rows.append(extra)
        info = zipfile.ZipInfo(f"raw_videos/{extra['SENTENCE_NAME']}.mp4")
        info.file_size = 1000
        info.compress_size = 800
        info.CRC = 1
        infos[info.filename] = info
    aligned = {
        row["SENTENCE_NAME"]: {
            **row,
            "START_REALIGNED": "10.0" if row["VIDEO_ID"] == "video2" else "1.0",
            "END_REALIGNED": "14.0" if row["VIDEO_ID"] == "video2" else "5.0",
        }
        for row in rows
    }
    selected = select_rows(
        rows,
        infos,
        quotas={signer: 3 for signer in SIGNERS},
        alignment_by_name=aligned,
        min_bidirectional_overlap=0.8,
        max_clips_per_video=2,
    )
    assert len(selected) == 3 * len(SIGNERS)
    assert all(row["original_interval_overlap_fraction"] >= 0.8 for row in selected)
    assert all(row["realigned_interval_overlap_fraction"] >= 0.8 for row in selected)
    assert max(Counter(row["video_name"] for row in selected).values()) == 2
    assert len({row["video_name"] for row in selected}) == 2 * len(SIGNERS)
    aligned[rows[0]["SENTENCE_NAME"]]["SENTENCE"] = "different translation"
    with pytest.raises(ValueError, match="identity differs"):
        select_rows(
            rows,
            infos,
            quotas={signer: 3 for signer in SIGNERS},
            alignment_by_name=aligned,
            min_bidirectional_overlap=0.8,
            max_clips_per_video=2,
        )


def test_scale_cache_accepts_verified_partial_receipts_only():
    names = ["first", "second"]
    records = [
        {
            "schema": "umi-studio-how2sign-training-clip/1",
            "selection_sha256": "selection-hash",
            "sentence_name": name,
        }
        for name in names
    ]
    scale = {
        "schema": "umi-studio-how2sign-training-scale/1",
        "rows": [{"sentence_name": name} for name in names],
    }
    assert set(checked_receipts(scale, "selection-hash", [json.dumps(records[0])])) == {"first"}
    with pytest.raises(ValueError, match="downloaded first"):
        checked_receipts(
            {**scale, "schema": "umi-studio-how2sign-training-pilot/1"},
            "selection-hash",
            [json.dumps(records[0])],
        )
    with pytest.raises(ValueError, match="invalid or duplicated"):
        checked_receipts(
            scale, "selection-hash", [json.dumps({**records[0], "sentence_name": "unknown"})]
        )


def test_encoder_windows_do_not_overlap_selection_rows():
    rows = [{"sentence_name": str(number)} for number in range(6)]
    assert selected_window(rows, 0, 3) == rows[:3]
    assert selected_window(rows, 3, None) == rows[3:]
    with pytest.raises(ValueError, match="outside"):
        selected_window(rows, 6, None)
    with pytest.raises(ValueError, match="outside"):
        selected_window(rows, 2, 7)


def test_encoder_window_keeps_reused_states_outside_its_range():
    rows = [{"sentence_name": str(number)} for number in range(6)]
    receipts = {row["sentence_name"]: {} for row in rows}
    completed = {"5": {}}
    assert pending_rows(selected_window(rows, 0, 3), receipts, completed, {}) == rows[:3]
    assert pending_rows(selected_window(rows, 3, 6), receipts, {}, completed) == rows[3:5]
    with pytest.raises(ValueError, match="overlap"):
        pending_rows(rows, receipts, completed, completed)


def test_skip_index_checks_cached_state_hash(tmp_path):
    index = tmp_path / "index.jsonl"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = cache_dir / "name.npz"
    cache.write_bytes(b"state")
    record = {
        "schema": "umi-studio-how2sign-encoder-cache-example/1",
        "sentence_name": "name",
        "selection_sha256": "selection",
        "clip_receipts_sha256": None,
        "source_clip_sha256": "source",
        "encoder_cache_sha256": sha256(cache),
    }
    index.write_text(json.dumps(record) + "\n")
    receipts = {"name": {"clip_sha256": "source"}}
    assert verified_index(index, "selection", None, receipts, cache_dir) == {"name": record}
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or altered"):
        verified_index(index, "selection", None, receipts, cache_dir)


def test_validation_cache_accepts_only_validation_receipts():
    selection = {
        "schema": "umi-studio-how2sign-validation-selection/1",
        "split": "validation",
        "training_excluded": True,
        "rows": [{"sentence_name": "first"}, {"sentence_name": "second"}],
    }
    record = {
        "schema": "umi-studio-how2sign-validation-clip/1",
        "split": "validation",
        "selection_sha256": "selection-hash",
        "sentence_name": "first",
    }
    assert set(checked_receipts(selection, "selection-hash", [json.dumps(record)])) == {"first"}
    with pytest.raises(ValueError, match="invalid or duplicated"):
        checked_receipts(selection, "selection-hash", [json.dumps({**record, "split": "training"})])


def test_validation_cache_evaluation_uses_sentence_ids_only():
    rows = [
        {
            "schema": "umi-studio-how2sign-encoder-cache-example/1",
            "split": "validation",
            "sentence_name": "one",
        },
        {
            "schema": "umi-studio-how2sign-encoder-cache-example/1",
            "split": "validation",
            "sentence_name": "two",
        },
    ]
    assert evaluation_identifiers(rows, "validation") == ["one", "two"]
    with pytest.raises(ValueError, match="schema differs"):
        evaluation_identifiers([{**rows[0], "split": "tuning"}], "tuning")
    with pytest.raises(ValueError, match="duplicate"):
        evaluation_identifiers([rows[0], rows[0]], "validation")


def test_validation_partition_seals_remaining_clips():
    names = [f"sentence-{number}" for number in range(17)]
    selection = {
        "schema": "umi-studio-how2sign-validation-selection/1",
        "split": "validation",
        "training_excluded": True,
        "rows": [
            {"sentence_name": name, "video_id": f"video-{number}", "signer": 1}
            for number, name in enumerate(names)
        ],
    }
    index = [
        {
            "schema": "umi-studio-how2sign-encoder-cache-example/1",
            "split": "validation",
            "selection_sha256": "selection-hash",
            "sentence_name": name,
        }
        for name in names[:16]
    ]
    baseline = [
        {
            "schema": "umi-studio-how2sign-cached-evaluation-example/1",
            "split": "validation",
            "selection_sha256": "selection-hash",
            "index_sha256": "index-hash",
            "example_id": name,
            "single_reference_wer": 0.5,
        }
        for name in names[:16]
    ]
    partition = build_partition(selection, index, baseline, "selection-hash", "index-hash", "b")
    assert partition["confirmation_sentence_names"] == [names[-1]]
    assert partition["tuning_baseline_mean_single_reference_wer"] == 0.5
    with pytest.raises(ValueError, match="incomplete"):
        build_partition(selection, index, baseline[:-1], "selection-hash", "index-hash", "b")


def test_full_video_confirmation_excludes_shared_source_ids():
    rows = []
    infos = {}
    for signer in (1, 2):
        for video_id in ("shared", f"unique{signer}"):
            video_name = f"{video_id}-{signer}-rgb_front"
            info = zipfile.ZipInfo(f"raw_videos/{video_name}.mp4")
            info.file_size = 1000
            info.compress_size = 800
            info.CRC = signer
            infos[info.filename] = info
            for number, start in enumerate((1.0, 22.0)):
                rows.append(
                    {
                        "VIDEO_ID": video_id,
                        "VIDEO_NAME": video_name,
                        "SENTENCE_ID": f"{video_id}_{number}",
                        "SENTENCE_NAME": f"{video_id}_{number}-{signer}-rgb_front",
                        "START_REALIGNED": str(start),
                        "END_REALIGNED": str(start + 4),
                        "SENTENCE": f"Sentence {number} for signer {signer}.",
                    }
                )
    selected = select_videos(rows, infos, set(), set(), set(), {1: 1, 2: 1})
    assert len(selected) == 2
    assert {row["signer"] for row in selected} == {1, 2}
    assert len({row["video_id"] for row in selected}) == 2
    assert all(len(row["clips"]) == 2 for row in selected)
    with pytest.raises(ValueError, match="overlaps official training"):
        select_videos(rows, infos, set(), {"shared"}, set(), {1: 1, 2: 1})


def test_prior_encoder_reuse_requires_matching_source_and_model():
    row = {"sentence_name": "clip", "signer": 1, "reference": "A signed sentence."}
    receipt = {"clip_sha256": "source-hash"}
    prior = {
        "schema": "umi-studio-how2sign-encoder-cache-example/1",
        "split": "training",
        "sentence_name": "clip",
        "signer": 1,
        "reference": "A signed sentence.",
        "source_clip_sha256": "source-hash",
        "clip_fps": 25,
        "video_decode_strategy": "opencv_sequential_full_range",
        "baseline_identity_sha256": "identity-hash",
        "baseline_runtime_sha256": "runtime-hash",
        "selection_sha256": "prior-selection-hash",
        "encoder_cache_sha256": "cache-hash",
    }
    reused = reusable_record(
        row, receipt, prior, "aligned-selection-hash", "identity-hash", "runtime-hash", "rev", "i"
    )
    assert reused["selection_sha256"] == "aligned-selection-hash"
    assert reused["clip_receipts_sha256"] is None
    assert reused["reuse_source"]["prior_encoder_cache_sha256"] == "cache-hash"
    with pytest.raises(ValueError, match="differs"):
        reusable_record(
            row,
            {"clip_sha256": "other"},
            prior,
            "aligned-selection-hash",
            "identity-hash",
            "runtime-hash",
            "rev",
            "i",
        )


def test_remote_archive_rejects_unbounded_large_read():
    remote = How2SignRemoteZip.__new__(How2SignRemoteZip)
    remote.position = 0
    remote.archive_size = ARCHIVE_SIZE
    with pytest.raises(ValueError, match="safety limit"):
        remote.read()
    assert remote.seek(-10, 2) == ARCHIVE_SIZE - 10
    remote._read_range = lambda start, end: bytes(end - start + 1)
    assert remote.read() == bytes(10)
    assert remote.tell() == ARCHIVE_SIZE
    remote.seek(0)
    with pytest.raises(ValueError, match="safety limit"):
        remote.read(MAX_RANGE_BYTES + 1)


def test_remote_archive_retries_transient_503(monkeypatch):
    remote = How2SignRemoteZip.__new__(How2SignRemoteZip)
    remote.url = "https://example.invalid/archive"
    remote.archive_size = 10
    remote.archive_name = "archive.zip"
    remote.bytes_downloaded = 0
    remote.requests = 0
    attempts = []
    pauses = []

    class Response:
        def __init__(self):
            self.status = 206
            self.headers = {
                "Content-Range": "bytes 0-0/10",
                "Content-Disposition": 'attachment; filename="archive.zip"',
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return b"Z"

    def urlopen(request, timeout):
        attempts.append((request, timeout))
        if len(attempts) < 3:
            raise urllib.error.HTTPError(remote.url, 503, "busy", {}, None)
        return Response()

    monkeypatch.setattr(how2sign_remote_zip.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(how2sign_remote_zip.time, "sleep", pauses.append)
    assert remote._read_range(0, 0) == b"Z"
    assert len(attempts) == 3
    assert pauses == [1, 2]
    assert remote.bytes_downloaded == remote.requests == 1


def test_remote_archive_refreshes_after_range_response_is_ignored(monkeypatch):
    remote = How2SignRemoteZip.__new__(How2SignRemoteZip)
    remote.url = "https://example.invalid/stale"
    remote.archive_size = 10
    remote.archive_name = "archive.zip"
    remote.bytes_downloaded = 0
    remote.requests = 0
    attempts = []
    pauses = []

    class Response:
        def __init__(self, valid):
            self.status = 206 if valid else 200
            self.headers = {
                "Content-Disposition": 'attachment; filename="archive.zip"',
                **({"Content-Range": "bytes 0-0/10"} if valid else {}),
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return b"Z"

    def urlopen(request, timeout):
        attempts.append((request.full_url, timeout))
        return Response(valid=len(attempts) > 1)

    monkeypatch.setattr(how2sign_remote_zip.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(remote, "_confirmation_url", lambda: "https://example.invalid/fresh")
    monkeypatch.setattr(how2sign_remote_zip.time, "sleep", pauses.append)
    assert remote._read_range(0, 0) == b"Z"
    assert [url for url, _ in attempts] == [
        "https://example.invalid/stale",
        "https://example.invalid/fresh",
    ]
    assert pauses == [1]
    assert remote.bytes_downloaded == remote.requests == 1


def test_validation_selection_is_aligned_balanced_and_disjoint():
    original = []
    realigned = []
    infos = {}
    for signer in (1, 2, 3):
        for number in range(3):
            video_id = f"video{signer}_{number}"
            video_name = f"{video_id}-{signer}-rgb_front"
            sentence_name = f"{video_id}_0-{signer}-rgb_front"
            row = {
                "VIDEO_ID": video_id,
                "VIDEO_NAME": video_name,
                "SENTENCE_ID": f"{video_id}_0",
                "SENTENCE_NAME": sentence_name,
                "START": "1.0",
                "END": "5.0",
                "SENTENCE": "A short validation sentence.",
            }
            original.append(row)
            realigned.append(
                {
                    **row,
                    "START_REALIGNED": "10.0" if number == 2 else "1.2",
                    "END_REALIGNED": "14.0" if number == 2 else "4.8",
                }
            )
            info = zipfile.ZipInfo(f"raw_videos/{sentence_name}.mp4")
            info.file_size = 1000
            info.compress_size = 800
            info.CRC = number
            infos[info.filename] = info
    selected = select_validation_rows(original, realigned, infos, set(), 4, 0.8)
    assert len(selected) == 4
    assert {row["signer"] for row in selected} == {1, 2, 3}
    assert len({row["video_id"] for row in selected}) == 4
    assert all(row["original_interval_overlap_fraction"] >= 0.8 for row in selected)
    assert all(row["realigned_interval_overlap_fraction"] >= 0.8 for row in selected)
    assert all(not row["video_id"].endswith("_2") for row in selected)
    with pytest.raises(ValueError, match="overlaps official training"):
        select_validation_rows(original, realigned, infos, {original[0]["VIDEO_ID"]}, 4, 0.8)
    realigned[0]["SENTENCE"] = "A changed translation."
    with pytest.raises(ValueError, match="row identities differ"):
        select_validation_rows(original, realigned, infos, set(), 4, 0.8)
