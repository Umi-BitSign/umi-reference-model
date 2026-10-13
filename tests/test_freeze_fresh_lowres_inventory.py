from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_fresh_lowres_inventory import (
    asl_stem_wiki_examples,
    fleurs_examples,
    how2sign_examples,
    how2sign_precut_expansion_examples,
    validate_inventory,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_how2sign_inventory_strips_all_cached_model_fields(tmp_path: Path) -> None:
    video_dir = tmp_path / "clips"
    video_dir.mkdir()
    (video_dir / "clip-1.mp4").write_bytes(b"video")
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema": "umi-studio-how2sign-training-aligned/1",
            "rows": [
                {
                    "sentence_name": "clip-1",
                    "reference": "example text",
                    "signer": 2,
                    "video_id": "source-video",
                }
            ],
        },
    )
    index = tmp_path / "index.jsonl"
    _write_jsonl(
        index,
        [
            {
                "schema": "umi-studio-how2sign-encoder-cache-example/1",
                "split": "training",
                "clip_fps": 25,
                "sentence_name": "clip-1",
                "reference": "example text",
                "signer": 2,
                "resampled_clip_sha256": "a" * 64,
                "encoder_cache_sha256": "b" * 64,
                "baseline_identity_sha256": "c" * 64,
            }
        ],
    )
    rows = how2sign_examples(
        workspace=tmp_path,
        source="how2sign_aligned",
        index_paths=[index],
        selection_path=selection,
        video_dir=video_dir,
    )
    assert rows[0]["source_group"] == "how2sign:source-video"
    assert rows[0]["video"] == "clips/clip-1.mp4"
    assert not any("cache" in key or "baseline" in key for key in rows[0])
    assert validate_inventory(rows, tmp_path) == {"how2sign_aligned": 1}


def test_raw_how2sign_inventory_gets_groups_from_expansion_videos(tmp_path: Path) -> None:
    video_dir = tmp_path / "clips"
    video_dir.mkdir()
    (video_dir / "raw-clip.mp4").write_bytes(b"video")
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema": "umi-studio-how2sign-raw-training-expansion/2",
            "training_only": True,
            "videos": [
                {
                    "video_id": "raw-source",
                    "video_name": "raw-source-5-rgb_front",
                    "signer": 5,
                    "clips": [{"sentence_name": "raw-clip", "reference": "raw text"}],
                }
            ],
        },
    )
    index = tmp_path / "index.jsonl"
    _write_jsonl(
        index,
        [
            {
                "schema": "umi-studio-how2sign-encoder-cache-example/1",
                "split": "training",
                "clip_fps": 25,
                "sentence_name": "raw-clip",
                "reference": "raw text",
                "signer": 5,
                "resampled_clip_sha256": "a" * 64,
            }
        ],
    )
    rows = how2sign_examples(
        workspace=tmp_path,
        source="how2sign_raw",
        index_paths=[index],
        selection_path=selection,
        video_dir=video_dir,
    )
    assert rows[0]["source_group"] == "how2sign:raw-source"
    assert validate_inventory(rows, tmp_path) == {"how2sign_raw": 1}


def test_aligned_inventory_accepts_receipted_subset_across_indexes(tmp_path: Path) -> None:
    video_dir = tmp_path / "clips"
    video_dir.mkdir()
    for name in ("clip-1", "clip-2"):
        (video_dir / f"{name}.mp4").write_bytes(name.encode())
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema": "umi-studio-how2sign-training-aligned/1",
            "rows": [
                {
                    "sentence_name": name,
                    "reference": f"text {name}",
                    "signer": 2,
                    "video_id": f"source-{name}",
                }
                for name in ("clip-1", "clip-2", "missing-clip")
            ],
        },
    )
    indexes = []
    for position, name in enumerate(("clip-1", "clip-2"), start=1):
        path = tmp_path / f"index-{position}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "schema": "umi-studio-how2sign-encoder-cache-example/1",
                    "split": "training",
                    "clip_fps": 25,
                    "sentence_name": name,
                    "reference": f"text {name}",
                    "signer": 2,
                    "resampled_clip_sha256": str(position) * 64,
                }
            ],
        )
        indexes.append(path)
    rows = how2sign_examples(
        workspace=tmp_path,
        source="how2sign_aligned",
        index_paths=indexes,
        selection_path=selection,
        video_dir=video_dir,
        allow_incomplete_selection=True,
    )
    assert [row["example_id"] for row in rows] == ["clip-1", "clip-2"]
    assert rows[0]["source_index_sha256"] != rows[1]["source_index_sha256"]


def test_fleurs_inventory_keeps_document_group_and_signer(tmp_path: Path) -> None:
    video_dir = tmp_path / "clips"
    video_dir.mkdir()
    (video_dir / "example-1.mp4").write_bytes(b"video")
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema": "umi-studio-fleurs-dev-split/1",
            "training": [
                {
                    "example_id": "example-1",
                    "reference": "fleurs text",
                    "signer": 3,
                    "document_id": "document-1",
                }
            ],
        },
    )
    index = tmp_path / "index.jsonl"
    _write_jsonl(
        index,
        [
            {
                "schema": "umi-studio-fleurs-dev-encoder-cache-example/1",
                "split": "training",
                "clip_fps": 25,
                "example_id": "example-1",
                "reference": "fleurs text",
                "signer": 3,
                "document_id": "document-1",
                "clip_sha256": "d" * 64,
            }
        ],
    )
    rows = fleurs_examples(
        workspace=tmp_path,
        index_path=index,
        selection_path=selection,
        video_dir=video_dir,
    )
    assert rows[0]["source_group"] == "fleurs:document-1"
    assert rows[0]["signer"] == 3
    assert validate_inventory(rows, tmp_path) == {"fleurs": 1}


def test_asl_stem_wiki_inventory_uses_verified_source_video(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "stem-1.mp4"
    video.write_bytes(b"video")
    selection = tmp_path / "selection.json"
    record = {
        "schema": "umi-studio-asl-stem-wiki-training-example/1",
        "split": "training",
        "example_id": "stem-1",
        "participant": "P22",
        "article": "Standard score",
        "archive_member": "ASL_STEM_Wiki/videos/stem-1.mp4",
        "archive_file_size": 5,
        "source_group": "asl-stem-wiki:P22:Standard score",
        "reference": "Cluster analysis and multidimensional scaling",
    }
    _write_json(
        selection,
        {
            "schema": "umi-studio-asl-stem-wiki-training-selection/1",
            "training_only": True,
            "selected_examples": 1,
            "records": [record],
        },
    )
    from benchmark_fleurs_dev import sha256

    receipts = tmp_path / "receipts.jsonl"
    _write_jsonl(
        receipts,
        [
            {
                "schema": "umi-studio-asl-stem-wiki-training-video/1",
                "example_id": "stem-1",
                "participant": "P22",
                "article": "Standard score",
                "archive_member": record["archive_member"],
                "selection_sha256": sha256(selection),
                "video_size_bytes": 5,
                "video_sha256": sha256(video),
                "video_probe": {"stream": {"r_frame_rate": "30000/1001"}},
            }
        ],
    )
    rows = asl_stem_wiki_examples(
        workspace=tmp_path,
        selection_path=selection,
        receipts_path=receipts,
        video_dir=video_dir,
    )
    assert rows[0]["source"] == "asl_stem_wiki"
    assert rows[0]["signer"] == 22
    assert rows[0]["video_fps"] == pytest.approx(29.97002997)
    assert validate_inventory(rows, tmp_path) == {"asl_stem_wiki": 1}


def test_precut_how2sign_expansion_uses_original_clip_receipt(tmp_path: Path) -> None:
    from benchmark_fleurs_dev import sha256

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "extra-1.mp4"
    video.write_bytes(b"video")
    record = {
        "sentence_name": "extra-1",
        "video_id": "source-video",
        "signer": 8,
        "reference": "more training text",
        "archive_member": "raw_videos/extra-1.mp4",
        "archive_crc32": 123,
        "archive_file_size": 5,
    }
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema": "umi-studio-how2sign-precut-expansion/1",
            "training_only": True,
            "clips": 1,
            "records": [record],
        },
    )
    receipts = tmp_path / "receipts.jsonl"
    _write_jsonl(
        receipts,
        [
            {
                "schema": "umi-studio-how2sign-precut-expansion-clip/1",
                "selection_sha256": sha256(selection),
                "sentence_name": "extra-1",
                "archive_member": record["archive_member"],
                "archive_crc32": 123,
                "clip_size_bytes": 5,
                "clip_sha256": sha256(video),
                "video_stream": {"r_frame_rate": "24/1"},
            }
        ],
    )
    rows = how2sign_precut_expansion_examples(
        workspace=tmp_path,
        selection_path=selection,
        receipts_path=receipts,
        video_dir=video_dir,
    )
    assert rows[0]["source"] == "how2sign_precut_expansion"
    assert rows[0]["source_group"] == "how2sign:source-video"
    assert rows[0]["video_fps"] == 24
    assert validate_inventory(rows, tmp_path) == {"how2sign_precut_expansion": 1}


def test_precut_how2sign_expansion_rejects_partial_receipts(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema": "umi-studio-how2sign-precut-expansion/1",
            "training_only": True,
            "clips": 1,
            "records": [{"sentence_name": "missing"}],
        },
    )
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text("")
    with pytest.raises(ValueError, match="receipts are incomplete"):
        how2sign_precut_expansion_examples(
            workspace=tmp_path,
            selection_path=selection,
            receipts_path=receipts,
            video_dir=tmp_path,
        )


def test_precut_how2sign_expansion_accepts_only_a_receipted_prefix(
    tmp_path: Path,
) -> None:
    from benchmark_fleurs_dev import sha256

    video = tmp_path / "first.mp4"
    video.write_bytes(b"video")
    records = [
        {
            "sentence_name": name,
            "video_id": "source",
            "signer": 8,
            "reference": "text",
            "archive_member": f"raw_videos/{name}.mp4",
            "archive_crc32": position,
            "archive_file_size": 5,
        }
        for position, name in enumerate(("first", "second"), start=1)
    ]
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "schema": "umi-studio-how2sign-precut-expansion/1",
            "training_only": True,
            "clips": 2,
            "records": records,
        },
    )
    receipt = {
        "schema": "umi-studio-how2sign-precut-expansion-clip/1",
        "selection_sha256": sha256(selection),
        "sentence_name": "first",
        "archive_member": records[0]["archive_member"],
        "archive_crc32": 1,
        "clip_size_bytes": 5,
        "clip_sha256": sha256(video),
        "video_stream": {"r_frame_rate": "24/1"},
    }
    receipts = tmp_path / "receipts.jsonl"
    _write_jsonl(receipts, [receipt])
    rows = how2sign_precut_expansion_examples(
        workspace=tmp_path,
        selection_path=selection,
        receipts_path=receipts,
        video_dir=tmp_path,
        allow_incomplete_selection=True,
    )
    assert [row["example_id"] for row in rows] == ["first"]

    receipt["sentence_name"] = "second"
    receipt["archive_member"] = records[1]["archive_member"]
    receipt["archive_crc32"] = 2
    _write_jsonl(receipts, [receipt])
    with pytest.raises(ValueError, match="receipts are incomplete"):
        how2sign_precut_expansion_examples(
            workspace=tmp_path,
            selection_path=selection,
            receipts_path=receipts,
            video_dir=tmp_path,
            allow_incomplete_selection=True,
        )


def test_inventory_rejects_cross_source_how2sign_duplicate(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    row = {
        "schema": "umi-studio-fresh-lowres-training-example/1",
        "source": "how2sign_aligned",
        "example_id": "same",
        "source_group": "how2sign:doc",
        "signer": 3,
        "reference": "text",
        "video": "video.mp4",
        "video_sha256": "a" * 64,
        "video_fps": 25,
        "source_index_sha256": "b" * 64,
        "source_selection_sha256": "c" * 64,
    }
    duplicate = dict(row, source="how2sign_precut_expansion")
    with pytest.raises(ValueError, match="repeats a How2Sign clip"):
        validate_inventory([row, duplicate], tmp_path)


def test_inventory_rejects_duplicate_source_identity(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    row = {
        "schema": "umi-studio-fresh-lowres-training-example/1",
        "source": "fleurs",
        "example_id": "same",
        "source_group": "fleurs:doc",
        "signer": 3,
        "reference": "text",
        "video": "video.mp4",
        "video_sha256": "a" * 64,
        "video_fps": 25,
        "source_index_sha256": "b" * 64,
        "source_selection_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_inventory([row, dict(row)], tmp_path)
