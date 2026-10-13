"""Clip-receipt reuse checks for a larger aligned training selection."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rebind_how2sign_clip_receipts import SOURCE_FIELDS, rebound_receipts


def _fixture(tmp_path: Path):
    common = {key: f"same-{key}" for key in SOURCE_FIELDS}
    rows = []
    receipts = []
    for number in range(3):
        name = f"sentence{number}"
        clip = f"clip-{number}".encode()
        row = {
            "sentence_name": name,
            "archive_member": f"raw_videos/{name}.mp4",
            "archive_crc32": number,
            "archive_file_size": len(clip),
            "reference": f"Reference {number}",
        }
        rows.append(row)
        if number < 2:
            (tmp_path / f"{name}.mp4").write_bytes(clip)
            receipts.append(
                {
                    "schema": "umi-studio-how2sign-training-clip/1",
                    "sentence_name": name,
                    "selection_sha256": "prior-selection-hash",
                    "archive_member": row["archive_member"],
                    "archive_crc32": row["archive_crc32"],
                    "clip_size_bytes": len(clip),
                    "clip_sha256": hashlib.sha256(clip).hexdigest(),
                }
            )
    prior = {"schema": "umi-studio-how2sign-training-aligned/1", **common, "rows": rows[:2]}
    expanded = {"schema": "umi-studio-how2sign-training-aligned/1", **common, "rows": rows}
    return prior, expanded, receipts


def _rebind(tmp_path: Path, prior: dict, expanded: dict, receipts: list[dict]) -> list[dict]:
    return rebound_receipts(
        prior,
        expanded,
        receipts,
        tmp_path,
        "prior-selection-hash",
        "expanded-selection-hash",
        "prior-receipts-hash",
    )


def test_rebinds_only_verified_prior_clips(tmp_path):
    prior, expanded, receipts = _fixture(tmp_path)
    rebound = _rebind(tmp_path, prior, expanded, receipts)
    assert [record["sentence_name"] for record in rebound] == ["sentence0", "sentence1"]
    assert {record["selection_sha256"] for record in rebound} == {"expanded-selection-hash"}
    assert {record["reuse_source"]["prior_clip_receipts_sha256"] for record in rebound} == {
        "prior-receipts-hash"
    }
    assert receipts[0]["selection_sha256"] == "prior-selection-hash"


def test_rejects_changed_row_receipt_or_clip(tmp_path):
    prior, expanded, receipts = _fixture(tmp_path)
    expanded["rows"][0] = {**expanded["rows"][0], "reference": "changed"}
    with pytest.raises(ValueError, match="exact subset"):
        _rebind(tmp_path, prior, expanded, receipts)
    expanded["rows"][0] = prior["rows"][0]
    receipts[0]["archive_crc32"] = 99
    with pytest.raises(ValueError, match="prior receipt"):
        _rebind(tmp_path, prior, expanded, receipts)
    receipts[0]["archive_crc32"] = 0
    (tmp_path / "sentence0.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="downloaded clip"):
        _rebind(tmp_path, prior, expanded, receipts)
    del prior["metadata_sha256"]
    with pytest.raises(ValueError, match="different training sources"):
        _rebind(tmp_path, prior, expanded, receipts)
