"""FSboard selection excludes sensitive clips and is signer balanced."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fetch_fsboard_validation import validate_selection
from select_fsboard_validation import select_rows


def _row(signer: str, index: int, *, sensitive: bool = False) -> dict:
    return {
        "clipFilename": f"{signer}-{index}.mp4",
        "clipFileMd5": "a" * 32,
        "clipFileSize": 1000,
        "clipStartTimeS": 0,
        "clipEndTimeS": 5,
        "phrase": f"phrase {signer} {index}",
        "signerId": signer,
        "fullVideoFilename": f"{signer}-recording-{index}.mp4",
        "signingType": "fs",
        "sensitiveContent": sensitive,
    }


def test_selection_is_deterministic_and_balanced():
    rows = [_row(signer, index) for signer in ("a", "b") for index in range(4)]
    first = select_rows(rows, 17, 2)
    assert first == select_rows(list(reversed(rows)), 17, 2)
    assert [row["signer_id"] for row in first] == ["a", "a", "b", "b"]
    assert len({row["source_recording"] for row in first}) == 4


def test_followup_excludes_prior_clips_phrases_and_recordings():
    rows = [_row(signer, index) for signer in ("a", "b") for index in range(5)]
    first = select_rows(rows, 17, 2)
    followup = select_rows(list(reversed(rows)), 17, 2, first)
    assert len(followup) == 4
    assert {row["clip_filename"] for row in first}.isdisjoint(
        {row["clip_filename"] for row in followup}
    )
    assert {row["source_recording"] for row in first}.isdisjoint(
        {row["source_recording"] for row in followup}
    )
    assert {row["reference"] for row in first}.isdisjoint({row["reference"] for row in followup})


def test_sensitive_and_unsafe_clips_are_excluded():
    rows = [_row("a", 0, sensitive=True), _row("a", 1)]
    rows.append({**_row("a", 2), "clipFilename": "../escape.mp4"})
    selected = select_rows(rows, 17, 1)
    assert [row["clip_filename"] for row in selected] == ["a-1.mp4"]


def test_not_enough_recordings_fails():
    rows = [_row("a", 0), {**_row("a", 1), "fullVideoFilename": "a-recording-0.mp4"}]
    with pytest.raises(ValueError, match="distinct validation recordings"):
        select_rows(rows, 17, 2)


def test_downloader_rejects_test_or_unsafe_paths():
    row = select_rows([_row("a", 0)], 17, 1)[0]
    selection = {
        "schema": "umi-studio-fsboard-validation-pilot/1",
        "dataset": "googleai/fsboard",
        "source_split": "validation",
        "test_used": False,
        "rows": [row],
    }
    assert validate_selection(selection) == [row]
    with pytest.raises(ValueError, match="selection"):
        validate_selection({**selection, "source_split": "test"})
    with pytest.raises(ValueError, match="clip"):
        validate_selection({**selection, "rows": [{**row, "source_path": "../test.mp4"}]})
