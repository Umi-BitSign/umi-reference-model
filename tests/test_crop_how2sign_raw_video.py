"""Publisher-realigned raw training interval selection checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cache_how2sign_encoder import checked_receipts
from crop_how2sign_raw_video import select_rows


def _row(name: str, start: float, end: float, sentence: str = "three signed words") -> dict:
    return {
        "VIDEO_ID": "source",
        "VIDEO_NAME": "source-5-rgb_front",
        "SENTENCE_NAME": name,
        "START_REALIGNED": str(start),
        "END_REALIGNED": str(end),
        "SENTENCE": sentence,
    }


def _receipt() -> dict:
    return {
        "video_name": "source-5-rgb_front",
        "video_id": "source",
        "signer": 5,
        "video_sha256": "source-hash",
        "video_probe": {"duration_sec": 20.0},
    }


def test_selects_nonoverlapping_training_intervals_in_time_order():
    rows = [
        _row("later", 8, 12),
        _row("excluded", 5, 7),
        _row("short", 1, 2),
        _row("earlier", 2, 5),
    ]
    selected = select_rows(rows, _receipt(), {"excluded"})
    assert [row["sentence_name"] for row in selected] == ["earlier", "later"]
    assert selected[0]["signer"] == 5
    assert selected[0]["source_video_sha256"] == "source-hash"


def test_rejects_overlapping_or_out_of_range_intervals():
    with pytest.raises(ValueError, match="overlap"):
        select_rows([_row("one", 1, 4), _row("two", 3, 6)], _receipt(), set())
    with pytest.raises(ValueError, match="outside"):
        select_rows([_row("one", 18, 21)], _receipt(), set())


def test_explicit_source_duration_filter_keeps_other_intervals():
    rows = [_row("valid", 2, 5), _row("truncated", 18, 21)]
    assert [row["sentence_name"] for row in select_rows(rows, _receipt(), set(), True)] == ["valid"]
    with pytest.raises(ValueError, match="outside"):
        select_rows([_row("invalid", -1, 2)], _receipt(), set(), True)


def test_raw_encoder_requires_matching_training_crop_receipt():
    row = select_rows([_row("one", 2, 5)], _receipt(), set())[0]
    selection = {
        "schema": "umi-studio-how2sign-training-raw-realigned/1",
        "training_only": True,
        "source_video_sha256": "source-hash",
        "source_video_receipt_sha256": "receipt-hash",
        "rows": [row],
    }
    record = {
        **row,
        "schema": "umi-studio-how2sign-raw-realigned-clip/1",
        "selection_sha256": "selection-hash",
        "source_video_receipt_sha256": "receipt-hash",
        "clip_sha256": "clip-hash",
    }
    assert set(checked_receipts(selection, "selection-hash", [json.dumps(record)])) == {"one"}
    with pytest.raises(ValueError, match="differs from source interval"):
        checked_receipts(
            selection,
            "selection-hash",
            [json.dumps({**record, "start_sec": 3.0})],
        )
    with pytest.raises(ValueError, match="training-only"):
        checked_receipts({**selection, "training_only": False}, "selection-hash", [])
