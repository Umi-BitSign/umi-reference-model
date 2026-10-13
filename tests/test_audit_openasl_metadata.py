"""OpenASL metadata auditing preserves source groups and protected splits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_openasl_metadata import summarize_rows, timestamp_seconds


def _row(vid: str, yid: str, text: str, split: str = "train") -> dict:
    return {
        "vid": vid,
        "yid": yid,
        "start": "00:00:01.000",
        "end": "00:00:04.000",
        "raw-text": text,
        "tokenized-text": text.lower(),
        "gloss": "",
        "split": split,
    }


def test_timestamp_seconds():
    assert timestamp_seconds("01:02:03.500") == 3723.5


def test_audit_excludes_protected_and_nontraining_rows():
    rows = [
        _row("a", "source-a", "three useful words"),
        _row("b", "source-b", "protected exact text"),
        _row("c", "source-c", "validation has words", "val"),
    ]
    bboxes = {name: [0.1, 0.1, 0.9, 0.9] for name in ("a", "b", "c")}
    summary, eligible = summarize_rows(rows, bboxes, {"protected exact text"})
    assert [row["vid"] for row in eligible] == ["a"]
    assert eligible[0]["source_group"] == "openasl:source-a"
    assert summary["protected_exact_reference_rows"] == 1
    assert summary["rows_by_split"] == {"train": 2, "val": 1}
