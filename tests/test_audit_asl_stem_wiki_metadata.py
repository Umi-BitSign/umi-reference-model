"""ASL STEM Wiki audit filters missing, duplicate, and protected examples."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_asl_stem_wiki_metadata import summarize_rows


def _row(filename: str, sentence: str, participant: str = "P0") -> dict:
    return {
        "participant": participant,
        "article": "Article",
        "section index": "0",
        "sentence index": filename,
        "video filename": filename,
        "video length": "5.0",
        "sentence text": sentence,
    }


def test_audit_keeps_only_available_unique_unprotected_rows():
    rows = [
        _row("good.mp4", "This is a usable sentence."),
        _row("protected.mp4", "Protected reference words."),
        _row("missing.mp4", "This video is absent."),
        _row("duplicate.mp4", "First duplicate sentence."),
        _row("duplicate.mp4", "Second duplicate sentence."),
    ]
    archive = {
        name: {"compressed_size": 100}
        for name in ("good.mp4", "protected.mp4", "duplicate.mp4", "orphan.mp4")
    }
    summary, eligible = summarize_rows(
        rows, archive, {"protected reference words"}
    )
    assert [row["video filename"] for row in eligible] == ["good.mp4"]
    assert summary["metadata_rows_missing_archive_video"] == 1
    assert summary["archive_videos_missing_metadata_row"] == 1
    assert summary["duplicate_filename_rows"] == 2
    assert summary["protected_exact_reference_rows"] == 1
