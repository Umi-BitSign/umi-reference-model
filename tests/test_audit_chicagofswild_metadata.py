import csv
import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_chicagofswild_metadata import FIELDS, MEMBERS, audit_archive


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    for index, row in enumerate(rows):
        writer.writerow({"": str(index), **row})
    return stream.getvalue().encode()


def row(
    filename: str,
    *,
    partition: str,
    signer: int,
    url: str,
    label: str = "abc",
) -> dict[str, str]:
    return {
        "filename": filename,
        "url": url,
        "start_time": "00:00:01.000",
        "number_of_frames": "12",
        "width": "640",
        "height": "360",
        "label_proc": label,
        "label_raw": label,
        "label_notes": "",
        "partition": partition,
        "signer": str(signer),
    }


def write_archive(path: Path, first: list[dict], second: list[dict]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(MEMBERS[0], csv_bytes(first))
        archive.writestr(MEMBERS[1], csv_bytes(second))


def test_audit_reports_split_overlap_and_conservative_filter(tmp_path: Path) -> None:
    first = [
        row("train-a", partition="train", signer=1, url="https://example/a"),
        row("train-b", partition="train", signer=2, url="https://example/b"),
        row("dev", partition="dev", signer=1, url="https://example/c"),
        row("test", partition="test", signer=3, url="https://example/b"),
    ]
    second = [
        row("plus-train", partition="train", signer=10, url=""),
        row("plus-dev", partition="dev", signer=11, url=""),
        row("plus-test", partition="test", signer=12, url=""),
    ]
    archive = tmp_path / "csv.zip"
    write_archive(archive, first, second)

    result = audit_archive(archive)
    summary = result["members"][MEMBERS[0]]
    assert summary["metadata_rows"] == 4
    assert summary["split_overlap"]["train_dev"]["shared_signers"] == ["1"]
    assert summary["split_overlap"]["train_test"]["shared_usable_source_urls"] == [
        "https://example/b"
    ]
    assert summary["conservative_train_filter"] == {
        "publisher_train_rows": 2,
        "rows_excluded_for_held_out_signer": 1,
        "rows_excluded_for_held_out_usable_source_url": 1,
        "rows_excluded_by_union": 2,
        "remaining_rows": 0,
        "training_rows_without_usable_source_url": 0,
        "source_url_check_complete": True,
    }
    assert result["frame_archives_downloaded"] is False


def test_audit_rejects_duplicate_filenames(tmp_path: Path) -> None:
    duplicated = [
        row("same", partition="train", signer=1, url=""),
        row("same", partition="dev", signer=2, url=""),
    ]
    valid = [row("other", partition="test", signer=3, url="")]
    archive = tmp_path / "csv.zip"
    write_archive(archive, duplicated, valid)

    with pytest.raises(ValueError, match="duplicate filename"):
        audit_archive(archive)
