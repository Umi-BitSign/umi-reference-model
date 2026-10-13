"""Bounded split-archive reads and raw-video split checks."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fetch_how2sign_raw_video import checked_training_video
from how2sign_split_remote_zip import SplitRemoteZip, parse_central_directory


def _archive():
    source = b"video-bytes-" * 2048
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("raw_videos/abc-5-rgb_front.mp4", source)
    payload = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        directory = payload[archive.start_dir : payload.rfind(zipfile.stringEndArchive)]
    entries = parse_central_directory(directory, 2)
    boundary = len(payload) // 2
    parts = (payload[:boundary], payload[boundary:])
    manifest = {
        "schema": "umi-studio-how2sign-raw-front-train-archive/1",
        "archive_parts": [
            {"name": f"part{number}", "file_id": f"id{number}", "size_bytes": len(part)}
            for number, part in enumerate(parts)
        ],
        "total_size_bytes": len(payload),
    }
    remote = SplitRemoteZip(
        manifest, part_factory=lambda part: io.BytesIO(parts[int(part["file_id"][-1])])
    )
    remote._entries = entries
    return remote, entries, source


def test_reads_member_across_part_boundary_and_checks_crc():
    remote, entries, source = _archive()
    name = "raw_videos/abc-5-rgb_front.mp4"
    assert entries[name].file_size == len(source)
    assert b"".join(remote.iter_member(name)) == source
    assert set(remote._readers) == {0, 1}


def test_rejects_download_above_member_limit():
    remote, entries, _ = _archive()
    name = "raw_videos/abc-5-rgb_front.mp4"
    with pytest.raises(ValueError, match="download size limit"):
        list(remote.iter_member(name, max_compressed_bytes=entries[name].compressed_size - 1))


def test_raw_video_must_be_training_only():
    _, entries, _ = _archive()
    entry = entries["raw_videos/abc-5-rgb_front.mp4"]
    training = [{"VIDEO_ID": "abc", "VIDEO_NAME": "abc-5-rgb_front"}]
    assert checked_training_video(training, [], "abc-5-rgb_front", entry) == ("abc", 5)
    with pytest.raises(ValueError, match="official validation"):
        checked_training_video(training, [{"VIDEO_ID": "abc"}], "abc-5-rgb_front", entry)
