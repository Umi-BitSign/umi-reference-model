"""ASL STEM Wiki metadata acquisition rejects changed archive structure."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fetch_asl_stem_wiki_metadata import METADATA_MEMBERS, validate_inventory


def _info(name: str, *, directory: bool = False) -> zipfile.ZipInfo:
    return zipfile.ZipInfo(name + ("/" if directory else ""))


def test_accepts_metadata_and_exact_video_count():
    infos = [_info(name) for name in METADATA_MEMBERS]
    infos.extend(
        _info(f"ASL_STEM_Wiki/videos/video-{number}.mp4")
        for number in range(2)
    )
    by_name = validate_inventory(infos, expected_videos=2)
    assert set(METADATA_MEMBERS) <= set(by_name)


def test_rejects_missing_or_nonvideo_members():
    infos = [_info(name) for name in METADATA_MEMBERS]
    infos.append(_info("ASL_STEM_Wiki/videos/video.txt"))
    with pytest.raises(ValueError, match="video inventory"):
        validate_inventory(infos, expected_videos=1)
    with pytest.raises(ValueError, match="metadata members"):
        validate_inventory(infos[1:], expected_videos=1)
