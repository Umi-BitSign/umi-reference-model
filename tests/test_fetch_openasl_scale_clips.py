from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fetch_openasl_scale_clips import download_clip, media_probe


def test_download_clip_uses_exact_section_and_fallback(tmp_path: Path) -> None:
    row = {
        "youtube_id": "abcdefghijk",
        "start_sec": 12.3456,
        "end_sec": 17.8912,
    }
    results = [
        SimpleNamespace(returncode=1, stderr="blocked"),
        SimpleNamespace(returncode=0, stderr=""),
    ]
    with patch("fetch_openasl_scale_clips.subprocess.run", side_effect=results) as run:
        strategy, attempts = download_clip(tmp_path, 7, row)
    assert strategy == "hls"
    assert [item["strategy"] for item in attempts] == ["https", "hls"]
    command = run.call_args.args[0]
    assert "*12.346-17.891" in command
    assert "--force-keyframes-at-cuts" in command
    assert "--concurrent-fragments" in command
    assert any("openasl-scale-0007-abcdefghijk.%(ext)s" in item for item in command)


def test_video_duration_excludes_track_timestamp_delay(tmp_path: Path) -> None:
    import json

    payload = {
        "streams": [{"duration": "3.003", "start_time": "2.961292",
                     "width": 720, "height": 480, "avg_frame_rate": "24000/1001"}],
        "format": {"duration": "5.964292"},
    }
    with patch("fetch_openasl_scale_clips.subprocess.run",
               return_value=SimpleNamespace(stdout=json.dumps(payload))):
        result = media_probe(tmp_path / "clip.mp4")
    assert result["duration_sec"] == 3.003
    assert result["container_duration_sec"] == 5.964292
    assert result["video_start_time_sec"] == 2.961292
