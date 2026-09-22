from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("filename", ["worker.py", "worker.amd64.py"])
@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_decoder_filter_and_encoder_have_separate_thread_bounds(
    filename: str, mirrored: bool, scaled: bool
) -> None:
    path = Path(__file__).resolve().parents[1] / "docker" / "mediapipe-holistic" / filename
    spec = importlib.util.spec_from_file_location("thread_bound_worker", path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    command = worker._decode_command(
        video_proc_path="/input/synthetic.mp4",
        video_descriptor=42,
        media=SimpleNamespace(
            source_displayed_width=640,
            source_displayed_height=360,
            derived_width=320 if scaled else 640,
            derived_height=180 if scaled else 360,
        ),
        selection=SimpleNamespace(source_pts=(0, 2048, 2048, 4096)),
        clip=SimpleNamespace(start_us=0),
        input_mirrored=mirrored,
    )
    input_at = command.index("-i")
    assert command.index("-threads") < input_at < command.index("-threads:v")
    for flag in ("-threads", "-filter_threads", "-threads:v"):
        assert command[command.index(flag) + 1] == "1"
    # Resource bounds preserve selected PTS, mirror/scale order and raw RGB output.
    filters = command[command.index("-vf") + 1]
    assert filters.startswith(worker._select_filter([0, 2048, 4096]))
    assert ("hflip" in filters) == mirrored
    assert ("scale=320:180:flags=" + worker.SCALE_FLAGS in filters) == scaled
    assert filters.endswith("setsar=1/1")
    assert command[command.index("-frames:v") + 1] == "3"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[-3:] == ["-f", "rawvideo", "pipe:1"]
