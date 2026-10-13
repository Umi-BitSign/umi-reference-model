from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_asl_stem_wiki_landmarks import container_command


def test_container_command_is_cpu_bounded_and_offline(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    clip = workspace / "data/example.mp4"
    command = container_command(
        docker_socket="unix:///ignored.sock",
        clip=clip,
        output_name="example.npz",
        landmark_dir=tmp_path,
    )
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cpus") + 1] == "2"
    assert command[command.index("--memory") + 1] == "3g"
    assert "/workspace/data/example.mp4" in command
    assert command[-1] == "/output/example.npz"


def test_container_command_rejects_clip_outside_workspace(tmp_path: Path) -> None:
    clip = tmp_path / "outside.mp4"
    with pytest.raises(ValueError):
        container_command(
            docker_socket="unix:///ignored.sock",
            clip=clip,
            output_name="outside.npz",
            landmark_dir=tmp_path,
        )
