from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import normalize_terra_word_data as normalize


def original():
    return {"fps": 29.97, "duration_sec": 3.0, "decoded_frames": 90, "width": 640, "height": 480}


def test_complete_25fps_video_keeps_geometry_and_duration():
    actual = {**original(), "fps": 25, "decoded_frames": 75}
    normalize.validate_output(original(), actual)
    actual["duration_sec"] = 3.04
    actual["decoded_frames"] = 76
    normalize.validate_output(original(), actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fps", 30),
        ("width", 320),
        ("height", 240),
        ("duration_sec", 2.5),
        ("duration_sec", 3.5),
        ("duration_sec", float("nan")),
        ("decoded_frames", 60),
    ],
)
def test_cropped_resized_partial_or_different_clock_video_is_rejected(field, value):
    actual = {**original(), "fps": 25, "decoded_frames": 75}
    actual[field] = value
    with pytest.raises(ValueError):
        normalize.validate_output(original(), actual)


def test_command_uses_whole_interval_and_never_crops_resizes_pads_or_cuts():
    command = normalize.command(Path("source.mp4"), Path("output.mp4"))
    assert not {"-ss", "-t", "-to", "-s"} & set(command)
    assert command[command.index("-vf") + 1] == "fps=25"
    assert command[command.index("-i") + 1] == "source.mp4"
    assert command[-1] == "output.mp4"
    assert command[command.index("-crf") + 1] == "18"
    assert all(command[i + 1] == "2" for i, value in enumerate(command) if value == "-threads")
    assert "-noautorotate" in command and "-nostdin" in command


def test_storage_guard_reserves_full_remaining_output_with_unchanged_physical_floor(
    tmp_path, monkeypatch
):
    policy = normalize.POLICY
    required = (
        policy["physical_free_floor_bytes"]
        + normalize.RAW_POLICY["temporary_headroom_bytes"]
        + normalize.RAW_POLICY["checkpoint_reserve_bytes"]
        + policy["maximum_total_derivative_bytes"]
    )
    monkeypatch.setattr(normalize.shutil, "disk_usage", lambda path: SimpleNamespace(free=required))
    normalize.guard(tmp_path, 64 * 1024**2)
    monkeypatch.setattr(
        normalize.shutil, "disk_usage", lambda path: SimpleNamespace(free=required - 1)
    )
    with pytest.raises(RuntimeError, match="floor"):
        normalize.guard(tmp_path)


def test_role_aliases_and_output_budget_overruns_are_rejected(tmp_path, monkeypatch):
    (tmp_path / "aliased-role").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        normalize.guard(tmp_path)
    policy = copy.deepcopy(normalize.POLICY)
    monkeypatch.setitem(normalize.POLICY, "maximum_total_derivative_bytes", 1)
    with pytest.raises(RuntimeError, match="budget"):
        normalize.guard(tmp_path / "new", 2)
    assert policy["physical_free_floor_bytes"] == 80 * 1024**3


def test_preparation_preserves_original_roles_without_authorizing_model_training():
    assert normalize.POLICY["model_training_ready"] is False
    assert normalize.POLICY["ROI_lineage_future_training_excluded_participants"] == ["P0", "P19"]
    assert normalize.POLICY["optimizer_updates"] == normalize.POLICY["new_Terra_requests"] == 0
