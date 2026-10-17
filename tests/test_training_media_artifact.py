from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from terra_training_recovery import artifact
from training_media_artifact import media_artifact


def test_real_data_file_uses_a_separate_media_binding_and_recovery_guard_stays_strict(tmp_path):
    path = tmp_path / "data/role/clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original source video bytes")
    assert media_artifact(path, root=tmp_path) == {
        "path": "data/role/clip.mp4",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="runs/ or checkpoints/"):
        artifact(path, root=tmp_path)
    old = media_artifact(path, root=tmp_path)
    path.write_bytes(b"changed source bytes")
    assert media_artifact(path, root=tmp_path) != old


@pytest.mark.parametrize("directory", ["runs", "checkpoints", "artifacts"])
def test_media_binding_cannot_adopt_files_from_another_artifact_domain(tmp_path, directory):
    path = tmp_path / directory / "file"
    path.parent.mkdir()
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="data/"):
        media_artifact(path, root=tmp_path)


def test_missing_media_directories_and_role_aliases_are_rejected(tmp_path):
    path = tmp_path / "data/training/clip.mp4"
    path.parent.mkdir(parents=True)
    for candidate in (path, path.parent):
        with pytest.raises(ValueError):
            media_artifact(candidate, root=tmp_path)
    path.write_bytes(b"x")
    (tmp_path / "data/development").symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        media_artifact(tmp_path / "data/development/clip.mp4", root=tmp_path)
