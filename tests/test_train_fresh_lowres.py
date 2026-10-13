from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from train_fresh_lowres import language_artifact_receipt


def test_language_artifact_receipt_accepts_pinned_t5_small(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "t5-small-df1b051c"
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"pinned weights")
    receipt = language_artifact_receipt(tmp_path, artifact)
    assert receipt["initialization"] == "google_t5_small"
    assert receipt["upstream_revision"] == "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
    assert receipt["weights_file"] == "model.safetensors"
    assert len(receipt["weights_sha256"]) == 64


def test_language_artifact_receipt_rejects_unpinned_model(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "unreviewed-model"
    artifact.mkdir(parents=True)
    with pytest.raises(ValueError, match="supported pinned artifact"):
        language_artifact_receipt(tmp_path, artifact)
