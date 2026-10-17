from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_terra_word_landmarks as audit
from terra_training_recovery import artifact


@pytest.mark.parametrize("mutation", [None, "different_worker", "failed_exit", "failure_receipt"])
def test_complete_audit_requires_authoritative_same_worker_exit0(tmp_path, monkeypatch, mutation):
    directory = tmp_path / "runs/batch"
    directory.mkdir(parents=True)
    monkeypatch.setattr(audit, "artifact", partial(artifact, root=tmp_path))
    (directory / "worker.json").write_text(json.dumps({"pid": 71}))
    (directory / "exit.json").write_text(
        json.dumps(
            {
                "pid": 72 if mutation == "different_worker" else 71,
                "exit_code": 1 if mutation == "failed_exit" else 0,
            }
        )
    )
    if mutation == "failure_receipt":
        (directory / "failure.json").write_text("{}")
    if mutation is None:
        assert audit.terminal(directory) == {
            "worker": artifact(directory / "worker.json", root=tmp_path),
            "terminal": artifact(directory / "exit.json", root=tmp_path),
        }
    else:
        with pytest.raises(ValueError, match="same-worker"):
            audit.terminal(directory)
