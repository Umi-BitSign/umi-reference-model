"""The fallback flag must exist before trainer dependencies import PyTorch."""

from __future__ import annotations

import builtins
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("previous_fallback", [None, "0"])
def test_launcher_sets_fallback_before_trainer_import(monkeypatch, previous_fallback):
    # Register restoration even when the original environment lacked the key.
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if previous_fallback is None:
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    original_import = builtins.__import__
    events = []

    def import_module(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "train_c5_face_learning_pilot":
            assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
            assert os.environ["OMP_NUM_THREADS"] == "2"
            assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
            events.append("import")
            return SimpleNamespace(main=lambda: events.append("train"))
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_module)
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/launch_c5_face_learning_pilot.py"),
        run_name="__main__",
    )
    assert events == ["import", "train"]
