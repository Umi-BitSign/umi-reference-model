from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_stationary_mps as protocol
from c5_full_coverage_mps import GATES as ORIGINAL_GATES
from c5_stationary_mps import GATES, RECIPE, unique_training, validate_teacher_binding


def example():
    return {
        "source": "how2sign_new",
        "id": "train",
        "cache": "data/example.npz",
        "cache_sha256": "cache",
        "reference": "training only",
        "role": "training",
        "training_allowed": True,
        "loss_weight": 2.0,
        "signer": 1,
    }


def exposure(row, index):
    return {
        **row,
        "original_corpus_loss_weight": row["loss_weight"],
        "loss_weight": 1.0,
        "exposure": index,
        "group_exposure": index,
        "repeat_cycle": index - 1,
    }


def test_repeated_exposures_share_exactly_one_immutable_teacher_identity():
    row = example()
    result = unique_training([exposure(row, 1), exposure(row, 2)], [row])
    assert result == {("how2sign_new", "train"): row}


@pytest.mark.parametrize(
    "change",
    [
        {"reference": "changed"},
        {"cache_sha256": "different"},
        {"loss_weight": 2.0},
        {"training_allowed": False},
        {"exposure": 2},
    ],
)
def test_replay_rejects_changed_reference_cache_objective_role_or_position(change):
    row = example()
    candidate = exposure(row, 1)
    candidate.update(change)
    with pytest.raises(ValueError, match="protected training identity"):
        unique_training([candidate], [row])


def test_replay_requires_every_unique_clip_and_no_duplicate_teacher_entries():
    row = example()
    other = {**row, "id": "other"}
    with pytest.raises(ValueError, match="omits"):
        unique_training([exposure(row, 1)], [row, other])
    with pytest.raises(ValueError, match="repeats"):
        unique_training([exposure(row, 1)], [row, row])


def test_teacher_binding_pins_reference_cache_parent_and_training_role():
    row = example()
    record = {
        **row,
        "parent_checkpoint_sha256": "parent",
        "reference_sha256": hashlib.sha256(row["reference"].encode()).hexdigest(),
        "input_policy": "verified_native224_legacy_post_mediapipe_stride2",
    }
    validate_teacher_binding(record, row, "parent")
    for key in (
        "reference_sha256",
        "cache_sha256",
        "parent_checkpoint_sha256",
        "role",
        "input_policy",
    ):
        changed = deepcopy(record)
        changed[key] = "changed"
        with pytest.raises(ValueError, match="reused teacher identity"):
            validate_teacher_binding(changed, row, "parent")


def test_stationary_recipe_keeps_original_quality_and_resource_limits():
    assert GATES == ORIGINAL_GATES
    assert RECIPE["training_exposures"] == 48_800
    assert RECIPE["unique_training_examples"] == 7529
    assert RECIPE["gradient_accumulation"] == 4
    assert RECIPE["maximum_process_mps_bytes"] == 6 * 1024**3
    assert (
        RECIPE["teacher_validation"] == "all_unique_training_prefixes_before_any_optimizer_update"
    )


def test_teacher_tensor_checks_exact_frozen_bytes(tmp_path, monkeypatch):
    target = tmp_path / "teacher.npz"
    values = np.zeros((1, 4, 384), dtype=np.float32)
    np.savez_compressed(target, logits=values)
    record = {
        "teacher_path": str(target),
        "teacher_sha256": protocol.sha256(target),
        "shape": [1, 4, 384],
    }
    monkeypatch.setattr(protocol, "data_path", lambda value: Path(value))
    assert np.array_equal(protocol.read_teacher(record), values)
    np.savez_compressed(target, logits=values + 1)
    with pytest.raises(ValueError, match="tensor bytes changed"):
        protocol.read_teacher(record)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((1, 4, 384), dtype=np.float64),
        np.full((1, 4, 384), float("nan"), dtype=np.float32),
        np.zeros((1, 5, 384), dtype=np.float32),
    ],
)
def test_teacher_tensor_rejects_wrong_dtype_nonfinite_values_or_shape(
    values, tmp_path, monkeypatch
):
    target = tmp_path / "teacher.npz"
    np.savez_compressed(target, logits=values)
    record = {
        "teacher_path": str(target),
        "teacher_sha256": protocol.sha256(target),
        "shape": [1, 4, 384],
    }
    monkeypatch.setattr(protocol, "data_path", lambda value: Path(value))
    with pytest.raises(ValueError, match="shape, dtype or finiteness"):
        protocol.read_teacher(record)


@pytest.mark.parametrize("initial", [None, "0"])
def test_launcher_sets_fallback_before_trainer_import(initial):
    launcher = Path(__file__).resolve().parents[1] / "tools/launch_c5_stationary_mps.py"
    script = """
import importlib.abc, importlib.util, os, runpy, sys
class Spy(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "train_c5_stationary_mps":
            assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
            assert os.environ["OMP_NUM_THREADS"] == "2"
            assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec): return None
    def exec_module(self, module): module.main = lambda: print("import-env-ok")
sys.meta_path.insert(0, Spy())
runpy.run_path(sys.argv[1], run_name="__main__")
"""
    environment = dict(os.environ)
    if initial is None:
        environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    else:
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = initial
    result = subprocess.run(
        [sys.executable, "-c", script, str(launcher)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "import-env-ok"
