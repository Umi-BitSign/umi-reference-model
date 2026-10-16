from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import roi_appearance_protocol as pilot
import train_roi_appearance as shared
import train_roi_short_corpus as broad


def test_broader_recipe_changes_only_epoch_budget_and_fixed_diagnostic_times():
    changed = {k for k in pilot.RECIPE if pilot.RECIPE[k] != broad.RECIPE[k]}
    assert changed == {"epochs", "evaluation_epochs"}
    assert broad.RECIPE["epochs"] == 8
    assert broad.RECIPE["evaluation_epochs"] == [0, 1, 4, 8]
    assert broad.RECIPE["seed"] == pilot.SEED
    assert broad.RECIPE["batch_frames"] == 8


def test_proof_cases_include_longest_and_supplemental_but_not_diagnostic():
    records = [
        {"frames": count, "roi_training_allowed": train, "original_row": {"source": source}}
        for count, train, source in (
            (10, True, "asl_stem_original"),
            (149, True, "asl_stem_original"),
            (40, True, "asl_stem_supplemental"),
            (999, False, "asl_stem_supplemental"),
        )
    ]
    assert broad.proof_indices(records) == [0, 1, 2]
    with pytest.raises(ValueError):
        broad.proof_indices([records[-1]])


def test_reusable_builder_uses_explicit_recipe_and_complete_frame_inventory(monkeypatch):
    class Model:
        def parameters(self):
            return [torch.ones(2)]

        def state_dict(self):
            return {"weight": torch.ones(2)}

    monkeypatch.setattr(pilot, "RoiAppearance", Model)
    monkeypatch.setattr(pilot, "sha256", lambda p: "hash")
    monkeypatch.setattr(pilot, "artifact", lambda p: {"path": p.name, "sha256": "hash"})
    monkeypatch.setattr(pilot, "runtime_binding", dict)
    plan = {
        "source_sha256": {},
        "resource_policy": "budget",
        "resource_policy_sha256": "hash",
        "ROI_diagnostic_participants": ["P0", "P19"],
        "reserved_STEM_participants": ["P4"],
    }
    rows = [{"roi_training_allowed": v} for v in (True, True, False)]
    arrays = [{"pixels": np.zeros((n, 1), dtype=np.uint8)} for n in (7, 2, 100)]
    result = pilot.model_specification(
        plan, {"records": rows}, arrays, Path("plan"), Path("data"), broad.RECIPE
    )
    assert result["training_frames_per_epoch"] == 9
    assert result["planned_frame_exposures"] == 72
    assert result["planned_optimizer_updates"] == 16
    assert result["recipe"] is broad.RECIPE
    assert result["feature_gate"] == pilot.GATE
    assert result["translation_promotion_authorized"] is False


def test_shared_runner_accepts_explicit_protocol_without_mutating_module_globals():
    assert "protocol" in inspect.signature(shared.run).parameters
    assert "validate" in inspect.signature(shared.qualify).parameters
    assert "validate" in inspect.signature(shared.fit).parameters
    assert "data_path" in inspect.signature(shared.checked_proof).parameters
    assert "batch_frames" in inspect.signature(shared.evaluate).parameters
    assert pilot.RECIPE["epochs"] == 64
    assert pilot.RECIPE["evaluation_epochs"] == [0, 1, 4, 16, 64]
