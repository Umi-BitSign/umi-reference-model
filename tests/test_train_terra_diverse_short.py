from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_terra_diverse_short as module


def test_candidate_selection_description_is_the_only_optimizer_recipe_change():
    assert {k for k in module.PROFILE if module.PROFILE[k] != module.RECIPE[k]} == {
        "candidate_selection"
    }
    assert module.PROFILE["local_steps_per_outer_batch"] == 16
    assert module.PROFILE["WER_is_diagnostic_only"] is True
    assert module.PROFILE["training_Terra_requests"] == 0


def test_profile_reuses_numerics_evaluation_and_checkpoint_code_without_modifying_parent():
    before = module.original_driver.checked
    driver = module.runtime()
    for name in ("run", "fit", "checked_proof"):
        assert (
            getattr(driver, name).__code__.co_code
            == getattr(module.original_driver, name).__code__.co_code
        )
    assert driver.evaluate is module.original_driver.evaluate
    assert driver.batches is module.original_driver.batches
    assert driver.rate is module.original_driver.rate
    assert driver.checked is module.checked
    assert driver.checked_rewards is module.graded_rewards
    assert driver.RECIPE is module.PROFILE
    assert module.original_driver.checked is before
    assert module.original_driver.RECIPE is module.RECIPE


def test_changed_driver_cannot_be_loaded(monkeypatch):
    monkeypatch.setattr(module, "DRIVER_SHA", "changed")
    with pytest.raises(ValueError, match="driver changed"):
        module.runtime()
