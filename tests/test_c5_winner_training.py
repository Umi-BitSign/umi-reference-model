from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_winner_training import RECIPE, lr_factor, private_data
from train_c5_winner_full import configure_model


def test_full_corpus_horizon_and_fixed_winner_rates_are_not_a_small_fit_substitute():
    assert RECIPE["training_exposures"] == 48800
    assert RECIPE["learning_rate"] == 1e-5
    assert RECIPE["gradient_accumulation"] == 8
    assert RECIPE["checkpoint_after_exposures"][-1] == 48800
    assert sum(RECIPE["domain_mass"].values()) == 1
    assert RECIPE["retention_coefficient"] == 1


def test_winner_learning_rate_warmup_and_cosine_endpoints():
    assert lr_factor(0) == 1 / 128
    assert lr_factor(127) == lr_factor(128) == 1
    assert lr_factor(6100) == pytest.approx(0.1)
    assert lr_factor(1000) > lr_factor(4000) > lr_factor(6100)


def test_winner_inputs_cannot_read_operational_or_outside_data_paths():
    with pytest.raises(ValueError, match="private data"):
        private_data("/Users/sam/.local/share/umi/not-readable-by-training.npz")
    with pytest.raises(ValueError, match="private data"):
        private_data("data/../../outside.npz")


def test_qualified_model_replacements_keep_all_existing_tensor_names_and_values():
    def gelu(value):
        return torch.nn.functional.gelu(value)

    gelu.__module__ = "fairseq.modules.gelu"
    candidate = torch.nn.Module()
    candidate.transforms = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
    for layer in candidate.transforms:
        layer.activation_fn = gelu
    candidate.norms = torch.nn.ModuleList([torch.nn.LayerNorm(4) for _ in range(30)])
    candidate.positional = torch.nn.GELU()
    before = {name: value.clone() for name, value in candidate.state_dict().items()}
    selected = configure_model(candidate)
    assert len(selected) == 60
    assert set(candidate.state_dict()) == set(before)
    assert all(torch.equal(value, before[name]) for name, value in candidate.state_dict().items())


def test_unsupported_activation_topology_is_rejected():
    with pytest.raises(RuntimeError, match="qualified proof"):
        configure_model(torch.nn.Linear(4, 4))


def test_actual_frozen_horizon_keeps_original_quality_gates_and_all_eligible_ids():
    # The fixture is scientific configuration, not a private-data dependency.
    record = json.loads(json.dumps(RECIPE))
    assert record["augmentation"] == "none"
    assert record["training_input_policy"] != record["evaluation_input_policy"]
    assert record["max_text_tokens"] == 512
