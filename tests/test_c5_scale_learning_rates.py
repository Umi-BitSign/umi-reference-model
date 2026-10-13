import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from c5_scale_learning_rates import SPLIT_RATES, optimizer_groups, split_recipe
from freeze_c5_openasl_scale_protocol import RECIPE
from run_c5_stem_projection_frozen_tuning import tuning_config


@pytest.mark.parametrize("recipe_id,rates", SPLIT_RATES.items())
def test_groups_cover_parameters_once_and_preserve_fixed_rates(recipe_id, rates):
    parameters = {
        "encoder.adapter.signhubert.q.lora_b.weight": torch.nn.Parameter(torch.zeros(2)),
        "decoder.block.0.q.lora_b.weight": torch.nn.Parameter(torch.zeros(3)),
    }
    recipe = split_recipe(RECIPE, recipe_id)
    groups = optimizer_groups(parameters, recipe)
    assert [group["lr"] for group in groups] == list(rates)
    assert {id(parameter) for group in groups for parameter in group["params"]} == {
        id(parameter) for parameter in parameters.values()
    }
    tuning_config({"schema": "umi-studio-c5-frozen-replay-plan/1", "recipe": recipe})
    recipe["visual_learning_rate"] *= 2
    with pytest.raises(ValueError, match="rates differ"):
        optimizer_groups(parameters, recipe)
    with pytest.raises(ValueError, match="recipe changed"):
        tuning_config({"schema": "umi-studio-c5-frozen-replay-plan/1", "recipe": recipe})


def test_split_groups_reject_unknown_or_shared_parameters():
    visual = torch.nn.Parameter(torch.zeros(2))
    decoder = torch.nn.Parameter(torch.zeros(2))
    recipe = split_recipe(RECIPE, next(iter(SPLIT_RATES)))
    with pytest.raises(ValueError, match="exact visual/decoder partition"):
        optimizer_groups({"encoder.signhubert.a": visual, "decoder.a": decoder, "direct.a": decoder}, recipe)
    with pytest.raises(ValueError, match="repeats a parameter"):
        optimizer_groups({"encoder.signhubert.a": visual, "decoder.a": visual}, recipe)
