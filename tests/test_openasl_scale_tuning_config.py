import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_c5_openasl_scale_protocol import RECIPE
from run_c5_stem_projection_frozen_tuning import tuning_config


@pytest.mark.parametrize("recipe_id,rate", [
    ("c5-openasl-scale-projection-frozen-v1", 4e-6),
    ("c5-openasl-scale-fused-lora-fixed-v1", 4e-6),
    ("c5-openasl-scale-fused-lora-high-lr-v1", 1e-4),
])
def test_scale_recipe_cannot_change_after_training(recipe_id, rate):
    recipe = {**copy.deepcopy(RECIPE), "id": recipe_id, "learning_rate": rate}
    plan = {"schema": "umi-studio-c5-frozen-replay-plan/1", "recipe": recipe}
    recipe, schema = tuning_config(plan)
    assert recipe["checkpoint_after_examples"] == [128, 256, 512, 768]
    assert schema == "umi-studio-c5-openasl-scale-tuning-decision/1"
    plan["recipe"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="recipe changed"):
        tuning_config(plan)
