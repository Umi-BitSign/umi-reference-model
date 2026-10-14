from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_feature_winner_protocol import RECIPE
from tiny_rgb_winner_student_protocol import RECIPE as DIRECT_RECIPE


def test_feature_route_keeps_small_inputs_and_complete_winner_sequence_weights():
    for key in (
        "short_side",
        "target_fps",
        "max_frames",
        "max_text_tokens",
        "frozen_language_parameters",
    ):
        assert RECIPE[key] == DIRECT_RECIPE[key]
    assert RECIPE["capacity"] == "large"
    assert RECIPE["retained_native_semantic_encoder"]
    assert RECIPE["frozen_semantic_encoder_parameters"] == 503411144
    assert RECIPE["trainable_parameter_tensors"] == 90
    assert RECIPE["trainable_parameters"] == 17526750
    assert not RECIPE["initial_diagnostic_weights_reused"]


def test_feature_anchor_weight_change_is_explicit_before_new_numerical_proof():
    assert RECIPE["latent_coefficient"] == DIRECT_RECIPE["latent_coefficient"] == 0.5
    assert RECIPE["feature_coefficient"] == 1.0
    assert DIRECT_RECIPE["feature_coefficient"] == 0.1
