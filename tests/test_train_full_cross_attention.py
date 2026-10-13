"""Full cross-attention trainer input locks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from train_full_cross_attention import check_inputs


def test_full_cross_attention_requires_frozen_hashes_and_recipe(tmp_path):
    fleurs = tmp_path / "fleurs.jsonl"
    how2sign = tmp_path / "how2sign.jsonl"
    fleurs.write_text('{"example_id":"fleurs"}\n')
    how2sign.write_text('{"selection_sha256":"selection"}\n')
    recipe = {
        "id": "full_cross_attention_e1",
        "epochs": 1,
        "learning_rate": 0.00003,
        "weight_decay": 0.1,
        "label_smoothing": 0.2,
    }
    plan = {
        "schema": "umi-studio-full-cross-attention-plan/1",
        "training_examples": {"fleurs": 313, "how2sign": 1024, "flores2m": 0},
        "trainable_scope": "decoder cross-attention q, k, v and o projection weights only",
        "fleurs_repeat": 1,
        "how2sign_repeat": 1,
        "seed": 17,
        "fleurs_training_index_sha256": sha256(fleurs),
        "how2sign_training_index_sha256": sha256(how2sign),
        "how2sign_selection_sha256": "selection",
        "recipes": [recipe],
    }
    assert check_inputs(plan, fleurs, how2sign) == recipe
    changed = {**plan, "recipes": [{**recipe, "learning_rate": 0.01}]}
    with pytest.raises(ValueError, match="recipe"):
        check_inputs(changed, fleurs, how2sign)
    how2sign.write_text(json.dumps({"selection_sha256": "other"}) + "\n")
    with pytest.raises(ValueError, match="inputs"):
        check_inputs(plan, fleurs, how2sign)
