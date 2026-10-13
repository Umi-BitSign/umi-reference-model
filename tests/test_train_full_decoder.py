"""Full-decoder training locks and deterministic example order."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_fleurs_dev import sha256
from freeze_full_decoder_plan import RECIPE
from train_full_decoder import check_inputs, scheduled_examples


def test_full_decoder_requires_frozen_hashes_and_scope(tmp_path):
    fleurs = tmp_path / "fleurs.jsonl"
    how2sign = tmp_path / "how2sign.jsonl"
    fleurs.write_text('{"example_id":"fleurs"}\n')
    how2sign.write_text('{"selection_sha256":"selection"}\n')
    plan = {
        "schema": "umi-studio-full-decoder-plan/1",
        "training_examples": {"fleurs": 313, "how2sign": 1024, "flores2m": 0},
        "trainable_scope": "complete ByT5 decoder, decoder embedding and output head",
        "fleurs_repeat": 1,
        "how2sign_repeat": 1,
        "seed": 17,
        "fleurs_training_index_sha256": sha256(fleurs),
        "how2sign_training_index_sha256": sha256(how2sign),
        "how2sign_selection_sha256": "selection",
        "recipes": [RECIPE],
    }
    assert check_inputs(plan, fleurs, how2sign) == RECIPE
    with pytest.raises(ValueError, match="inputs"):
        check_inputs({**plan, "recipes": [{**RECIPE, "learning_rate": 0.01}]}, fleurs, how2sign)
    how2sign.write_text(json.dumps({"selection_sha256": "other"}) + "\n")
    with pytest.raises(ValueError, match="inputs"):
        check_inputs(plan, fleurs, how2sign)


def test_full_decoder_schedule_is_deterministic_and_complete():
    fleurs = [{"id": f"f{i}"} for i in range(313)]
    how2sign = [{"id": f"h{i}"} for i in range(1024)]
    first = scheduled_examples(fleurs, how2sign, 17)
    second = scheduled_examples(fleurs, how2sign, 17)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {row["id"] for row in first} == {row["id"] for row in fleurs + how2sign}
