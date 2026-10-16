from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_roi_short_translation as evaluation


def development():
    return {
        "reserved_STEM_participants": ["P4"],
        "records": [
            {
                "source": "asl_stem_original",
                "example_id": str(i),
                "role": "development",
                "training_allowed": False,
                "signer": "P4",
            }
            for i in range(34)
        ],
    }


def test_reserved_coverage_is_exact_and_outer_permissions_are_authoritative():
    data = development()
    assert len(evaluation.reserved_rows(data)) == 34
    data["records"][0]["original_pair"] = {"raw": {"training_allowed": True}}
    assert len(evaluation.reserved_rows(data)) == 34
    data["records"][0]["training_allowed"] = True
    with pytest.raises(ValueError):
        evaluation.reserved_rows(data)


def test_cannot_drop_repeat_or_substitute_reserved_cases():
    data = development()
    for change in ("drop", "repeat", "source", "signer"):
        bad = copy.deepcopy(data)
        if change == "drop":
            bad["records"].pop()
        elif change == "repeat":
            bad["records"][1] = bad["records"][0]
        elif change == "source":
            bad["records"][0]["source"] = "private_service"
        else:
            bad["records"][0]["signer"] = "training_person"
        with pytest.raises(ValueError):
            evaluation.reserved_rows(bad)


def test_control_text_must_match_exactly_before_candidate_decode():
    evaluation.require_control("A test.", "A test.")
    for value in ("a test.", "A test", "unrelated"):
        with pytest.raises(ValueError):
            evaluation.require_control(value, "A test.")


def test_frozen_decode_retains_published_horizon_and_has_no_reference_input():
    seen = {}

    class Language:
        def generate(self, **kwargs):
            seen.update(kwargs)
            return torch.tensor([[0, 2, 1]])

    tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=1, decode=lambda tokens, **kw: "output"
    )
    memory = torch.ones(1, 7, 1536)
    generation = {
        "num_beams": 5,
        "max_length": 2048,
        "length_penalty": 0.6,
        "no_repeat_ngram_size": 16,
    }
    text, tokens = evaluation.decode(Language(), tokenizer, generation, memory)
    assert text == "output" and tokens == [0, 2, 1]
    assert seen["encoder_outputs"].last_hidden_state is memory
    assert all(seen[k] == v for k, v in generation.items())
    assert seen["do_sample"] is False and seen["early_stopping"] is True
    assert not {"labels", "input_ids", "decoder_input_ids", "reference"} & seen.keys()


def test_predicted_streams_use_every_frame_and_only_original_video_pose(monkeypatch):
    batches = []

    class Batch:
        def __init__(self, count):
            self.count = count

        def to(self, device):
            assert device == "mps"
            return self

    def appearance(batch):
        batches.append(batch.count)
        return {
            n: torch.full((batch.count, 384), float(i + 1))
            for i, n in enumerate(evaluation.STREAMS)
        }

    monkeypatch.setattr(evaluation, "pixel_tensor", lambda pixels: Batch(len(pixels)))
    original = {n: np.full((9, 384), 999, dtype=np.float32) for n in evaluation.STREAMS}
    original["pose_features"] = np.ones((9, 14), dtype=np.float32)
    duty = SimpleNamespace(run=lambda f: f())
    result = evaluation.predicted_streams(
        appearance, np.zeros((9, 3, 64, 64, 3), dtype=np.uint8), original, duty
    )
    assert batches == [8, 1]
    assert result["face_features"].shape == (9, 384)
    assert torch.all(result["face_features"] == 1)
    assert torch.all(result["right_hand_features"] == 3)
    result["pose_features"].zero_()
    assert original["pose_features"].sum() == 126


def test_pose_timeline_mismatch_is_rejected_before_any_appearance_call():
    original = {n: np.ones((3, 384), dtype=np.float32) for n in evaluation.STREAMS}
    original["pose_features"] = np.ones((2, 14), dtype=np.float32)
    with pytest.raises(ValueError):
        evaluation.predicted_streams(
            None, np.zeros((3, 3, 64, 64, 3), dtype=np.uint8), original, None
        )


def test_prediction_preserves_development_role_and_reports_only_diagnostic_WER():
    row = {**development()["records"][0], "source_group": "group", "reference": "A test"}
    result = evaluation.prediction(row, "A")
    assert result["training_allowed"] is False and result["role"] == "development"
    assert result["WER_diagnostic"] == 0.5
    assert not evaluation.POLICY["promotion_authorized"]
