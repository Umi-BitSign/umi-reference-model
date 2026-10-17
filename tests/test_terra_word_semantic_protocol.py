from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_roi_semantic_learning as shared
import terra_word_semantic_protocol as protocol


class Tokenizer:
    def __call__(self, text, truncation):
        assert truncation is False
        return SimpleNamespace(input_ids=[ord(value) + 3 for value in text] + [1])

    def decode(self, tokens, **kwargs):
        return "".join(chr(value - 3) for value in tokens if value > 2)


def case(identity, reference, frames, participant="P2", role="training"):
    return {
        "record": {
            "original": {
                "row": {
                    "example_id": identity,
                    "reference": reference,
                    "label": reference.lower(),
                    "participant": participant,
                    "role": role,
                    "training_allowed": role == "training",
                }
            },
            "ROI_lineage_training_allowed": role == "training" and participant not in ("P0", "P19"),
            "normalized_probe": {"decoded_frames": frames, "width": 640, "height": 480},
        },
        "receipt": {"path": f"checkpoints/inputs/{identity}.json", "sha256": identity},
    }


def inventory():
    return [
        case("long", "Evolution", 100),
        case("caption", "Electromechanics", 10),
        case("short", "Fish", 3),
        case("other", "Overview", 20),
        case("same_label_short", "Evolution", 1),
        case("z-tie", "History", 100),
    ]


def cases():
    return protocol.select_cases(inventory())


def pool(case, donor, tokenizer):
    row = protocol.training_case(case)
    return [
        {
            "tokens": tokenizer(row["reference"], truncation=False).input_ids,
            "text": row["reference"],
            "kind": "human_anchor",
            "truncated": False,
            "termination": "eos",
        },
        protocol.training_negative(case, donor, tokenizer),
    ]


def pools(values, tokenizer):
    return [
        pool(value, values[(index + 1) % len(values)], tokenizer)
        for index, value in enumerate(values)
    ]


def test_metadata_extremes_and_distinct_labels_are_deterministic_without_reading_dev_text():
    records = inventory()
    development = case("dev", "never read", 10000, "P4", "development")
    development["record"]["original"]["row"].pop("reference")
    records.append(development)
    original = copy.deepcopy(records)
    expected = ["long", "caption", "short"]
    for ordered in (records, list(reversed(records))):
        selected = protocol.select_cases(ordered)
        assert [c["record"]["original"]["row"]["example_id"] for c in selected] == expected
    assert records == original


@pytest.mark.parametrize("participant", ["P0", "P19", "P4", "P10"])
def test_reserved_and_retained_diagnostic_training_cannot_enter_proof(participant):
    with pytest.raises(ValueError, match="roles"):
        protocol.select_cases([*inventory(), case("unsafe", "Private", 200, participant)])


def test_selection_rejects_duplicate_identities_and_insufficient_distinct_labels():
    records = inventory()
    with pytest.raises(ValueError, match="identities"):
        protocol.select_cases([*records, records[0]])
    with pytest.raises(ValueError, match="distinct"):
        protocol.select_cases([records[0], records[4]])


def test_donor_training_permission_is_checked_before_caption_tokenization():
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("held-out label reached tokenizer")

    with pytest.raises(ValueError, match="roles"):
        protocol.training_negative(cases()[0], case("dev", "No", 5, "P4", "development"), forbidden)
    assert not calls


@pytest.mark.parametrize("mutation", ["donor", "tokens", "anchor", "kind", "duplicate"])
def test_candidate_tokens_anchors_and_predeclared_training_donors_cannot_drift(mutation):
    selected, tokenizer = cases(), Tokenizer()
    generated = pools(selected, tokenizer)
    protocol.validate_pools(selected, generated, tokenizer)
    if mutation == "donor":
        generated[0][-1]["donor_input_receipt"] = selected[0]["receipt"]
    elif mutation == "tokens":
        generated[0][-1]["tokens"][0] += 1
    elif mutation == "anchor":
        generated[0][0]["text"] = "changed"
    elif mutation == "kind":
        generated[0][0]["kind"] = "greedy"
    else:
        generated[0].insert(1, copy.deepcopy(generated[0][0]))
    with pytest.raises(ValueError):
        protocol.validate_pools(selected, generated, tokenizer)


def test_actual_training_pair_mapping_and_original_calibration_checker_are_reused():
    selected, tokenizer = cases(), Tokenizer()
    paired = protocol.paired_inputs(selected, pools(selected, tokenizer), tokenizer)
    scores = [0] * len(paired["items"])
    for index, (value, mapping) in enumerate(zip(selected, paired["cases"], strict=True)):
        assert mapping["word_input_receipt"] == value["receipt"]
        assert (
            paired["items"][mapping["indices"][-1]]["candidate"]
            == protocol.training_case(selected[(index + 1) % 3])["reference"]
        )
        scores[mapping["indices"][0]] = 10
    for control in paired["controls"]:
        scores[control["index"]] = control["range"][0]
    assert protocol.qualities is shared.qualities
    assert protocol.qualities(paired, scores) == [[10, 0]] * 3
    tied = list(scores)
    for mapping in paired["cases"]:
        for index in mapping["indices"]:
            tied[index] = 10
    with pytest.raises(ValueError, match="contrasting"):
        protocol.qualities(paired, tied)
    generic = next(control for control in paired["controls"] if control["range"] == [0, 0])
    scores[generic["index"]] = 1
    with pytest.raises(ValueError, match="rubric controls"):
        protocol.qualities(paired, scores)


@pytest.mark.parametrize("duplicate_rollout", [False, True])
def test_pool_preparation_uses_complete_verified_training_clips_and_does_not_substitute_donors(
    monkeypatch, duplicate_rollout
):
    selected, tokenizer, observed = cases(), Tokenizer(), []

    def verified_input(value, purpose):
        assert purpose == "training"
        frames = protocol.expected_header(value["record"])["sampled_frames"]
        return {
            "pixels": np.zeros((frames, 3, 112, 112, 3), dtype=np.uint8),
            "pose_features": np.zeros((frames, 14), dtype=np.float32),
        }

    def candidate_pool(student, tokenizer, clip, mask, reference, **kwargs):
        index = len(observed)
        observed.append(reference)
        assert clip.pixels.shape == (
            protocol.expected_header(selected[index]["record"])["sampled_frames"],
            3,
            3,
            112,
            112,
        )
        assert mask.all() and mask.shape == (1, len(clip.pixels))
        assert kwargs["seed"] == 20261017 + index
        assert kwargs["maximum_tokens"] == 512
        emitted = pool(selected[index], selected[(index + 1) % 3], tokenizer)
        if duplicate_rollout:
            emitted[-1]["kind"] = "greedy"
            emitted[-1].pop("donor_input_receipt")
            return emitted
        return emitted[:1]

    monkeypatch.setattr(protocol, "load_verified_input", verified_input)
    monkeypatch.setattr(protocol, "candidate_pool", candidate_pool)
    if duplicate_rollout:
        with pytest.raises(ValueError, match="no substitution"):
            protocol.prepare_pools(nn.Linear(1, 1), tokenizer, selected)
        assert len(observed) == 1
    else:
        generated = protocol.prepare_pools(nn.Linear(1, 1), tokenizer, selected)
        assert len(observed) == 3
        protocol.validate_pools(selected, generated, tokenizer)
