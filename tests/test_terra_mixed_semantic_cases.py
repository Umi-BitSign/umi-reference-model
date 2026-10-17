from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import terra_mixed_native_input as mixed
import terra_mixed_semantic_cases as protocol
import test_terra_mixed_native_input as inputs


def phrase(ident, frames, caption):
    case = {"kind": "phrase", "phrase": inputs.phrase_record(ident, frames=frames * 2)}
    row = case["phrase"]["original_row"]
    row["reference"] = caption
    row["original_pair"]["raw"]["reference"] = caption
    row["original_pair"]["teacher"]["reference"] = caption
    return case


@pytest.fixture
def small_inventory():
    return [
        {"kind": "word", "word": inputs.word_case("word-z", frames=6)},
        phrase("phrase-z", 149, "Long complete timeline"),
        phrase("phrase-a", 149, "Different complete timeline"),
        phrase("long-caption", 30, "An extensivelylongcaptionword anotherlongcaptionword"),
        {"kind": "word", "word": inputs.word_case("word-a", frames=6, label="Other")},
    ]


def test_case_selection_covers_longest_phrase_caption_and_shortest_easier_word(small_inventory):
    selected = protocol.select_cases(small_inventory)
    rows = protocol.summary(selected)
    assert [row["example_id"] for row in rows] == ["phrase-a", "long-caption", "word-a"]
    assert [row["sampled_frames"] for row in rows] == [149, 30, 3]
    assert [row["kind"] for row in rows] == ["phrase", "phrase", "word"]
    assert protocol.donor_indices(selected) == [1, 2, 0]
    assert protocol.POLICY["model_initializer_selected"] is False
    assert protocol.POLICY["optimizer_recipe_selected"] is False


def test_selection_is_stable_under_inventory_order_and_unrelated_grade_metadata(small_inventory):
    expected = protocol.select_cases(small_inventory)
    random.Random(27).shuffle(small_inventory)
    assert protocol.select_cases(small_inventory) == expected
    for case in small_inventory:
        if case["kind"] == "phrase":
            case["phrase"]["original_row"]["irrelevant_historical_grade"] = 0
    assert [protocol.identity(case) for case in protocol.select_cases(small_inventory)] == [
        protocol.identity(case) for case in expected
    ]


def test_longest_caption_uses_complete_UTF8_bytes():
    cases = [
        phrase("timeline", 149, "Longest clip"),
        phrase("utf8", 40, "é" * 12 + " x"),
        phrase("ascii", 30, "abcdefghijklmnopqrst y"),
        {"kind": "word", "word": inputs.word_case("word", frames=6)},
    ]
    selected = protocol.select_cases(cases)
    assert protocol.identity(selected[1])[1] == "utf8"
    assert protocol.summary(selected)[1]["complete_reference_UTF8_bytes"] == 26


def test_development_roles_are_rejected_before_ranking_headers(small_inventory, monkeypatch):
    small_inventory[-1] = {
        "kind": "word",
        "word": inputs.word_case("reserved", signer="P4", role="development"),
    }
    monkeypatch.setattr(
        protocol, "expected_header", lambda *a: pytest.fail("ranked forbidden role")
    )
    with pytest.raises(ValueError):
        protocol.select_cases(small_inventory)


@pytest.mark.parametrize("mutation", ["duplicate_id", "duplicate_reference", "no_word"])
def test_selector_rejects_incomplete_or_ambiguous_training_cases(small_inventory, mutation):
    if mutation == "duplicate_id":
        small_inventory[1] = phrase("word-z", 149, "Another phrase")
    elif mutation == "duplicate_reference":
        small_inventory[:] = [
            phrase("first", 149, "Complete phrase."),
            phrase("second", 30, "COMPLETE PHRASE"),
            small_inventory[0],
        ]
    else:
        small_inventory[:] = [case for case in small_inventory if case["kind"] == "phrase"]
    with pytest.raises(ValueError):
        protocol.select_cases(small_inventory)


@pytest.fixture
def complete_binding():
    """Synthetic receipt metadata for binding tests, never actual native-input evidence."""
    inventory = inputs.full_inventory.__wrapped__()
    cases, _ = mixed.select_inputs(*inventory)
    cases[214] = phrase("phrase0", 84, "An extensivelylongcaptionword anotherlongcaptionword")
    bound = []
    for index, case in enumerate(cases):
        header = mixed.expected_header(case)
        frames = header["sampled_frames"]
        arrays = {
            "pixels": {"shape": [frames, 3, 112, 112, 3], "dtype": "uint8", "sha256": "0" * 64},
            "pose_features": {"shape": [frames, 14], "dtype": "float32", "sha256": "1" * 64},
            "frame_indices": {
                "shape": [frames],
                "dtype": "int64",
                "sha256": header["frame_indices_sha256"],
            },
        }
        if case["kind"] == "word":
            case["word"]["verified_input"] = {"arrays": copy.deepcopy(arrays)}
        saved = {
            "case": case,
            "header": header,
            "arrays": arrays,
            "streamed_pixels_equal_native_reference": True,
            "original_pose_and_easier_word_inputs_preserved": True,
        }
        bound.append(
            {
                "case": case,
                "receipt": {
                    "path": f"checkpoints/mixed/input-{index:04d}.json",
                    "sha256": f"{index:064x}",
                },
                "verified_input": saved,
            }
        )
    verified = {
        "records": bound,
        "counts": {"word": 214, "phrase": 318},
        "model_training_qualification_still_required": True,
    }
    return copy.deepcopy(protocol.select_cases(cases)), verified


def test_proposal_rebinds_to_the_same_three_complete_actual_receipt_descriptors(complete_binding):
    selected, verified = complete_binding
    bound = protocol.bind_cases(selected, verified)
    assert [record["case"] for record in bound] == selected
    assert len({record["receipt"]["path"] for record in bound}) == 3
    assert [mixed.expected_header(record["case"])["sampled_frames"] for record in bound] == [
        101,
        84,
        6,
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "counts",
        "case_order",
        "case_binding",
        "array_shape",
        "word_hash",
        "receipt_alias",
    ],
)
def test_rebinding_cannot_select_new_cases_or_accept_changed_preflight_evidence(
    complete_binding, mutation
):
    selected, verified = complete_binding
    if mutation == "incomplete":
        verified["records"].pop()
    elif mutation == "counts":
        verified["counts"]["word"] = 213
    elif mutation == "case_order":
        selected[:2] = selected[1::-1]
    elif mutation == "case_binding":
        selected[0]["phrase"]["landmarks"]["sha256"] = "f" * 64
    else:
        bound = protocol.bind_cases(selected, verified)
        if mutation == "array_shape":
            bound[0]["verified_input"]["arrays"]["pixels"]["shape"][2] = 64
        elif mutation == "word_hash":
            bound[-1]["verified_input"]["arrays"]["pixels"]["sha256"] = "f" * 64
        else:
            bound[0]["receipt"] = copy.deepcopy(bound[1]["receipt"])
    with pytest.raises(ValueError):
        protocol.bind_cases(selected, verified)
