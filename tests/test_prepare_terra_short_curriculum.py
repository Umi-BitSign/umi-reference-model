from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_terra_short_curriculum as module


def pair(i, *, reference="Carbon telephone transmitter", signer="P0"):
    return {
        "raw": {
            "source": "asl_stem_original",
            "example_id": f"clip{i}",
            "reference": reference,
            "signer": signer,
            "source_group": f"asl_stem:{signer}:article{i}",
            "role": "training",
            "training_allowed": True,
        },
        "header": {"source_frames": 100, "source_fps": 25},
        "feature_pretraining_allowed": True,
    }


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setitem(module.POLICY, "minimum_training_clips", 1)
    monkeypatch.setitem(module.POLICY, "minimum_development_clips", 1)
    return [pair(0), pair(1, signer="P4")]


def test_complete_labels_and_original_pairs_preserved_with_new_outer_roles(small):
    original = copy.deepcopy(small)
    selected = module.select(small, {"P4"})
    assert small == original
    assert selected[1]["role"] == "development"
    assert selected[1]["training_allowed"] is False
    assert selected[1]["original_pair"]["raw"]["training_allowed"] is True
    assert module.training_pairs(selected) == [small[0]]


@pytest.mark.parametrize(
    "label",
    [
        "{{harvnb|Feynman Vol.",
        "m(t)&quot;}},&quot;i&quot;",
        "December 13, 2019.",
        "This is too long.",
    ],
)
def test_markup_dates_and_long_text_cannot_become_atomic_training_examples(small, label):
    small.append(pair(2, reference=label))
    assert len(module.select(small, {"P4"})) == 2


def test_complete_two_word_hyphenated_label_is_not_rewritten(small):
    small.append(pair(2, reference="Space-filling polyhedra"))
    selected = module.select(small, {"P4"})
    assert selected[-1]["reference"] == "Space-filling polyhedra"


@pytest.mark.parametrize("kind", ["role", "permission", "duplicate", "feature_permission"])
def test_protected_or_duplicate_original_source_is_rejected(small, kind):
    if kind == "role":
        small[0]["raw"]["role"] = "open_development"
    elif kind == "permission":
        small[0]["raw"]["training_allowed"] = False
    elif kind == "duplicate":
        small.append(copy.deepcopy(small[0]))
    else:
        small[0]["feature_pretraining_allowed"] = False
    with pytest.raises(ValueError):
        module.select(small, {"P4"})


def test_duration_uses_original_timeline_not_assumed_sampling_rate(small):
    small.append(pair(2))
    small[-1]["header"] = {"source_frames": 300, "source_fps": 10}
    assert len(module.select(small, {"P4"})) == 2


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), "25", True])
def test_invalid_original_frame_rate_is_rejected(small, fps):
    small[0]["header"]["source_fps"] = fps
    with pytest.raises(ValueError):
        module.select(small, {"P4"})


@pytest.mark.parametrize("permission", [True, 0, 1, None])
def test_development_cannot_enter_training_via_rewritten_outer_permission(small, permission):
    selected = module.select(small, {"P4"})
    selected[1]["training_allowed"] = permission
    with pytest.raises(ValueError):
        module.training_pairs(selected)


def test_absent_development_or_insufficient_training_is_not_a_valid_split(small):
    with pytest.raises(ValueError):
        module.select(small, set())
    with pytest.raises(ValueError):
        module.select(small, {"P0", "P4"})
