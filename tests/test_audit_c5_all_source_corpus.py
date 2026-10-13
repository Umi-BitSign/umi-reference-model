import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_c5_all_source_corpus import add_declared_face_protection, exclusion_reason
from benchmark_fleurs_dev import sha256


def test_rejects_other_clip_from_protected_source_and_matching_reference():
    groups = {"how2sign": {"abcdefghijk"}, "fleurs": {"dev_document_1"}, "openasl": {"old-source"}}
    assert exclusion_reason(
        {"source": "how2sign_new", "id": "abcdefghijk_99-1-rgb_front", "reference": "different text"},
        groups, set(),
    ) == "protected_how2sign_source"
    assert exclusion_reason(
        {"source": "fleurs", "id": "dev_document_1_sentence_99_signer_3", "reference": "different text"},
        groups, set(),
    ) == "protected_fleurs_document"
    assert exclusion_reason(
        {"source": "asl_stem_original", "id": "new", "reference": "The same phrase."},
        groups, {("the", "same", "phrase")},
    ) == "protected_reference"
    assert exclusion_reason(
        {"source": "openasl_scale", "id": "new", "youtube_id": "new-source", "reference": "new phrase"},
        groups, set(),
    ) is None


def frozen_plan(tmp_path, payload):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload))
    snapshot = tmp_path / "source-snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(json.dumps({"plan_sha256": sha256(plan)}))
    return plan


def test_pilot_holdout_protects_all_actors_and_other_source_references(tmp_path):
    plan = frozen_plan(tmp_path, {
        "schema": "umi-studio-c5-face-learning-pilot-plan/1",
        "examples": [
            {"role": "training", "training_allowed": True, "source_group": "how2sign:TRAINING123", "reference": "training text"},
            {"role": "pilot_source_holdout", "training_allowed": False, "source_group": "how2sign:abcdefghijk", "reference": "Held-out phrase."},
        ],
    })
    groups = {"how2sign": set(), "fleurs": set(), "openasl": set()}
    references = set()
    add_declared_face_protection(plan, groups, references)
    assert groups["how2sign"] == {"abcdefghijk"}
    assert exclusion_reason(
        {"source": "how2sign_new", "id": "abcdefghijk_99-11-rgb_front", "reference": "different"},
        groups, references,
    ) == "protected_how2sign_source"
    assert exclusion_reason(
        {"source": "asl_stem_new", "id": "different", "reference": "held out phrase"},
        groups, references,
    ) == "protected_reference"


def test_fresh_face_benchmark_remains_protected_after_scoring(tmp_path):
    plan = frozen_plan(tmp_path, {
        "schema": "umi-studio-c5-fresh-face-benchmark-plan/1",
        "training_allowed": False,
        "examples": [{"video_id": "abcdefghijk", "reference": "fresh phrase"}],
    })
    groups = {"how2sign": set(), "fleurs": set(), "openasl": set()}
    add_declared_face_protection(plan, groups, set())
    assert groups["how2sign"] == {"abcdefghijk"}


def test_changed_protected_plan_fails_closed(tmp_path):
    plan = frozen_plan(tmp_path, {
        "schema": "umi-studio-c5-fresh-face-benchmark-plan/1",
        "training_allowed": False, "examples": [],
    })
    plan.write_text(plan.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen source snapshot"):
        add_declared_face_protection(plan, {"how2sign": set()}, set())
