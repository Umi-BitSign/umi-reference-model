from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingertip_probe_split as splitter


def record(number, source="asl_stem_word", signer=None, group=None):
    signer = signer or f"P{number + 40}"
    return {
        "source": source,
        "signer": signer,
        "source_group": group or f"asl_stem:{signer}:article",
        "example_id": str(number),
        "canonical_video_sha256": hashlib.sha256(str(number).encode()).hexdigest(),
    }


def assignment_map(result):
    return {
        (item["source"], item["example_id"]): (item["component_id"], item["role"])
        for item in result["assignments"]
    }


def test_same_stem_signer_is_linked_across_word_original_and_supplemental_tranches():
    rows = [record(index) for index in range(6)]
    rows += [record(9, "asl_stem_original", "P40"), record(10, "asl_stem_supplemental", "P40")]
    result = assignment_map(splitter.split(rows))
    assert result[("asl_stem_word", "0")] == result[("asl_stem_original", "9")]
    assert result[("asl_stem_word", "0")] == result[("asl_stem_supplemental", "10")]


def test_source_groups_and_exact_video_bytes_form_transitive_components():
    rows = [
        record(index, "how2sign_precut_expansion", str(index), f"how2sign:{index}")
        for index in range(6)
    ]
    rows[1]["source_group"] = rows[0]["source_group"]
    rows[2]["canonical_video_sha256"] = rows[1]["canonical_video_sha256"]
    result = splitter.split(rows)
    mapping = assignment_map(result)
    assert (
        mapping[("how2sign_precut_expansion", "0")] == mapping[("how2sign_precut_expansion", "2")]
    )
    assert len(result["components"]) == 4


def test_openasl_unknown_people_are_grouped_by_source_without_claiming_person_disjointness():
    rows = [
        record(index, "openasl_scale", "unknown", f"openasl:video{index}") for index in range(6)
    ]
    rows.append(record(9, "openasl_scale", "another-placeholder", "openasl:video0"))
    result = splitter.split(rows)
    assert len(result["components"]) == 6
    assert result["openasl_person_disjointness_proven"] is False
    assert all(item["disjointness_basis"] == "source_video" for item in result["assignments"])
    assert (
        assignment_map(result)[("openasl_scale", "0")]
        == assignment_map(result)[("openasl_scale", "9")]
    )


def test_roles_are_permutation_stable_and_ignore_targets_captions_and_outcomes():
    rows = [record(index) for index in range(12)]
    before = copy.deepcopy(rows)
    expected = assignment_map(splitter.split(rows))
    assert rows == before
    for item in rows:
        item.update(targets="unavailable", measured_accuracy=1e9, weak_clip_caption="changed")
    assert assignment_map(splitter.split(list(reversed(rows)))) == expected


def test_every_input_is_assigned_once_and_all_roles_exist_when_feasible():
    rows = [record(index) for index in range(11)]
    result = splitter.split(rows)
    assert len(result["assignments"]) == len(rows)
    assert {item["record_index"] for item in result["assignments"]} == set(range(len(rows)))
    assert {item["role"] for item in result["assignments"]} == set(splitter.ROLES)


@pytest.mark.parametrize(
    "change", [{"source": "new_source"}, {"signer": "P10"}, {"source_group": "wrong"}]
)
def test_unknown_or_reserved_identity_conventions_are_rejected(change):
    row = record(0)
    row.update(change)
    with pytest.raises(ValueError):
        splitter.split([row])
