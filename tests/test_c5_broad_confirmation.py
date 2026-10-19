from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import freeze_c5_broad_confirmation as selector


def how_recording(signer, group, name=None):
    name = name or f"{group}-{signer}"
    clips = [
        {
            "sentence_name": f"{name}-{i}",
            "video_name": name,
            "video_id": group,
            "signer": signer,
            "reference": f"Unique {name} sentence number {i}",
            "start_sec": i * 4,
            "end_sec": i * 4 + 3,
        }
        for i in range(2)
    ]
    source = {
        "video_name": name,
        "video_id": group,
        "signer": signer,
        "compressed_size": 100,
        "file_size": 120,
        "candidate_clips": [r["sentence_name"] for r in clips],
    }
    return source, clips


def test_protection_extraction_uses_metadata_identity_and_reference_not_hypotheses():
    sources, texts = set(), set()
    selector.protections(
        {
            "records": [
                {
                    "source_group": "openasl:abc",
                    "youtube_id": "def",
                    "reference": "Hello, WORLD!",
                    "hypothesis": "DO NOT USE THIS",
                }
            ],
            "other": {"VIDEO_ID": "ghi", "SENTENCE": "Another reference here"},
        },
        sources,
        texts,
    )
    assert sources == {"abc", "def", "ghi"}
    assert texts == {"hello world", "another reference here"}


def test_balanced_matching_reassigns_shared_content_instead_of_dropping_a_signer(monkeypatch):
    monkeypatch.setattr(selector, "SIGNERS", (1, 2))
    monkeypatch.setattr(selector, "GROUPS_PER_SIGNER", 1)
    pool = []
    for signer, group in [(1, "a"), (1, "b"), (2, "a"), (2, "c")]:
        recording, clips = how_recording(signer, group)
        pool.append({"recording": recording, "clips": clips})
    chosen, clips = selector.choose_how2sign(pool)
    assert [(r["recording"]["signer"], r["recording"]["video_id"]) for r in chosen] == [
        (1, "b"),
        (2, "a"),
    ]
    assert len(clips) == 4


def test_missing_distinct_content_coverage_fails_closed(monkeypatch):
    monkeypatch.setattr(selector, "SIGNERS", (1, 2))
    monkeypatch.setattr(selector, "GROUPS_PER_SIGNER", 1)
    pool = []
    for signer in (1, 2):
        recording, clips = how_recording(signer, "shared")
        pool.append({"recording": recording, "clips": clips})
    with pytest.raises(ValueError, match="all signer/content groups"):
        selector.choose_how2sign(pool)


def test_recording_name_and_signer_join_is_checked_before_selection():
    source, clips = how_recording(1, "a")
    clips[0]["video_name"] = "other"
    with pytest.raises(ValueError, match="mixes identities"):
        selector.how2sign_pool(clips, [source], set(), set())


def test_how2sign_source_reference_and_byte_exclusions_are_deterministic():
    a, ac = how_recording(1, "a")
    b, bc = how_recording(2, "b")
    before = copy.deepcopy([a, b, ac, bc])
    pool = selector.how2sign_pool(ac + bc, [a, b], set(), set())
    assert pool == selector.how2sign_pool(list(reversed(ac + bc)), [b, a], set(), set())
    assert selector.how2sign_pool(ac, [a], {"a"}, set()) == []
    assert selector.how2sign_pool(ac, [a], set(), {selector.normalized(ac[0]["reference"])}) == []
    a["compressed_size"] = selector.MAX_COMPRESSED + 1
    assert selector.how2sign_pool(ac, [a], set(), set()) == []
    a["compressed_size"] = before[0]["compressed_size"]
    assert [a, b, ac, bc] == before


def open_rows():
    rows = []
    boxes = {}
    for i in range(5):
        vid = f"clip{i}"
        rows.append(
            {
                "vid": vid,
                "yid": f"source{i}",
                "split": "valid",
                "start": "00:00:01.0",
                "end": "00:00:05.0",
                "raw-text": f"Unique sentence number {i}",
            }
        )
        boxes[vid] = [0.0, 0.0, 1.0, 1.0]
    rows.append(
        {
            "vid": "training",
            "yid": "source0",
            "split": "train",
            "start": "00:00:10",
            "end": "00:00:15",
            "raw-text": "Separate training sentence here",
        }
    )
    boxes["training"] = [0.0, 0.0, 1.0, 1.0]
    return rows, boxes


def test_official_split_overlap_never_substitutes_for_local_source_exclusion():
    rows, boxes = open_rows()
    pool = selector.openasl_pool(rows, boxes, set(), set())
    assert "source0" in pool
    assert all(r["official_split"] == "valid" for records in pool.values() for r in records)
    excluded = selector.openasl_pool(rows, boxes, {"source0"}, set())
    assert "source0" not in excluded


@pytest.mark.parametrize("fault", ["duplicate_id", "bbox", "clock", "reference"])
def test_openasl_bad_metadata_and_reference_overlap_cannot_enter_cohort(fault):
    rows, boxes = open_rows()
    texts = set()
    if fault == "duplicate_id":
        rows.append(dict(rows[0]))
    elif fault == "bbox":
        boxes["clip0"] = [0, 0, float("nan"), 1]
    elif fault == "clock":
        rows[0]["end"] = "00:99:10"
    elif fault == "reference":
        texts.add(selector.normalized(rows[0]["raw-text"]))
    assert "source0" not in selector.openasl_pool(rows, boxes, set(), texts)


def test_primary_and_fallback_sources_are_separate_and_ignore_input_order(monkeypatch):
    monkeypatch.setattr(selector, "OPENASL_PRIMARY", 2)
    monkeypatch.setattr(selector, "OPENASL_FALLBACK", 1)
    rows, boxes = open_rows()
    first = selector.choose_openasl(selector.openasl_pool(rows, boxes, set(), set()), set())
    second = selector.choose_openasl(
        selector.openasl_pool(list(reversed(rows)), boxes, set(), set()), set()
    )
    assert first == second
    primary, fallback = first
    assert len(primary) == 2 and len(fallback) == 1
    assert not {r["youtube_id"] for r in primary} & {r["youtube_id"] for r in fallback}
    with pytest.raises(ValueError, match="primary/fallback coverage"):
        selector.choose_openasl({"one": [primary[0]]}, set())


def test_missing_inputs_refuse_before_creating_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(selector, "ROOT", tmp_path)
    output = tmp_path / "runs/reservation"
    with pytest.raises(ValueError, match="metadata input missing"):
        selector.freeze(output)
    assert not output.exists()
