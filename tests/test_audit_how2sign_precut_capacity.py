import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_how2sign_precut_capacity import aligned_eligible, audit


def original(name: str = "clip", signer: int = 1) -> dict:
    return {
        "VIDEO_ID": "video",
        "VIDEO_NAME": f"video-{signer}-rgb_front",
        "SENTENCE_ID": "sentence",
        "SENTENCE_NAME": name,
        "START": "0",
        "END": "4",
        "SENTENCE": "a short sentence here",
    }


def realigned(name: str = "clip", signer: int = 1) -> dict:
    return {
        **{key: value for key, value in original(name, signer).items() if key not in ("START", "END")},
        "START_REALIGNED": "0.2",
        "END_REALIGNED": "4.2",
    }


def test_overlap_filter() -> None:
    assert aligned_eligible(original(), realigned())
    shifted = {**realigned(), "START_REALIGNED": "2", "END_REALIGNED": "6"}
    assert not aligned_eligible(original(), shifted)
    with pytest.raises(ValueError):
        aligned_eligible(original(), {**realigned(), "SENTENCE": "different"})


def test_audit_counts_unselected() -> None:
    source = {"one": original("one"), "two": original("two")}
    aligned = {"one": realigned("one"), "two": realigned("two")}
    selected = [{"sentence_name": "one", "signer": 1, "start_sec": 0.0, "end_sec": 4.0}]
    counts = audit(source, aligned, selected)
    assert counts["metadata_eligible_examples"] == 2
    assert counts["per_signer"]["1"]["metadata_only_unselected_examples"] == 1
