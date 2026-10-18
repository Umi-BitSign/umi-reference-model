from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingerspelling_spans as spans


def test_native_packet_order_and_variable_frame_durations_preserve_presentation_time():
    probe = {
        "streams": [
            {"codec_name": "h264", "start_time": "0", "nb_frames": "3", "time_base": "1/1000"}
        ],
        "format": {"start_time": "0"},
        "packets": [
            {"pts": 0, "duration": 30},
            {"pts": 100, "duration": 50},
            {"pts": 30, "duration": 70},
        ],
    }
    clock = spans.native_clock(probe)
    assert clock == [Fraction(0), Fraction(3, 100), Fraction(1, 10), Fraction(3, 20)]
    assert spans.mapped_intervals([[1, 3]], clock) == [(Fraction(3, 100), Fraction(3, 20))]


def test_native_frames_are_never_mistaken_for_canonical_25fps_indices():
    clock = [Fraction(index, 30) for index in range(301)]
    interval = spans.mapped_intervals([[240, 300]], clock)
    result = spans.rasterize(interval, 250, Fraction(10))
    assert sum(result["sample_is_fingerspelling"]) == 50
    assert not result["sample_is_fingerspelling"][199]
    assert result["sample_is_fingerspelling"][200]


def test_overlapping_spans_do_not_double_count_positive_occupancy():
    result = spans.rasterize(
        [(Fraction(1, 100), Fraction(3, 100)), (Fraction(2, 100), Fraction(1, 20))],
        2,
        Fraction(2, 25),
    )
    assert result["positive_frame_cell_fraction"] == [0.75, 0.25]
    assert result["sample_is_fingerspelling"] == [False, True]


def test_partial_source_tail_is_explicit_and_not_treated_as_a_full_negative_frame():
    result = spans.rasterize([], 3, Fraction(3, 50))
    assert result["covered_frame_cell_fraction"] == [1.0, 0.5, 0.0]
    assert result["sample_annotation_valid"] == [True, True, False]
    assert result["positive_frame_cell_fraction"] == [0.0, 0.0, 0.0]


def test_unpaired_text_is_preserved_without_assigning_guessed_word_labels():
    parsed = spans.parse_annotations("[(0, 2), (3, 5)]", "['one']")
    assert parsed["spans"] == [[0, 2], [3, 5]]
    assert parsed["labels"] == ["one"]
    assert parsed["paired_labels"] is None and not parsed["text_pairing_valid"]


def test_malformed_boolean_text_is_preserved_for_review_without_guessing_a_word():
    parsed = spans.parse_annotations("[(0, 2)]", "[True]")
    assert parsed["labels"] == [True]
    assert not parsed["label_text_valid"] and not parsed["text_pairing_valid"]
    assert parsed["paired_labels"] is None


@pytest.mark.parametrize(
    "value", ["[(-1, 2)]", "[(2, 2)]", "[(3, 1)]", "[(True, 3)]", "[(0, 2.5)]"]
)
def test_invalid_annotation_endpoints_are_rejected(value):
    with pytest.raises(ValueError, match="endpoints"):
        spans.parse_annotations(value, "['label']")


def test_out_of_range_native_annotations_are_rejected_without_clipping():
    # Actual clock-audit case: a 148-frame source and a published [169,234) span.
    # A guessed 30fps interpretation would fit, but this API never switches clocks.
    with pytest.raises(ValueError, match="exceeds native"):
        spans.mapped_intervals([[169, 234]], [Fraction(frame, 15) for frame in range(149)])
