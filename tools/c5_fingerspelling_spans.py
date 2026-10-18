"""Preserve publisher fingerspelling intervals and map native frame clocks.

The publisher implementation slices [start:end], so intervals here are half
open. These are fingerspelling annotations, not general lexical sign spans.
"""

from __future__ import annotations

import ast
from fractions import Fraction
from itertools import pairwise


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_annotations(span_text, labels_text):
    spans, labels = ast.literal_eval(span_text), ast.literal_eval(labels_text)
    require(isinstance(spans, list) and isinstance(labels, list), "expected annotation lists")
    valid_text = all(isinstance(label, str) and bool(label.strip()) for label in labels)
    for span in spans:
        require(isinstance(span, tuple | list) and len(span) == 2, "invalid interval pair")
        start, end = span
        require(
            type(start) is int and type(end) is int and 0 <= start < end,
            "interval endpoints must be increasing nonnegative integers",
        )
    # Unmatched label lists remain explicit. Detection intervals can still be
    # inspected, but their text must never be paired using a truncated zip.
    paired = len(spans) == len(labels) and valid_text
    return {
        "spans": [list(span) for span in spans],
        "labels": labels,
        "label_text_valid": valid_text,
        "text_pairing_valid": paired,
        "paired_labels": labels if paired else None,
    }


def native_clock(probe):
    require(len(probe["streams"]) == 1, "expected one selected video stream")
    stream = probe["streams"][0]
    require(stream["codec_name"] == "h264", "unverified packet/frame codec convention")
    require(
        Fraction(stream["start_time"]) == Fraction(probe["format"]["start_time"]) == 0,
        "nonzero source origins need a separately verified transform",
    )
    packets = sorted(probe["packets"], key=lambda packet: packet["pts"])
    require(len(packets) == int(stream["nb_frames"]) > 0, "packet/frame count mismatch")
    points = [packet["pts"] for packet in packets]
    require(all(type(point) is int for point in points), "invalid native PTS")
    require(points[0] == 0 and all(a < b for a, b in pairwise(points)), "invalid PTS order")
    durations = [packet["duration"] for packet in packets]
    require(
        all(type(value) is int and value > 0 for value in durations), "invalid native durations"
    )
    require(
        all(
            start + duration <= following
            for start, duration, following in zip(
                points[:-1], durations[:-1], points[1:], strict=True
            )
        ),
        "overlapping native frame cells",
    )
    base = Fraction(stream["time_base"])
    require(base > 0, "invalid native time base")
    # Presentation order, not H.264 decode/packet order, sets frame ordinals.
    boundaries = [point * base for point in points]
    boundaries.append((points[-1] + durations[-1]) * base)
    return boundaries


def mapped_intervals(spans, boundaries):
    mapped = []
    frames = len(boundaries) - 1
    for start, end in spans:
        require(0 <= start < end <= frames, "annotation exceeds native frame timeline")
        mapped.append((boundaries[start], boundaries[end]))
    return mapped


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        require(0 <= start < end, "invalid time interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def rasterize(intervals, frames, source_end, fps=25):
    """Return exact sample labels and soft frame-cell occupancy on the new clock.

    Covered fractions expose any derivative tail beyond source video. Overlapping
    annotations are unioned for binary targets and remain separate in records.
    """
    require(type(frames) is int and frames > 0 and fps == 25, "invalid canonical timeline")
    require(source_end > 0, "invalid source end")
    merged = union_intervals(intervals)
    require(all(end <= source_end for _, end in merged), "annotation exceeds source time")
    samples, sample_valid, positive, covered = [], [], [], []
    for frame in range(frames):
        start, end = Fraction(frame, fps), Fraction(frame + 1, fps)
        known = max(Fraction(0), min(end, source_end) - start)
        overlap = sum(
            (max(Fraction(0), min(end, right) - max(start, left)) for left, right in merged),
            Fraction(0),
        )
        samples.append(any(left <= start < right for left, right in merged))
        sample_valid.append(start < source_end)
        positive.append(float(overlap * fps))
        covered.append(float(known * fps))
    return {
        "sample_is_fingerspelling": samples,
        "sample_annotation_valid": sample_valid,
        "positive_frame_cell_fraction": positive,
        "covered_frame_cell_fraction": covered,
    }
