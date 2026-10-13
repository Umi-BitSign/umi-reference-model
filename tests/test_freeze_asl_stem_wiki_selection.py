"""ASL STEM Wiki selection balances participants under diversity caps."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_asl_stem_wiki_selection import select_rows


def test_selection_meets_balanced_quotas_and_diversity_caps():
    rows = []
    for participant in ("P0", "P1", "P2"):
        for index in range(6):
            rows.append(
                {
                    "participant": participant,
                    "article": f"article-{index % 3}",
                    "section index": "0",
                    "sentence index": str(index),
                    "video filename": f"{participant}-{index}.mp4",
                    "normalized_reference": f"reference {participant} {index}",
                }
            )
    selected, quotas = select_rows(
        rows,
        target=12,
        base_quota=4,
        max_per_article=4,
        max_per_reference=1,
        seed=7,
    )
    assert len(selected) == 12
    assert Counter(row["participant"] for row in selected) == Counter(quotas)
    assert max(Counter(row["article"] for row in selected).values()) <= 4
    assert len({row["normalized_reference"] for row in selected}) == 12


def test_larger_selection_preserves_smaller_selection_prefix_set():
    rows = []
    for participant in ("P0", "P1", "P2"):
        for index in range(12):
            rows.append(
                {
                    "participant": participant,
                    "article": f"article-{index % 4}",
                    "section index": "0",
                    "sentence index": str(index),
                    "video filename": f"{participant}-{index}.mp4",
                    "normalized_reference": f"reference {participant} {index}",
                }
            )
    smaller, _ = select_rows(
        rows,
        target=12,
        base_quota=4,
        max_per_article=20,
        max_per_reference=1,
        seed=7,
    )
    larger, _ = select_rows(
        rows,
        target=24,
        base_quota=4,
        max_per_article=20,
        max_per_reference=1,
        seed=7,
    )
    assert {row["video filename"] for row in smaller}.issubset(
        {row["video filename"] for row in larger}
    )
