"""The ASL STEM continuation schedule preserves replay balance."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_c5_stem_continuation_plan import select_schedule


def test_select_schedule_has_exact_sources_signers_and_participants():
    replay = []
    for signer in (1, 2, 3, 4, 5, 8, 9, 11):
        replay.extend(
            {
                "id": f"h-{signer}-{index}",
                "source": "how2sign_original",
                "signer": signer,
            }
            for index in range(30)
        )
    replay.extend(
        {"id": f"f-{index}", "source": "fleurs", "signer": 3}
        for index in range(80)
    )
    stem = [
        {
            "id": f"s-{participant}-{index}",
            "source": "asl_stem",
            "signer": f"P{participant}",
            "participant": f"P{participant}",
            "article": f"article-{index}",
        }
        for participant in range(36)
        for index in range(8)
    ]
    schedule = select_schedule(replay, stem)
    assert len(schedule) == 512
    assert Counter(row["source"] for row in schedule) == {
        "asl_stem": 256,
        "how2sign_original": 192,
        "fleurs": 64,
    }
    assert Counter(
        row["signer"] for row in schedule if row["source"].startswith("how2sign")
    ) == {signer: 24 for signer in (1, 2, 3, 4, 5, 8, 9, 11)}
    assert len({row["participant"] for row in schedule if row["source"] == "asl_stem"}) == 36
