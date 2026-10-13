"""The scaled STEM plan consumes both tranches with balanced replay."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_c5_stem_scale_plan import select_schedule


def test_scale_schedule_counts_all_stem_and_balances_how2sign():
    replay = []
    for signer in (1, 2, 3, 4, 5, 8, 9, 11):
        replay.extend(
            {
                "id": f"h-{signer}-{index}",
                "source": "how2sign_original" if index % 2 else "how2sign_raw",
                "signer": signer,
            }
            for index in range(100)
        )
    replay.extend(
        {"id": f"f-{index}", "source": "fleurs", "signer": 3}
        for index in range(313)
    )
    stem = [
        {
            "id": f"s-{tranche}-{index}",
            "source": f"asl_stem_{tranche}",
            "signer": f"P{index % 36}",
            "participant": f"P{index % 36}",
        }
        for tranche in ("original", "supplemental")
        for index in range(1_024)
    ]
    schedule = select_schedule(replay, stem)
    assert len(schedule) == 3_112
    assert sum(row["source"].startswith("asl_stem") for row in schedule) == 2_048
    assert Counter(
        row["signer"] for row in schedule if row["source"].startswith("how2sign")
    ) == {signer: 94 for signer in (1, 2, 3, 4, 5, 8, 9, 11)}
    assert sum(row["source"] == "fleurs" for row in schedule) == 312
