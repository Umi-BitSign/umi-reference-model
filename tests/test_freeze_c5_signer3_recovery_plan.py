"""The focused recovery adds signer-3 diversity without dropping replay."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_c5_signer3_recovery_plan import select_schedule


def test_signer3_schedule_focuses_one_signer():
    replay = []
    for signer in (1, 2, 3, 4, 5, 8, 9, 11):
        replay.extend(
            {
                "id": f"h-{signer}-{index}",
                "source": "how2sign_raw",
                "signer": signer,
            }
            for index in range(200)
        )
    replay.extend(
        {"id": f"f-{index}", "source": "fleurs", "signer": 3}
        for index in range(313)
    )
    schedule = select_schedule(replay)
    counts = Counter(
        row["signer"] for row in schedule if row["source"].startswith("how2sign")
    )
    assert len(schedule) == 768
    assert counts[3] == 192
    assert all(counts[signer] == 64 for signer in (1, 2, 4, 5, 8, 9, 11))
    assert sum(row["source"] == "fleurs" for row in schedule) == 128
