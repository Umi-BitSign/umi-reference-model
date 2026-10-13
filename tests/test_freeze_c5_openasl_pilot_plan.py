from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_c5_openasl_pilot_plan import select_schedule


def test_select_schedule_balances_replay_and_keeps_all_openasl() -> None:
    replay = []
    for signer in (1, 2, 3, 4, 5, 8, 9, 11):
        replay.extend(
            {
                "id": f"h-{signer}-{index}",
                "source": "how2sign_original",
                "signer": signer,
            }
            for index in range(24)
        )
    replay.extend(
        {"id": f"f-{index}", "source": "fleurs", "signer": 3}
        for index in range(72)
    )
    stem = []
    for source in ("asl_stem_original", "asl_stem_supplemental"):
        stem.extend(
            {
                "id": f"{source}-{index}",
                "source": source,
                "signer": index % 8,
                "participant": index % 8,
            }
            for index in range(64)
        )
    openasl = [
        {
            "id": f"o-{index}",
            "source": "openasl",
            "signer": f"openasl:{index // 2}",
            "source_group": f"openasl:{index // 2}",
        }
        for index in range(120)
    ]
    schedule = select_schedule(replay, stem, openasl)
    counts = Counter(row["source"] for row in schedule)
    assert len(schedule) == 512
    assert counts == {
        "how2sign_original": 192,
        "fleurs": 72,
        "asl_stem_original": 64,
        "asl_stem_supplemental": 64,
        "openasl": 120,
    }
    assert {row["id"] for row in schedule if row["source"] == "openasl"} == {
        row["id"] for row in openasl
    }
