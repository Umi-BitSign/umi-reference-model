from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_c5_crop_calibration_plan import select


def test_select_caps_signers_and_excludes_holdout() -> None:
    rows = []
    for signer, count in ((1, 30), (2, 14), (3, 30), (4, 30), (5, 30), (8, 30), (9, 11), (11, 30)):
        for index in range(count):
            rows.append(
                {
                    "schema": "umi-studio-how2sign-precut-expansion-stream-cache/1",
                    "split": "training",
                    "temporal_stride": 2,
                    "sentence_name": f"{signer}-{index}",
                    "signer": signer,
                }
            )
    holdout = [{"sentence_name": "1-0"}]
    chosen = select(rows, holdout)
    assert "1-0" not in {row["sentence_name"] for row in chosen}
    assert Counter(row["signer"] for row in chosen) == Counter(
        {1: 24, 2: 14, 3: 24, 4: 24, 5: 24, 8: 24, 9: 11, 11: 24}
    )
