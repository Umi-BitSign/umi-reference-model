from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_c5_how2sign_expansion_holdout import PER_SIGNER_CAP, select_holdout


def test_holdout_excludes_seen_and_caps_each_signer() -> None:
    rows = [
        {"sentence_name": f"{signer}-{index}", "signer": signer}
        for signer in (1, 2, 3, 4, 5, 8, 9, 11)
        for index in range(25)
    ]
    seen = {f"{signer}-{index}" for signer in (1, 2, 3, 4, 5, 8, 9, 11) for index in range(3)}
    selected = select_holdout(rows, seen)
    assert not ({row["sentence_name"] for row in selected} & seen)
    assert Counter(row["signer"] for row in selected) == {
        signer: PER_SIGNER_CAP for signer in (1, 2, 3, 4, 5, 8, 9, 11)
    }
    assert [row["sentence_name"] for row in selected] == [
        row["sentence_name"] for row in select_holdout(list(reversed(rows)), seen)
    ]
