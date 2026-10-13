from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_c5_how2sign_expansion_scale_plan import (
    EXPANSION_TARGETS,
    FLEURS_EXAMPLES,
    RECIPE,
    REPLAY_PER_SIGNER,
    select_scale_schedule,
)


def _row(source: str, identifier: str, signer: int) -> dict:
    return {"source": source, "id": identifier, "signer": signer}


def test_scaled_schedule_exhausts_scarce_signers_deterministically() -> None:
    expansion = [
        _row("how2sign_precut_expansion", f"e-{signer}-{index}", signer)
        for signer, target in EXPANSION_TARGETS.items()
        for index in range(target + (0 if signer in {2, 3, 4, 9, 11} else 3))
    ]
    replay = [
        _row("how2sign_original", f"r-{signer}-{index}", signer)
        for signer in EXPANSION_TARGETS
        for index in range(REPLAY_PER_SIGNER + 2)
    ]
    replay.extend(
        _row("fleurs", f"f-{index}", 4)
        for index in range(FLEURS_EXAMPLES + 2)
    )
    first = select_scale_schedule(replay, expansion)
    second = select_scale_schedule(list(reversed(replay)), list(reversed(expansion)))
    assert [(row["source"], row["id"]) for row in first] == [
        (row["source"], row["id"]) for row in second
    ]
    assert len(first) == RECIPE["train_examples"]
    selected = [
        row for row in first if row["source"] == "how2sign_precut_expansion"
    ]
    assert Counter(row["signer"] for row in selected) == Counter(EXPANSION_TARGETS)
