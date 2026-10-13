"""The visual LoRA decode-length screen uses training-only paired WER."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_visual_lora_decode_length import PENALTIES, choose, summarize


def _records() -> list[dict]:
    records = []
    signers = [1, 2, 3, 5, 8] * 5 + [1, 2, 3]
    for signer in signers:
        records.append(
            {
                "source": "how2sign_original",
                "signer": signer,
                "decodes": {
                    str(penalty): {
                        "wer": 0.8 if penalty == 1.0 else 0.77,
                        "hypothesis": "a slightly longer output",
                    }
                    for penalty in PENALTIES
                },
            }
        )
    for _ in range(4):
        records.append(
            {
                "source": "fleurs",
                "signer": 3,
                "decodes": {
                    str(penalty): {
                        "wer": 0.9,
                        "hypothesis": "another output",
                    }
                    for penalty in PENALTIES
                },
            }
        )
    return records


def test_selects_lowest_passing_penalty_on_a_tie():
    result = summarize(_records())
    assert choose(result) == 1.1
    assert result["1.1"]["how2sign_signers_improved"] == 5


def test_rejects_penalties_without_enough_gain():
    records = _records()
    for row in records:
        for penalty in PENALTIES[1:]:
            row["decodes"][str(penalty)]["wer"] = row["decodes"]["1.0"]["wer"]
    assert choose(summarize(records)) is None
