import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_decode_length import PENALTIES, choose, summarize


def test_summary_and_guard() -> None:
    records = []
    for i in range(32):
        source = "fleurs" if i < 4 else "how2sign_new"
        signer = 3 if i < 4 else (1, 2, 3, 5, 8)[(i - 4) % 5]
        records.append({
            "source": source,
            "example_id": str(i),
            "signer": signer,
            "baseline_wer": 1.0,
            "decodes": {
                str(penalty): {
                    "wer": (
                        1.0 if penalty == 1.0
                        else 1.01 if source == "fleurs" else 0.9
                    ),
                    "hypothesis": "one word",
                }
                for penalty in PENALTIES
            },
        })
    summary = summarize(records)
    assert summary["1.3"]["how2sign_signers_improved"] == 5
    assert choose(summary) == 0.7
