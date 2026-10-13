from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from probe_fresh_lowres_training_fit import _select_seen


def test_select_seen_spans_the_training_run() -> None:
    rows = [
        {"source": "test", "example_id": str(index), "reference": str(index)}
        for index in range(10)
    ]
    examples = [
        {"source": row["source"], "example_id": row["example_id"]} for row in rows
    ]
    selected = _select_seen(rows, examples, 4)
    assert [row["example_id"] for row in selected] == ["0", "3", "6", "9"]
