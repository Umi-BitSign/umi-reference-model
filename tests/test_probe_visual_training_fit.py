"""Training-fit diagnostics select only examples in the completed stage-one run."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from freeze_visual_lora_plan import SOURCE_LAYOUT
from probe_visual_training_fit import QUOTAS, STAGE_STEPS, choose_rows, seen_training_ids, summarize


def test_frozen_training_fit_selection():
    seen = seen_training_ids(STAGE_STEPS)
    rows = {
        source: [json.loads(line) for line in (ROOT / layout["index"]).read_text().splitlines()]
        for source, layout in SOURCE_LAYOUT.items()
    }
    selected = choose_rows(rows, seen)
    assert len(selected) == 32
    assert Counter((row["source"], row["signer"]) for row in selected) == Counter({
        (source, signer): count
        for source, signers in QUOTAS.items()
        for signer, count in signers.items()
    })
    rows["fleurs"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="source index differs"):
        choose_rows(rows, seen)


def test_training_fit_summary_rejects_partial_result():
    with pytest.raises(ValueError, match="incomplete"):
        summarize([])
    records = [
        {"source": "fleurs", "baseline_wer": 0.8, "candidate_wer": 0.7}
        for _ in range(32)
    ]
    summary = summarize(records)
    assert summary["fleurs"]["candidate_minus_baseline_mean_wer"] == pytest.approx(-0.1)
