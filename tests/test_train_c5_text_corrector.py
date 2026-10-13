import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from train_c5_text_corrector import metric_summary


def test_metric_summary_groups_sources():
    rows = [
        {"source": "a", "example_id": "1", "reference": "one two", "hypothesis": "one", "single_reference_wer": 0.5},
        {"source": "b", "example_id": "2", "reference": "three", "hypothesis": "four", "single_reference_wer": 1.0},
    ]
    summary = metric_summary(rows, ["one two", "three"])
    assert summary["corrected_mean_wer"] == 0
    assert summary["mean_wer_change"] == -0.75
    assert set(summary["sources"]) == {"a", "b"}
    assert summary["improved"] == 2
