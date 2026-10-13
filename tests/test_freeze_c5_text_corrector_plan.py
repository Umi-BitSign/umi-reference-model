import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_c5_text_corrector_plan import split_rows


def test_split_is_source_stratified_and_stable():
    rows = [
        {"source": source, "example_id": f"{source}-{index}"}
        for source in ("a", "b")
        for index in range(10)
    ]
    train, holdout = split_rows(rows)
    assert len(train) == 16
    assert len(holdout) == 4
    assert {row["source"] for row in holdout} == {"a", "b"}
    assert split_rows(list(reversed(rows))) == (train, holdout)
    assert not {
        (row["source"], row["example_id"]) for row in train
    } & {(row["source"], row["example_id"]) for row in holdout}
