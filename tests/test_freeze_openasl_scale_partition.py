import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_openasl_scale_partition import partition_rows


def test_partition_is_source_disjoint_and_stable():
    rows = [
        {
            "youtube_id": f"source-{source:03d}",
            "example_id": f"source-{source:03d}-clip-{clip}",
        }
        for source in range(180)
        for clip in range(2)
    ]
    training, confirmation, reserve = partition_rows(rows)
    assert len(training) == 240
    assert len(confirmation) == 60
    assert len(reserve) == 60
    assert {row["youtube_id"] for row in training}.isdisjoint(
        {row["youtube_id"] for row in confirmation}
    )
    assert {row["youtube_id"] for row in confirmation} == {
        row["youtube_id"] for row in reserve
    }
    assert partition_rows(list(reversed(rows))) == (training, confirmation, reserve)
