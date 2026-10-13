import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from benchmark_fleurs_dev import sha256
from prepare_openasl_scale_landmarks import checked_source_clips


def test_checked_source_clips_binds_each_example(tmp_path: Path):
    rows = [
        {"example_id": "a", "youtube_id": "source00001"},
        {"example_id": "b", "youtube_id": "source00001"},
    ]
    receipts = []
    for index, row in enumerate(rows):
        path = tmp_path / f"clip-{index}.mp4"
        path.write_bytes(f"video-{index}".encode())
        receipts.append(
            {
                "schema": "umi-studio-openasl-scale-clip-receipt/1",
                "selection_sha256": "selection",
                "example_id": row["example_id"],
                "youtube_id": row["youtube_id"],
                "filename": path.name,
                "video_sha256": sha256(path),
            }
        )
    sources = checked_source_clips(
        {"records": rows}, "selection", tmp_path, receipts
    )
    assert set(sources) == {"a", "b"}
    assert sources["a"]["path"].name == "clip-0.mp4"
