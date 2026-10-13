import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_openasl_scale_selection import select_scale_rows


def test_selects_all_available_sources_outside_pilot():
    sources = [f"source-{index}" for index in range(256)]
    candidates = {
        "schema": "umi-studio-openasl-source-candidates/1",
        "selected_clips": 512,
        "selected_sources": sources,
        "freezer_sha256": "frozen",
        "records": [
            {"youtube_id": source, "example_id": f"{source}-{clip}"}
            for source in sources
            for clip in range(2)
        ],
    }
    unavailable = set(sources[240:])
    probes = [
        {
            "schema": "umi-studio-openasl-source-probe/1",
            "youtube_id": source,
            "available": source not in unavailable,
        }
        for source in sources
    ]
    pilot_sources = set(sources[:60])
    pilot = {
        "schema": "umi-studio-openasl-pilot-selection/1",
        "source_videos": 60,
        "records": [
            {"youtube_id": source, "example_id": f"pilot-{source}-{clip}"}
            for source in pilot_sources
            for clip in range(2)
        ],
    }
    rows, selected = select_scale_rows(candidates, probes, pilot)
    assert len(rows) == 360
    assert len(selected) == 180
    assert not pilot_sources & {row["youtube_id"] for row in rows}
    assert not unavailable & {row["youtube_id"] for row in rows}
