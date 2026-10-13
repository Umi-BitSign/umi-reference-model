"""Freeze all newly available source-disjoint OpenASL candidate clips."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def select_scale_rows(
    candidates: dict, probes: list[dict], pilot: dict
) -> tuple[list[dict], list[dict]]:
    if (
        candidates.get("schema") != "umi-studio-openasl-source-candidates/1"
        or candidates.get("selected_clips") != 512
        or len(candidates.get("selected_sources", [])) != 256
        or pilot.get("schema") != "umi-studio-openasl-pilot-selection/1"
        or pilot.get("source_videos") != 60
        or len(pilot.get("records", [])) != 120
    ):
        raise ValueError("OpenASL scale parent selections differ")
    expected_selection = candidates.get("freezer_sha256")
    if not expected_selection:
        raise ValueError("OpenASL candidate selection lacks provenance")
    by_source = {}
    for probe in probes:
        if (
            probe.get("schema") != "umi-studio-openasl-source-probe/1"
            or probe["youtube_id"] in by_source
        ):
            raise ValueError("OpenASL source probes are invalid or duplicated")
        by_source[probe["youtube_id"]] = probe
    if set(by_source) != set(candidates["selected_sources"]):
        raise ValueError("OpenASL source probes are incomplete")
    used_sources = {row["youtube_id"] for row in pilot["records"]}
    if len(used_sources) != 60:
        raise ValueError("OpenASL pilot source count differs")
    selected_sources = [
        youtube_id
        for youtube_id in candidates["selected_sources"]
        if by_source[youtube_id]["available"] and youtube_id not in used_sources
    ]
    selected_set = set(selected_sources)
    rows = [
        row for row in candidates["records"] if row["youtube_id"] in selected_set
    ]
    if (
        len(selected_sources) != 180
        or len(rows) != 360
        or len({row["example_id"] for row in rows}) != len(rows)
        or any(sum(item["youtube_id"] == source for item in rows) != 2 for source in selected_sources)
    ):
        raise ValueError("OpenASL scale selection no longer has 180 new two-clip sources")
    return rows, [by_source[source] for source in selected_sources]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not all(path.is_file() for path in (args.candidates, args.probes, args.pilot))
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL scale freeze needs complete inputs and a new runs/ output")
    candidates = json.loads(args.candidates.read_text())
    probes = [json.loads(line) for line in args.probes.read_text().splitlines()]
    pilot = json.loads(args.pilot.read_text())
    rows, selected_probes = select_scale_rows(candidates, probes, pilot)
    source_bytes = sum(row["filesize_approx"] for row in selected_probes)
    clip_bytes = sum(
        probe["filesize_approx"] / probe["duration_sec"] * row["duration_sec"]
        for probe in selected_probes
        for row in rows
        if row["youtube_id"] == probe["youtube_id"]
    )
    result = {
        "schema": "umi-studio-openasl-scale-selection/1",
        "objective": "add independent OpenASL source diversity beyond the 60-source pilot",
        "candidate_selection": str(args.candidates.resolve().relative_to(ROOT)),
        "candidate_selection_sha256": sha256(args.candidates),
        "source_probes": str(args.probes.resolve().relative_to(ROOT)),
        "source_probes_sha256": sha256(args.probes),
        "pilot_selection": str(args.pilot.resolve().relative_to(ROOT)),
        "pilot_selection_sha256": sha256(args.pilot),
        "available_candidate_sources": sum(row["available"] for row in probes),
        "excluded_prepared_pilot_sources": 60,
        "selected_new_sources": len(selected_probes),
        "selected_new_clips": len(rows),
        "selected_duration_hours": sum(row["duration_sec"] for row in rows) / 3600,
        "full_source_filesize_approx_bytes": source_bytes,
        "clip_only_filesize_estimate_bytes": round(clip_bytes),
        "clip_only_estimate_method": (
            "sum each probed source's approximate bytes per second times selected clip duration; "
            "container overhead, keyframe seek margins and format choice are excluded"
        ),
        "acquisition_policy": {
            "required_free_disk_reserve_gib": 100,
            "prefer_bounded_clip_ranges": True,
            "preserve_aspect_ratio": True,
            "redistribute_source_or_modified_video": False,
            "miner_health_policy": "user_authorized_maintenance",
        },
        "records": rows,
        "source_probes_selected": selected_probes,
        "freezer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(args.output)
    print(
        json.dumps(
            {
                "sources": result["selected_new_sources"],
                "clips": result["selected_new_clips"],
                "duration_hours": result["selected_duration_hours"],
                "full_source_gib": source_bytes / 2**30,
                "clip_only_estimate_gib": clip_bytes / 2**30,
                "output_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
