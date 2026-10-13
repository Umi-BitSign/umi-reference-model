"""Freeze the available source-disjoint OpenASL pilot clips."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.candidates.is_file()
        or not args.probes.is_file()
        or args.output.exists()
        or not args.output.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("OpenASL pilot needs frozen candidates, probes and a new output")
    candidates = json.loads(args.candidates.read_text())
    probes = [json.loads(line) for line in args.probes.read_text().splitlines()]
    if (
        candidates.get("schema") != "umi-studio-openasl-source-candidates/1"
        or len(probes) != 64
        or {row.get("schema") for row in probes}
        != {"umi-studio-openasl-source-probe/1"}
        or {row.get("selection_sha256") for row in probes}
        != {sha256(args.candidates)}
    ):
        raise ValueError("OpenASL pilot inputs differ")
    available = {row["youtube_id"] for row in probes if row["available"]}
    records = [
        row for row in candidates["records"] if row["youtube_id"] in available
    ]
    counts = Counter(row["youtube_id"] for row in records)
    if (
        len(available) != 60
        or len(records) != 120
        or set(counts.values()) != {2}
        or len({row["normalized_reference"] for row in records}) != len(records)
    ):
        raise ValueError("OpenASL pilot availability does not produce 60 complete sources")
    probe_by_id = {row["youtube_id"]: row for row in probes}
    result = {
        "schema": "umi-studio-openasl-pilot-selection/1",
        "candidate_selection_sha256": sha256(args.candidates),
        "source_probes_sha256": sha256(args.probes),
        "source_videos": len(available),
        "clips": len(records),
        "duration_hours": sum(row["duration_sec"] for row in records) / 3600,
        "source_duration_hours": sum(
            probe_by_id[yid]["duration_sec"] for yid in available
        )
        / 3600,
        "probed_source_filesize_approx_bytes": sum(
            probe_by_id[yid]["filesize_approx"] for yid in available
        ),
        "max_clips_per_source": 2,
        "source_video_disjoint_from_official_valid_test": True,
        "records": records,
        "freezer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(args.output)
    print(
        json.dumps(
            {
                "sources": result["source_videos"],
                "clips": result["clips"],
                "duration_hours": result["duration_hours"],
                "source_bytes_approx": result[
                    "probed_source_filesize_approx_bytes"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
