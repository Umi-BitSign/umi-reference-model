"""Pin sizes and SHA-256 digests for the 2M-Flores-ASL dev video files."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("schema") != "umi-studio-flores2m-dev-inventory/1":
        raise ValueError("unexpected dev inventory schema")
    revision = inventory["repository_revision"]
    root = f"https://huggingface.co/api/datasets/facebook/2M-Flores-ASL/tree/{revision}/"
    url = root + "data/dev?recursive=false&expand=true&limit=100"
    expected_paths = {"data/dev/" + row["video_url"].rsplit("/", 1)[1] for row in inventory["rows"]}
    files = {}
    while url:
        if not url.startswith(root):
            raise ValueError("pagination left the pinned repository")
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected file inventory response: {response.status}")
            page = json.load(response)
            link = response.headers.get("Link", "")
        for entry in page:
            if entry["type"] != "file" or not entry["path"].startswith("data/dev/"):
                raise ValueError("tree includes something outside the dev video directory")
            if entry["path"] not in expected_paths:
                continue
            if entry["path"] in files:
                raise ValueError("duplicate dev video file")
            if entry["size"] <= 0 or not re.fullmatch(
                r"[0-9a-f]{64}", entry.get("lfs", {}).get("oid", "")
            ):
                raise ValueError(f"dev video file lacks size or SHA-256: {entry['path']}")
            files[entry["path"]] = {
                "path": entry["path"],
                "size_bytes": entry["size"],
                "sha256": entry["lfs"]["oid"],
            }
        match = re.search(r'<(https://huggingface\.co/[^>]+)>; rel="next"', link)
        url = match.group(1) if match else None
    if set(files) != expected_paths:
        raise ValueError(
            f"dev file inventory mismatch: {len(files)} files, {len(expected_paths)} expected"
        )
    output = {
        "schema": "umi-studio-flores2m-dev-files/1",
        "repository_revision": revision,
        "files": [files[path] for path in sorted(files)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(f"pinned {len(files)} dev video sizes and SHA-256 digests")


if __name__ == "__main__":
    main()
