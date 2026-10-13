from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from evaluate_fresh_lowres_tuning import load_tuning


def test_load_tuning_requires_frozen_scope(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    rows = []
    for source, groups, count in (("how2sign", 16, 16), ("fleurs", 3, 7)):
        for index in range(count):
            example_id = f"{source}-{index}"
            (videos / f"{example_id}.mp4").write_bytes(b"video")
            rows.append(
                {
                    "schema": "umi-studio-fresh-lowres-tuning-example/1",
                    "split": "tuning",
                    "source": source,
                    "source_group": f"{source}:group-{index % groups}",
                    "example_id": example_id,
                    "reference": "reference",
                    "video": f"videos/{example_id}.mp4",
                    "video_sha256": "0" * 64,
                }
            )
    inventory = tmp_path / "tuning.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(load_tuning(inventory, tmp_path)) == 23
    rows.pop()
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="scope differs"):
        load_tuning(inventory, tmp_path)
