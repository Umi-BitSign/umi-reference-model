from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_fsboard_train_landmark_manifest import manifest_rows


def test_manifest_requires_all_train_shards_once() -> None:
    rows = [
        f"daun_v3/landmarks/daun_v3-train.arrow-{number:05d}-of-00100,600000000,2025-07-26"
        for number in range(100)
    ]
    listing = "name,size,creationDate\n" + "\n".join(rows) + "\n"
    parsed = manifest_rows(listing)
    assert len(parsed) == 100
    assert parsed[0]["shard_number"] == 0
    assert parsed[-1]["shard_number"] == 99
    with pytest.raises(ValueError, match="all 100"):
        manifest_rows("\n".join(rows[:-1]))
    with pytest.raises(ValueError, match="duplicated"):
        manifest_rows(listing + rows[0] + "\n")
