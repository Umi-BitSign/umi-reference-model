from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fetch_fsboard_train_shard import frozen_entry, publisher_size


def test_publisher_size_reads_exact_train_shard_without_test_rows() -> None:
    listing = (
        "Next Page Token = opaque\n"
        "name,size,creationDate\n"
        "daun_v3/landmarks/daun_v3-test.arrow-00001-of-00020,657729802,2025-07-26\n"
        "daun_v3/landmarks/daun_v3-train.arrow-00001-of-00100,657729802,2025-07-26\n"
    )
    assert publisher_size(
        listing, "daun_v3/landmarks/daun_v3-train.arrow-00001-of-00100"
    ) == 657729802
    with pytest.raises(ValueError, match="plausible"):
        publisher_size(listing, "daun_v3/landmarks/daun_v3-train.arrow-00002-of-00100")
    with pytest.raises(ValueError, match="plausible"):
        publisher_size(listing + listing.splitlines()[-1] + "\n", "daun_v3/landmarks/daun_v3-train.arrow-00001-of-00100")


def test_frozen_entry_rejects_manifest_drift() -> None:
    manifest = json.loads(
        (Path(__file__).resolve().parents[1]
         / "runs/fsboard-train-landmark-manifest-v1/manifest.json").read_text()
    )
    assert frozen_entry(manifest, 1)["size_bytes"] == 657729802
    manifest["shards"][1]["size_bytes"] += 1
    with pytest.raises(ValueError, match="digest"):
        frozen_entry(manifest, 1)
