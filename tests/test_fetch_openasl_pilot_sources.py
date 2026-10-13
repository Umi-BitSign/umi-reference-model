from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fetch_openasl_pilot_sources import DOWNLOAD_STRATEGIES, download_source


def test_download_source_uses_hls_after_https_failure(tmp_path: Path) -> None:
    results = [
        SimpleNamespace(returncode=1, stderr="HTTP Error 403"),
        SimpleNamespace(returncode=0, stderr=""),
    ]
    with patch("fetch_openasl_pilot_sources.subprocess.run", side_effect=results) as run:
        strategy, attempts = download_source(tmp_path, "abcdefghijk")
    assert strategy == "hls"
    assert [row["strategy"] for row in attempts] == ["https", "hls"]
    assert attempts[0]["error_tail"] == "HTTP Error 403"
    assert run.call_count == 2
    assert DOWNLOAD_STRATEGIES[1][1] in run.call_args.args[0]


def test_download_source_returns_all_failures(tmp_path: Path) -> None:
    result = SimpleNamespace(returncode=1, stderr="unavailable")
    with patch(
        "fetch_openasl_pilot_sources.subprocess.run",
        side_effect=[result] * len(DOWNLOAD_STRATEGIES),
    ):
        strategy, attempts = download_source(tmp_path, "abcdefghijk")
    assert strategy is None
    assert len(attempts) == len(DOWNLOAD_STRATEGIES)
