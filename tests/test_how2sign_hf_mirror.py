from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import how2sign_hf_mirror as mirror
import fetch_c5_broad_how2sign_mirror as adapter


class Response(io.BytesIO):
    def __init__(self, payload=b"abcd", status=206, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Range": f"bytes 4-7/{mirror.ARCHIVE_SIZE}"}


def remote():
    result = mirror.How2SignMirrorZip.__new__(mirror.How2SignMirrorZip)
    result.url = mirror.URL
    result.archive_size = mirror.ARCHIVE_SIZE
    result.bytes_downloaded = result.requests = 0
    return result


def test_exact_bounded_range_and_cache_identity(monkeypatch):
    requests = []

    def fetch(request, timeout):
        requests.append(request)
        assert timeout == 60
        return Response()

    monkeypatch.setattr(mirror.urllib.request, "urlopen", fetch)
    r = remote()
    assert r._read_range(4, 7) == b"abcd"
    assert r.requests == 1 and r.bytes_downloaded == 4
    assert requests[0].get_header("Range") == "bytes=4-7"
    assert requests[0].get_header("Accept-encoding") == "identity"
    assert mirror.REVISION in requests[0].full_url
    assert "range_start=4&range_end=7" in requests[0].full_url


@pytest.mark.parametrize("status,headers,payload", [
    (200, {"Content-Type": "text/html"}, b"quota"),
    (206, {"Content-Range": f"bytes 0-3/{mirror.ARCHIVE_SIZE}"}, b"abcd"),
    (206, {"Content-Range": "bytes 4-7/100"}, b"abcd"),
    (206, {"Content-Range": f"bytes 4-7/{mirror.ARCHIVE_SIZE}", "Content-Encoding": "gzip"}, b"abcd"),
    (206, None, b"abc"),
    (206, None, b"abcde"),
])
def test_wrong_response_fails_without_retry(monkeypatch, status, headers, payload):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        return Response(payload, status, headers)

    monkeypatch.setattr(mirror.urllib.request, "urlopen", fetch)
    r = remote()
    with pytest.raises(ValueError):
        r._read_range(4, 7)
    assert len(calls) == 1 and r.requests == r.bytes_downloaded == 0


def test_transient_retries_are_bounded(monkeypatch):
    calls, delays = [], []

    def fetch(request, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr(mirror.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(mirror.time, "sleep", delays.append)
    with pytest.raises(urllib.error.HTTPError):
        remote()._read_range(4, 7)
    assert len(calls) == 4 and delays == [1, 2, 4]


@pytest.mark.parametrize("start,end", [(-1, 1), (3, 2), (0, mirror.ARCHIVE_SIZE), (0, mirror.MAX_RANGE_BYTES)])
def test_range_limits_reject_before_network(start, end):
    with pytest.raises(ValueError):
        remote()._read_range(start, end)


def test_other_archive_rejected_before_network():
    with pytest.raises(ValueError):
        mirror.How2SignMirrorZip(file_id="id", archive_name="other.zip", archive_size=10)


def test_adapter_extends_frozen_sources_and_transport(monkeypatch):
    original = {"source_sha256": {"tools/original.py": "frozen"}, "archive": {"identity": "fixed"}}
    monkeypatch.setattr(adapter.original, "specification", lambda *args: original)
    result = adapter.specification(Path("reservation"), Path("data"))
    assert result["archive"] == {"identity": "fixed"}
    assert result["transport"] == mirror.provenance()
    assert result["transport"]["whole_archive_sha256_verified"] is False
    for name in ("fetch_c5_broad_how2sign_mirror.py", "how2sign_hf_mirror.py"):
        assert result["source_sha256"]["tools/" + name] == adapter.original.sha256(adapter.original.ROOT / "tools" / name)
    assert adapter.run.__globals__["checked"] is adapter.checked
    assert adapter.run.__globals__["How2SignRemoteZip"] is mirror.How2SignMirrorZip
    assert adapter.freeze.__globals__["specification"] is adapter.specification
