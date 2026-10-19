"""Pinned public mirror transport for the unchanged validation ZIP members."""

from __future__ import annotations

import time
import urllib.error
import urllib.request

from how2sign_remote_zip import How2SignRemoteZip, MAX_RANGE_BYTES

REVISION = "1231830fc1e8d77555a245ca22353b144e168111"
ARCHIVE_NAME = "val_raw_videos.zip"
ARCHIVE_SIZE = 17_528_664_599
LFS_SHA256 = "c6c3caf16f8ae610bcb5f36deb798f65221af07ed429953055976af80799b939"
URL = (
    "https://huggingface.co/datasets/aipieces/How2Sign/resolve/"
    + REVISION
    + "/original/green-screen-rgb-videos-frontal-view/"
    + ARCHIVE_NAME
)


def provenance():
    return {
        "repository": "aipieces/How2Sign",
        "revision": REVISION,
        "path": "original/green-screen-rgb-videos-frontal-view/" + ARCHIVE_NAME,
        "size": ARCHIVE_SIZE,
        "advertised_LFS_sha256": LFS_SHA256,
        "whole_archive_sha256_verified": False,
        "verification": "official directory fingerprint and individual member size/CRC",
    }


class How2SignMirrorZip(How2SignRemoteZip):
    def __init__(self, *, file_id, archive_name, archive_size):
        if archive_name != ARCHIVE_NAME or archive_size != ARCHIVE_SIZE:
            raise ValueError("mirror is pinned to the validation raw archive")
        super().__init__(file_id=file_id, archive_name=archive_name, archive_size=archive_size)

    def _confirmation_url(self):
        return URL

    def _read_range(self, start, end):
        if start < 0 or end < start or end >= self.archive_size:
            raise ValueError("invalid archive range")
        size = end - start + 1
        if size > MAX_RANGE_BYTES:
            raise ValueError("archive range exceeds per-request safety limit")
        # Include both endpoints in the public cache key. Never reuse a signed
        # redirect URL for a different range or send credentials.
        url = self.url + f"?download=true&range_start={start}&range_end={end}"
        for attempt in range(4):
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{self.archive_size}"
                        or response.headers.get("Content-Encoding", "identity") != "identity"
                    ):
                        raise ValueError("mirror did not return the exact unencoded archive range")
                    result = response.read(size + 1)
                    if len(result) != size:
                        raise ValueError("mirror archive range has an unexpected length")
                self.bytes_downloaded += size
                self.requests += 1
                return result
            except urllib.error.HTTPError as error:
                if attempt == 3 or error.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("mirror archive range unavailable")
