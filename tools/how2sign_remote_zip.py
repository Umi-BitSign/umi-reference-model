"""Read selected members of the official How2Sign archive with HTTP byte ranges.

The archive remains on the publisher's Google Drive. Every request must return
the exact requested range so a failed range request cannot download 31 GB.
"""

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ARCHIVE_FILE_ID = "1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW"
ARCHIVE_NAME = "train_rgb_front_clips.zip"
ARCHIVE_SIZE = 33_033_658_988
MAX_RANGE_BYTES = 64 * 1024 * 1024


class How2SignRemoteZip(io.BufferedIOBase):
    def __init__(
        self,
        *,
        file_id: str = ARCHIVE_FILE_ID,
        archive_name: str = ARCHIVE_NAME,
        archive_size: int = ARCHIVE_SIZE,
    ) -> None:
        if not file_id or not archive_name or archive_size < 1:
            raise ValueError("remote archive identity is incomplete")
        self.file_id = file_id
        self.archive_name = archive_name
        self.archive_size = archive_size
        self.position = 0
        self.bytes_downloaded = 0
        self.requests = 0
        self.url = self._confirmation_url()
        self._read_range(0, 0)

    def _confirmation_url(self) -> str:
        warning_url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
            {"export": "download", "id": self.file_id}
        )
        with urllib.request.urlopen(warning_url, timeout=30) as response:
            page = response.read(100_000).decode("utf-8")
        if self.archive_name not in page:
            raise RuntimeError("the publisher download page names another archive")
        match = re.search(r'name="uuid" value="([^"]+)"', page)
        if match is None:
            raise RuntimeError("the publisher download confirmation is unavailable")
        return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
            {
                "id": self.file_id,
                "export": "download",
                "confirm": "t",
                "uuid": match.group(1),
            }
        )

    def _read_range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.archive_size:
            raise ValueError("invalid archive range")
        size = end - start + 1
        if size > MAX_RANGE_BYTES:
            raise ValueError("archive range exceeds per-request safety limit")
        for attempt in range(4):
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    content_range = response.headers.get("Content-Range")
                    expected = f"bytes {start}-{end}/{self.archive_size}"
                    if response.status != 206 or content_range != expected:
                        if attempt < 3:
                            self.url = self._confirmation_url()
                            time.sleep(2**attempt)
                            continue
                        raise RuntimeError(f"unexpected archive response: {content_range}")
                    disposition = response.headers.get("Content-Disposition", "")
                    if self.archive_name not in disposition:
                        raise RuntimeError("archive filename differs from official source")
                    result = response.read(size + 1)
                    if len(result) != size:
                        if attempt < 3:
                            time.sleep(2**attempt)
                            continue
                        raise RuntimeError("archive range has an unexpected length")
                    self.bytes_downloaded += size
                    self.requests += 1
                    return result
            except urllib.error.HTTPError as error:
                if attempt < 3 and error.code in (403, 404, 429):
                    self.url = self._confirmation_url()
                    continue
                if attempt < 3 and error.code in (500, 502, 503, 504):
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError("archive range could not be fetched")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.archive_size + offset
        else:
            raise ValueError("invalid seek mode")
        if position < 0 or position > self.archive_size:
            raise ValueError("seek outside archive")
        self.position = position
        return position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.archive_size - self.position
        if size > MAX_RANGE_BYTES:
            raise ValueError("archive read exceeds per-request safety limit")
        if self.position == self.archive_size or size == 0:
            return b""
        end = min(self.position + size, self.archive_size) - 1
        result = self._read_range(self.position, end)
        self.position = end + 1
        return result
