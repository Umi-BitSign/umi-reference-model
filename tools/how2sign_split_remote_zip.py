"""Read selected members of the publisher's split How2Sign training ZIP.

Only exact HTTP ranges are fetched. The archive manifest pins all part sizes
and the ZIP64 central directory, so a changed publisher file is rejected.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import struct
import zipfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from how2sign_remote_zip import How2SignRemoteZip

RANGE_CHUNK = 8 * 1024 * 1024
MAX_MEMBER_COMPRESSED = 512 * 1024 * 1024


@dataclass(frozen=True)
class SplitZipEntry:
    name: str
    disk: int
    header_offset: int
    compressed_size: int
    file_size: int
    compression: int
    flags: int
    crc32: int


def _take_extra(value: bytes, cursor: int, format_code: str) -> tuple[int, int]:
    size = struct.calcsize(format_code)
    if cursor + size > len(value):
        raise ValueError("ZIP64 extra field is truncated")
    return struct.unpack_from(format_code, value, cursor)[0], cursor + size


def parse_central_directory(data: bytes, part_count: int) -> dict[str, SplitZipEntry]:
    entries = {}
    position = 0
    while position < len(data):
        if len(data) - position < zipfile.sizeCentralDir:
            raise ValueError("split ZIP central directory is truncated")
        fields = struct.unpack_from(zipfile.structCentralDir, data, position)
        if fields[zipfile._CD_SIGNATURE] != zipfile.stringCentralDir:
            raise ValueError("split ZIP central directory has an invalid signature")
        name_length = fields[zipfile._CD_FILENAME_LENGTH]
        extra_length = fields[zipfile._CD_EXTRA_FIELD_LENGTH]
        comment_length = fields[zipfile._CD_COMMENT_LENGTH]
        end = position + zipfile.sizeCentralDir + name_length + extra_length + comment_length
        if end > len(data):
            raise ValueError("split ZIP central directory entry is truncated")
        name_bytes = data[position + zipfile.sizeCentralDir :][:name_length]
        extra = data[position + zipfile.sizeCentralDir + name_length :][:extra_length]
        encoding = "utf-8" if fields[zipfile._CD_FLAG_BITS] & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        file_size = fields[zipfile._CD_UNCOMPRESSED_SIZE]
        compressed_size = fields[zipfile._CD_COMPRESSED_SIZE]
        header_offset = fields[zipfile._CD_LOCAL_HEADER_OFFSET]
        disk = fields[zipfile._CD_DISK_NUMBER_START]
        extra_position = 0
        while extra_position < len(extra):
            if extra_position + 4 > len(extra):
                raise ValueError("split ZIP extra field is truncated")
            tag, length = struct.unpack_from("<HH", extra, extra_position)
            value_start = extra_position + 4
            value_end = value_start + length
            if value_end > len(extra):
                raise ValueError("split ZIP extra field has an invalid length")
            if tag == 1:
                value = extra[value_start:value_end]
                cursor = 0
                if file_size == 0xFFFFFFFF:
                    file_size, cursor = _take_extra(value, cursor, "<Q")
                if compressed_size == 0xFFFFFFFF:
                    compressed_size, cursor = _take_extra(value, cursor, "<Q")
                if header_offset == 0xFFFFFFFF:
                    header_offset, cursor = _take_extra(value, cursor, "<Q")
                if disk == 0xFFFF:
                    disk, cursor = _take_extra(value, cursor, "<I")
            extra_position = value_end
        if (
            name in entries
            or not name.startswith("raw_videos/")
            or name.startswith("raw_videos/../")
            or "\\" in name
            or disk >= part_count
            or file_size == 0xFFFFFFFF
            or compressed_size == 0xFFFFFFFF
            or header_offset == 0xFFFFFFFF
            or disk == 0xFFFF
            or fields[zipfile._CD_COMPRESS_TYPE] not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        ):
            raise ValueError(f"invalid or duplicate split ZIP entry: {name}")
        entries[name] = SplitZipEntry(
            name=name,
            disk=disk,
            header_offset=header_offset,
            compressed_size=compressed_size,
            file_size=file_size,
            compression=fields[zipfile._CD_COMPRESS_TYPE],
            flags=fields[zipfile._CD_FLAG_BITS],
            crc32=fields[zipfile._CD_CRC],
        )
        position = end
    return entries


class SplitRemoteZip:
    def __init__(
        self, manifest: dict, part_factory: Callable[[dict], io.BufferedIOBase] | None = None
    ):
        parts = manifest.get("archive_parts")
        if (
            manifest.get("schema") != "umi-studio-how2sign-raw-front-train-archive/1"
            or not isinstance(parts, list)
            or not parts
            or any(
                not isinstance(part.get("file_id"), str)
                or not isinstance(part.get("name"), str)
                or type(part.get("size_bytes")) is not int
                or part["size_bytes"] < 1
                for part in parts
            )
        ):
            raise ValueError("split ZIP manifest is incomplete")
        self.manifest = manifest
        self.parts = parts
        self.starts = [0]
        for part in parts:
            self.starts.append(self.starts[-1] + part["size_bytes"])
        if self.starts[-1] != manifest.get("total_size_bytes"):
            raise ValueError("split ZIP part sizes differ from manifest")
        self._factory = part_factory or (
            lambda part: How2SignRemoteZip(
                file_id=part["file_id"],
                archive_name=part["name"],
                archive_size=part["size_bytes"],
            )
        )
        self._readers: dict[int, io.BufferedIOBase] = {}
        self._entries: dict[str, SplitZipEntry] | None = None

    def read_at(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or size > RANGE_CHUNK or offset + size > self.starts[-1]:
            raise ValueError("split ZIP read is outside the bounded archive")
        output = bytearray()
        while size:
            disk = bisect.bisect_right(self.starts, offset) - 1
            if disk not in self._readers:
                self._readers[disk] = self._factory(self.parts[disk])
            reader = self._readers[disk]
            reader.seek(offset - self.starts[disk])
            count = min(size, self.starts[disk + 1] - offset)
            block = reader.read(count)
            if len(block) != count:
                raise RuntimeError("split ZIP part returned a short byte range")
            output.extend(block)
            offset += count
            size -= count
        return bytes(output)

    def load_directory(self) -> dict[str, SplitZipEntry]:
        if self._entries is not None:
            return self._entries
        expected = self.manifest["zip64_central_directory"]
        tail_size = min(100_000, self.parts[-1]["size_bytes"])
        tail = self.read_at(self.starts[-1] - tail_size, tail_size)
        end_position = tail.rfind(zipfile.stringEndArchive64)
        locator_position = tail.rfind(zipfile.stringEndArchive64Locator)
        if end_position < 0 or locator_position < 0 or end_position + 56 > len(tail):
            raise ValueError("split ZIP64 end records are missing")
        end = struct.unpack_from(zipfile.structEndArchive64, tail, end_position)
        locator = struct.unpack_from(zipfile.structEndArchive64Locator, tail, locator_position)
        disk_count = len(self.parts)
        if (
            end[4] != expected["disk"]
            or end[5] != expected["disk"]
            or end[7] != expected["entries"]
            or end[8] != expected["size_bytes"]
            or end[9] != expected["offset_on_disk"]
            or locator[1] != expected["disk"]
            or locator[3] != disk_count
            or locator[2] != self.parts[-1]["size_bytes"] - tail_size + end_position
        ):
            raise ValueError("split ZIP64 directory position differs from manifest")
        start = self.starts[expected["disk"]] + expected["offset_on_disk"]
        directory = self.read_at(start, expected["size_bytes"])
        if hashlib.sha256(directory).hexdigest() != expected["sha256"]:
            raise ValueError("split ZIP central directory differs from manifest")
        entries = parse_central_directory(directory, disk_count)
        if len(entries) != expected["entries"]:
            raise ValueError("split ZIP entry count differs from manifest")
        self._entries = entries
        return entries

    def iter_member(
        self,
        name: str,
        *,
        max_compressed_bytes: int = MAX_MEMBER_COMPRESSED,
        check_health: Callable[[], None] | None = None,
    ) -> Iterator[bytes]:
        entry = self.load_directory()[name]
        if (
            entry.compressed_size > max_compressed_bytes
            or entry.file_size > 2 * max_compressed_bytes
        ):
            raise ValueError("split ZIP member exceeds download size limit")
        header_at = self.starts[entry.disk] + entry.header_offset
        header = struct.unpack(zipfile.structFileHeader, self.read_at(header_at, 30))
        if (
            header[zipfile._FH_SIGNATURE] != zipfile.stringFileHeader
            or header[zipfile._FH_COMPRESSION_METHOD] != entry.compression
            or header[zipfile._FH_GENERAL_PURPOSE_FLAG_BITS] & 1
        ):
            raise ValueError("split ZIP local header differs from central directory")
        name_length = header[zipfile._FH_FILENAME_LENGTH]
        extra_length = header[zipfile._FH_EXTRA_FIELD_LENGTH]
        encoding = "utf-8" if entry.flags & 0x800 else "cp437"
        local_name = self.read_at(header_at + 30, name_length).decode(encoding)
        if local_name != name:
            raise ValueError("split ZIP local filename differs from central directory")
        data_at = header_at + 30 + name_length + extra_length
        decoder = zlib.decompressobj(-15) if entry.compression == zipfile.ZIP_DEFLATED else None
        written = 0
        crc = 0
        for position in range(0, entry.compressed_size, RANGE_CHUNK):
            if check_health:
                check_health()
            count = min(RANGE_CHUNK, entry.compressed_size - position)
            compressed = self.read_at(data_at + position, count)
            result = decoder.decompress(compressed) if decoder else compressed
            written += len(result)
            crc = zlib.crc32(result, crc)
            yield result
        if decoder:
            tail = decoder.flush()
            written += len(tail)
            crc = zlib.crc32(tail, crc)
            if not decoder.eof or decoder.unused_data:
                raise ValueError("split ZIP deflate stream is incomplete")
            if tail:
                yield tail
        if written != entry.file_size or crc != entry.crc32:
            raise ValueError("split ZIP member size or CRC differs from central directory")
        if check_health:
            check_health()
