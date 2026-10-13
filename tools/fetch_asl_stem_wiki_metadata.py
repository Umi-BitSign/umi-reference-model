"""Fetch only metadata from the official ASL STEM Wiki ZIP by HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from benchmark_fleurs_dev import sha256
from how2sign_remote_zip import How2SignRemoteZip

SOURCE_URL = (
    "https://download.microsoft.com/download/4/c/f/"
    "4cfec788-7478-4e47-9a15-ace9b6a96198/ASL_STEM_Wiki.zip"
)
ARCHIVE_NAME = "ASL_STEM_Wiki.zip"
ARCHIVE_SIZE = 201_193_250_549
ARCHIVE_ETAG = (
    '"0xB450267680B5CC429DF45C7875E765E7367F35879CEF1624F23EA38CBE9B4576"'
)
ARCHIVE_LAST_MODIFIED = "Mon, 17 Feb 2025 16:57:59 GMT"
EXPECTED_VIDEOS = 64_266
METADATA_MEMBERS = (
    "ASL_STEM_Wiki/videos.csv",
    "ASL_STEM_Wiki/articles.csv",
    "ASL_STEM_Wiki/use.txt",
    "ASL_STEM_Wiki/article_fulltexts_segmented.csv",
)


class AslStemWikiRemoteZip(How2SignRemoteZip):
    def _confirmation_url(self) -> str:
        return SOURCE_URL


def checked_headers() -> dict:
    request = urllib.request.Request(SOURCE_URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        headers = response.headers
        if (
            response.status != 200
            or int(headers.get("Content-Length", 0)) != ARCHIVE_SIZE
            or headers.get("ETag") != ARCHIVE_ETAG
            or headers.get("Last-Modified") != ARCHIVE_LAST_MODIFIED
            or headers.get("Accept-Ranges") != "bytes"
            or ARCHIVE_NAME not in headers.get("Content-Disposition", "")
        ):
            raise ValueError("ASL STEM Wiki archive identity differs")
        return {
            "content_length": int(headers["Content-Length"]),
            "etag": headers["ETag"],
            "last_modified": headers["Last-Modified"],
        }


def inventory_record(info: zipfile.ZipInfo) -> dict:
    return {
        "name": info.filename,
        "crc32": info.CRC,
        "compressed_size": info.compress_size,
        "file_size": info.file_size,
        "compression": info.compress_type,
        "header_offset": info.header_offset,
    }


def validate_inventory(
    infos: list[zipfile.ZipInfo], expected_videos: int = EXPECTED_VIDEOS
) -> dict[str, zipfile.ZipInfo]:
    by_name = {info.filename: info for info in infos}
    if len(by_name) != len(infos) or any(
        name not in by_name or by_name[name].is_dir() for name in METADATA_MEMBERS
    ):
        raise ValueError("ASL STEM Wiki metadata members differ")
    videos = [
        info
        for info in infos
        if info.filename.startswith("ASL_STEM_Wiki/videos/") and not info.is_dir()
    ]
    if (
        len(videos) != expected_videos
        or any(not info.filename.endswith(".mp4") for info in videos)
        or len({Path(info.filename).name for info in videos}) != expected_videos
        or any(
            info.flag_bits & 1
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            for info in infos
        )
    ):
        raise ValueError("ASL STEM Wiki video inventory differs")
    return by_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    temporary = args.output_dir.with_name(args.output_dir.name + ".partial")
    if args.output_dir.exists() or temporary.exists() or args.manifest.exists():
        parser.error("metadata output already exists or is unfinished")
    headers = checked_headers()
    reader = AslStemWikiRemoteZip(
        file_id="direct-download",
        archive_name=ARCHIVE_NAME,
        archive_size=ARCHIVE_SIZE,
    )
    try:
        with zipfile.ZipFile(reader) as archive:
            infos = archive.infolist()
            by_name = validate_inventory(infos)
            canonical_inventory = json.dumps(
                [inventory_record(info) for info in infos],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            temporary.mkdir(parents=True)
            members = []
            for name in METADATA_MEMBERS:
                info = by_name[name]
                payload = archive.read(info)
                target = temporary / Path(name).name
                with target.open("xb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                members.append(
                    {
                        **inventory_record(info),
                        "local_name": target.name,
                        "sha256": sha256(target),
                    }
                )
        temporary.rename(args.output_dir)
        manifest = {
            "schema": "umi-studio-asl-stem-wiki-source/1",
            "source_url": SOURCE_URL,
            "archive_name": ARCHIVE_NAME,
            "archive_size_bytes": ARCHIVE_SIZE,
            "archive_etag": headers["etag"],
            "archive_last_modified": headers["last_modified"],
            "archive_entry_count": len(infos),
            "archive_video_count": EXPECTED_VIDEOS,
            "archive_inventory_sha256": hashlib.sha256(
                canonical_inventory
            ).hexdigest(),
            "range_requests": reader.requests,
            "range_bytes_downloaded": reader.bytes_downloaded,
            "metadata_members": members,
            "fetcher_sha256": sha256(Path(__file__)),
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        partial_manifest = args.manifest.with_suffix(".partial.json")
        with partial_manifest.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        partial_manifest.rename(args.manifest)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(
        f"pinned {len(infos)} archive entries and {len(members)} metadata members "
        f"using {reader.bytes_downloaded} ranged bytes"
    )


if __name__ == "__main__":
    main()
