from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import platform
import re
import selectors
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import rfc8785

RAW_SCHEMA: Final = "umi-raw-holistic-landmarks/1"
RECEIPT_SCHEMA: Final = "umi-raw-holistic-extraction-receipt/3"
COMPLETION_SCHEMA: Final = "umi-raw-holistic-extraction-completion/1"
VIDEO_PATH: Final = Path("/input/video")
MODEL_PATH: Final = Path("/model/holistic.task")
OUTPUT_ROOT: Final = Path("/output")
RAW_FILENAME: Final = "raw-holistic.npz"
RECEIPT_FILENAME: Final = "receipt.json"
MAXIMUM_SOURCE_VIDEO_BYTES: Final = 256 * 1024 * 1024
MAXIMUM_MODEL_BYTES: Final = 128 * 1024 * 1024
MAXIMUM_RAW_ARTIFACT_BYTES: Final = 32 * 1024 * 1024
MAXIMUM_FRAMES: Final = 120
TARGET_RATE: Final = 8
MINIMUM_CLIP_US: Final = 2_000_000
MAXIMUM_CLIP_US: Final = 15_000_000
MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE: Final = Fraction(60, 1)
MAXIMUM_BOOTSTRAP_CANDIDATE_FRAMES: Final = 902
MAXIMUM_SOURCE_LONG_SIDE: Final = 3840
MAXIMUM_SOURCE_SHORT_SIDE: Final = 2160
MAXIMUM_DERIVED_LONG_SIDE: Final = 1280
MAXIMUM_DERIVED_SHORT_SIDE: Final = 720
FFPROBE_TIMEOUT_SECONDS: Final = 120.0
FFMPEG_TIMEOUT_SECONDS: Final = 120.0
FFPROBE_METADATA_LIMIT: Final = 256 * 1024
FFPROBE_FRAMES_LIMIT: Final = 16 * 1024 * 1024
SUBPROCESS_STDERR_LIMIT: Final = 256 * 1024
ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
SAFE_SAMPLE_ID: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
HEX_SHA256: Final = re.compile(r"[0-9a-f]{64}")
EXPECTED_MODEL_SHA256: Final = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"
ALLOWED_MP4_MAJOR_BRANDS: Final = frozenset(
    {"avc1", "dash", "iso2", "iso3", "iso4", "iso5", "iso6", "isom", "M4V ", "mp41", "mp42"}
)
ALLOWED_BOOTSTRAP_VIDEO_PROFILES: Final = frozenset(
    {
        ("h264", "High", "yuv420p"),
        ("h264", "High 10", "yuv420p10le"),
        ("hevc", "Main 10", "yuv420p10le"),
    }
)
ALLOWED_BOOTSTRAP_AUDIO_CODECS: Final = frozenset({"aac"})
MAXIMUM_SAMPLE_ASPECT_RATIO_DEVIATION: Final = Fraction(1, 100)
SCALE_FLAGS: Final = "bilinear+accurate_rnd+full_chroma_int+bitexact"
IMAGE_TRACKS: Final = (
    ("pose_image", "pose_landmarks", 33),
    ("observed_hand_image_0", "left_hand_landmarks", 21),
    ("observed_hand_image_1", "right_hand_landmarks", 21),
    ("face_image", "face_landmarks", 478),
)
WORLD_TRACKS: Final = (
    ("pose_world", "pose_world_landmarks", 33),
    ("observed_hand_world_0", "left_hand_world_landmarks", 21),
    ("observed_hand_world_1", "right_hand_world_landmarks", 21),
)


class WorkerError(RuntimeError):
    """A bounded extraction failure safe to report without landmark values."""


@dataclass(frozen=True, slots=True)
class MountedInput:
    descriptor: int
    proc_path: str
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class ClipBounds:
    start_us: int
    end_us: int

    @property
    def start(self) -> Fraction:
        return Fraction(self.start_us, 1_000_000)

    @property
    def end(self) -> Fraction:
        return Fraction(self.end_us, 1_000_000)

    @property
    def duration(self) -> Fraction:
        return Fraction(self.end_us - self.start_us, 1_000_000)

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us


@dataclass(frozen=True, slots=True)
class MediaInfo:
    byte_count: int
    format_name: str
    major_brand: str
    codec_name: str
    codec_profile: str
    pixel_format: str
    has_b_frames: int
    audio_codecs: tuple[str, ...]
    audio_stream_profiles: tuple[tuple[str, str, str, int, int], ...]
    sample_aspect_ratio: Fraction
    sample_aspect_ratio_source: str
    coded_width: int
    coded_height: int
    source_displayed_width: int
    source_displayed_height: int
    derived_width: int
    derived_height: int
    rotation_degrees: int
    nominal_frame_rate: Fraction
    average_frame_rate: Fraction
    time_base: Fraction
    stream_start_pts: int
    stream_start_time: Fraction
    source_duration: Fraction
    candidate_frame_pts: tuple[int, ...]
    ffprobe_version: str
    ffmpeg_version: str


@dataclass(frozen=True, slots=True)
class FrameSelection:
    requested_times: tuple[Fraction, ...]
    source_frame_indices: tuple[int, ...]
    source_pts: tuple[int, ...]


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerError("subprocess JSON contains a duplicate object member")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                WorkerError(f"{label} contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"{label} did not return strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"{label} result must be a JSON object")
    return value


def _bounded_capture(
    command: list[str],
    *,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
    pass_fds: tuple[int, ...] = (),
) -> tuple[bytes, bytes]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        raise WorkerError("required media subprocess could not start") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    selected = selectors.DefaultSelector()
    selected.register(process.stdout, selectors.EVENT_READ, ("stdout", stdout_limit))
    selected.register(process.stderr, selectors.EVENT_READ, ("stderr", stderr_limit))
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    try:
        while selected.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError("media subprocess timed out")
            events = selected.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = [(key, selectors.EVENT_READ) for key in selected.get_map().values()]
            for key, _ in events:
                name, limit = key.data
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selected.unregister(key.fileobj)
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > limit:
                    raise WorkerError("media subprocess output exceeded its byte ceiling")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerError("media subprocess timed out")
        return_code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise WorkerError("media subprocess timed out") from exc
    except BaseException:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise
    finally:
        selected.close()
        process.stdout.close()
        process.stderr.close()
    if return_code != 0:
        raise WorkerError(f"media subprocess exited with status {return_code}")
    return bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _bounded_decode_to_file(
    command: list[str],
    *,
    output_descriptor: int,
    expected_bytes: int,
    timeout_seconds: float,
    stderr_limit: int,
    pass_fds: tuple[int, ...],
) -> None:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        raise WorkerError("ffmpeg could not start") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    selected = selectors.DefaultSelector()
    selected.register(process.stdout, selectors.EVENT_READ, "stdout")
    selected.register(process.stderr, selectors.EVENT_READ, "stderr")
    stderr = bytearray()
    observed = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        while selected.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError("ffmpeg frame decode timed out")
            events = selected.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = [(key, selectors.EVENT_READ) for key in selected.get_map().values()]
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selected.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    stderr.extend(chunk)
                    if len(stderr) > stderr_limit:
                        raise WorkerError("ffmpeg diagnostics exceeded their byte ceiling")
                else:
                    observed += len(chunk)
                    if observed > expected_bytes:
                        raise WorkerError("ffmpeg RGB output exceeded its exact byte contract")
                    _write_all(output_descriptor, chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerError("ffmpeg frame decode timed out")
        return_code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise WorkerError("ffmpeg frame decode timed out") from exc
    except BaseException:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise
    finally:
        selected.close()
        process.stdout.close()
        process.stderr.close()
    if return_code != 0:
        raise WorkerError(f"ffmpeg frame decode exited with status {return_code}")
    if observed != expected_bytes:
        raise WorkerError("ffmpeg RGB byte count disagrees with selected displayed frames")


def _hash_descriptor(descriptor: int, *, maximum_bytes: int) -> tuple[str, int]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as exc:
        raise WorkerError("mounted input is not seekable") from exc
    digest = hashlib.sha256()
    observed = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
        if not chunk:
            break
        observed += len(chunk)
        if observed > maximum_bytes:
            raise WorkerError("mounted input exceeded its byte ceiling while reading")
        digest.update(chunk)
    return digest.hexdigest(), observed


def _open_mounted_input(path: Path, *, maximum_bytes: int) -> MountedInput:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise WorkerError("required mounted input is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(initial.st_mode):
        raise WorkerError("required mounted input must be a regular non-symlink file")
    if initial.st_size <= 0 or initial.st_size > maximum_bytes:
        raise WorkerError("mounted input violates its byte ceiling")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WorkerError("required mounted input cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev != initial.st_dev or opened.st_ino != initial.st_ino
        ):
            raise WorkerError("mounted input changed before hashing")
        digest, observed = _hash_descriptor(descriptor, maximum_bytes=maximum_bytes)
        final = os.fstat(descriptor)
        if final.st_size != observed or final.st_size != opened.st_size:
            raise WorkerError("mounted input changed while hashing")
    except BaseException:
        os.close(descriptor)
        raise
    return MountedInput(
        descriptor=descriptor,
        proc_path=f"/proc/self/fd/{descriptor}",
        sha256=digest,
        byte_count=observed,
    )


def _verify_mounted_input_unchanged(value: MountedInput, *, maximum_bytes: int) -> None:
    digest, byte_count = _hash_descriptor(value.descriptor, maximum_bytes=maximum_bytes)
    if digest != value.sha256 or byte_count != value.byte_count:
        raise WorkerError("mounted input changed during extraction")


def _parse_fraction(value: object, *, label: str, positive: bool = True) -> Fraction:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise WorkerError(f"{label} is not a bounded rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise WorkerError(f"{label} is not a valid rational") from exc
    if positive and parsed <= 0:
        raise WorkerError(f"{label} must be positive")
    return parsed


def _parse_int(value: object, *, label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or re.fullmatch(r"-?[0-9]{1,20}", value) is None:
        raise WorkerError(f"{label} is not an integer")
    return int(value)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _to_microseconds(value: Fraction) -> int:
    scaled = value * 1_000_000
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if remainder * 2 > scaled.denominator:
        quotient += 1
    return quotient


def _decimal_microseconds(value_us: int) -> str:
    if value_us < 0:
        raise WorkerError("microsecond timestamp must be non-negative")
    seconds, micros = divmod(value_us, 1_000_000)
    return f"{seconds}.{micros:06d}"


def _candidate_probe_interval_us(
    clip: ClipBounds,
    *,
    stream_start_time: Fraction,
    nominal_frame_rate: Fraction,
    has_b_frames: int,
) -> tuple[int, int]:
    if nominal_frame_rate <= 0:
        raise WorkerError("candidate probe nominal frame rate must be positive")
    if type(has_b_frames) is not int or not 0 <= has_b_frames <= 2:
        raise WorkerError("candidate probe B-frame depth is outside the bootstrap profile")
    start_guard = Fraction(1, 1) / nominal_frame_rate
    end_guard = Fraction(has_b_frames + 1, 1) / nominal_frame_rate
    start = max(Fraction(0), stream_start_time + clip.start - start_guard)
    end = stream_start_time + clip.end + end_guard
    start_us = _to_microseconds(start)
    end_us = _to_microseconds(end)
    if end_us <= start_us:
        raise WorkerError("candidate probe interval is empty")
    return start_us, end_us


def _tool_version(tool: str) -> str:
    stdout, _ = _bounded_capture(
        [tool, "-version"],
        timeout_seconds=5.0,
        stdout_limit=32 * 1024,
        stderr_limit=32 * 1024,
    )
    try:
        first_line = stdout.decode("utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise WorkerError(f"{tool} version output is invalid") from exc
    if not first_line or len(first_line) > 512:
        raise WorkerError(f"{tool} version output is invalid")
    return first_line


def _rotation(stream: dict[str, Any]) -> int:
    rotations: list[int] = []
    side_data = stream.get("side_data_list", [])
    if side_data is not None:
        if not isinstance(side_data, list):
            raise WorkerError("video rotation side data is malformed")
        for entry in side_data:
            if isinstance(entry, dict) and "rotation" in entry:
                rotations.append(_parse_int(entry["rotation"], label="video rotation"))
    tags = stream.get("tags", {})
    if tags is not None:
        if not isinstance(tags, dict):
            raise WorkerError("video stream tags are malformed")
        if "rotate" in tags:
            rotations.append(_parse_int(tags["rotate"], label="video rotation"))
    normalized = {rotation % 360 for rotation in rotations}
    if len(normalized) > 1:
        raise WorkerError("video rotation metadata is contradictory")
    rotation = next(iter(normalized), 0)
    if rotation not in (0, 90, 180, 270):
        raise WorkerError("video rotation is not a right-angle display transform")
    return rotation


def _effective_display_dimensions(
    coded_width: int,
    coded_height: int,
    rotation: int,
    sample_aspect_ratio: Fraction,
) -> tuple[Fraction, Fraction]:
    if rotation in (0, 180):
        return Fraction(coded_width) * sample_aspect_ratio, Fraction(coded_height)
    return Fraction(coded_height), Fraction(coded_width) * sample_aspect_ratio


def _derived_dimensions(width: int | Fraction, height: int | Fraction) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise WorkerError("displayed dimensions must be positive")
    width = Fraction(width)
    height = Fraction(height)
    if width >= height:
        maximum_width, maximum_height = MAXIMUM_DERIVED_LONG_SIDE, MAXIMUM_DERIVED_SHORT_SIDE
    else:
        maximum_width, maximum_height = MAXIMUM_DERIVED_SHORT_SIDE, MAXIMUM_DERIVED_LONG_SIDE
    scale = min(Fraction(1), Fraction(maximum_width, width), Fraction(maximum_height, height))
    if scale == 1 and width.denominator == 1 and height.denominator == 1:
        return int(width), int(height)

    def nearest_bounded_even(value: Fraction, maximum: int) -> int:
        lower = 2 * (value // 2)
        upper = lower + 2
        if upper <= maximum and value - lower > upper - value:
            return upper
        return max(2, lower)

    derived_width = nearest_bounded_even(width * scale, maximum_width)
    derived_height = nearest_bounded_even(height * scale, maximum_height)
    return int(derived_width), int(derived_height)


def _candidate_frame_limit(clip: ClipBounds) -> int:
    interval_limit = _ceil_fraction(clip.duration * MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE) + 2
    return min(MAXIMUM_BOOTSTRAP_CANDIDATE_FRAMES, interval_limit)


def _validate_nominal_frame_rate(rate: Fraction) -> None:
    if rate > MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE:
        raise WorkerError(
            "source frame-rate metadata exceeds the bounded 60-fps bootstrap source profile"
        )


def _sample_aspect_ratio(value: object) -> tuple[Fraction, str]:
    if value in (None, "N/A"):
        return Fraction(1), "unspecified-assumed-square"
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{1,20}:[0-9]{1,20}", value) is None:
        raise WorkerError("source sample aspect ratio is malformed")
    ratio = _parse_fraction(value.replace(":", "/"), label="source sample aspect ratio")
    if abs(ratio - 1) > MAXIMUM_SAMPLE_ASPECT_RATIO_DEVIATION:
        raise WorkerError("bootstrap source sample aspect ratio differs from square by over 1%")
    return ratio, "declared-near-square"


def _validate_candidate_frames(
    points: list[int],
    *,
    time_base: Fraction,
    clip: ClipBounds,
) -> None:
    if not points or len(points) > _candidate_frame_limit(clip):
        raise WorkerError("clip frame count violates its bounded 60-fps bootstrap interval")
    if any(later <= earlier for earlier, later in pairwise(points)):
        raise WorkerError("clip presentation timestamps must be strictly increasing")
    del time_base


def _probe_media(
    video: MountedInput, clip: ClipBounds | None, *, require_umi_whole_video: bool = False
) -> tuple[MediaInfo, ClipBounds]:
    ffprobe_version = _tool_version("ffprobe")
    ffmpeg_version = _tool_version("ffmpeg")
    metadata_stdout, _ = _bounded_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "format=format_name,duration,size:format_tags=major_brand,compatible_brands:"
                "stream=index,codec_type,codec_name,profile,pix_fmt,sample_fmt,sample_rate,"
                "channels,width,height,sample_aspect_ratio,has_b_frames,"
                "r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration_ts:"
                "stream_tags=rotate:stream_side_data=rotation"
            ),
            "-of",
            "json",
            video.proc_path,
        ],
        timeout_seconds=FFPROBE_TIMEOUT_SECONDS,
        stdout_limit=FFPROBE_METADATA_LIMIT,
        stderr_limit=SUBPROCESS_STDERR_LIMIT,
        pass_fds=(video.descriptor,),
    )
    metadata = _decode_json(metadata_stdout, label="ffprobe metadata")
    streams = metadata.get("streams")
    format_data = metadata.get("format")
    if not isinstance(streams, list) or not isinstance(format_data, dict):
        raise WorkerError("ffprobe metadata omits streams or format")
    video_streams = [entry for entry in streams if entry.get("codec_type") == "video"]
    audio_streams = [entry for entry in streams if entry.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise WorkerError("source must contain exactly one video stream")
    if len(audio_streams) > 1:
        raise WorkerError("bootstrap source profile permits at most one ignored audio stream")
    stream = video_streams[0]
    if not isinstance(stream, dict):
        raise WorkerError("video stream metadata is malformed")
    format_name = format_data.get("format_name")
    tags = format_data.get("tags")
    if not isinstance(format_name, str) or not isinstance(tags, dict):
        raise WorkerError("source container metadata is incomplete")
    major_brand = tags.get("major_brand")
    if (
        "mp4" not in format_name.split(",")
        or not isinstance(major_brand, str)
        or major_brand not in ALLOWED_MP4_MAJOR_BRANDS
    ):
        raise WorkerError("source must be an MP4-family file, not QuickTime MOV or 3GP")
    codec_name = stream.get("codec_name")
    codec_profile = stream.get("profile")
    pixel_format = stream.get("pix_fmt")
    has_b_frames = _parse_int(stream.get("has_b_frames"), label="B-frame depth")
    if (
        not isinstance(codec_name, str)
        or not isinstance(codec_profile, str)
        or not isinstance(pixel_format, str)
        or (codec_name, codec_profile, pixel_format) not in ALLOWED_BOOTSTRAP_VIDEO_PROFILES
        or not 0 <= has_b_frames <= 2
    ):
        raise WorkerError("video codec, profile, or pixel format is outside the bootstrap pin set")
    audio_codecs: list[str] = []
    audio_stream_profiles: list[tuple[str, str, str, int, int]] = []
    for audio in audio_streams:
        codec = audio.get("codec_name")
        if (
            not isinstance(codec, str)
            or codec not in ALLOWED_BOOTSTRAP_AUDIO_CODECS
            or audio.get("profile") != "LC"
            or audio.get("sample_fmt") != "fltp"
            or _parse_int(audio.get("sample_rate"), label="audio sample rate") != 48_000
            or _parse_int(audio.get("channels"), label="audio channel count") != 2
        ):
            raise WorkerError("audio stream is outside the ignored AAC-LC pin set")
        audio_codecs.append(codec)
        audio_stream_profiles.append((codec, "LC", "fltp", 48_000, 2))
    sample_aspect_ratio, sample_aspect_ratio_source = _sample_aspect_ratio(
        stream.get("sample_aspect_ratio")
    )
    reported_size = _parse_int(format_data.get("size"), label="container byte count")
    if reported_size != video.byte_count:
        raise WorkerError("ffprobe byte count disagrees with the mounted source")
    width = _parse_int(stream.get("width"), label="coded width")
    height = _parse_int(stream.get("height"), label="coded height")
    if width <= 0 or height <= 0:
        raise WorkerError("coded video dimensions must be positive")
    rotation = _rotation(stream)
    source_displayed_width, source_displayed_height = (
        (height, width) if rotation in (90, 270) else (width, height)
    )
    if (
        max(source_displayed_width, source_displayed_height) > MAXIMUM_SOURCE_LONG_SIDE
        or min(source_displayed_width, source_displayed_height) > MAXIMUM_SOURCE_SHORT_SIDE
    ):
        raise WorkerError("bootstrap source exceeds the bounded 4K orientation-neutral profile")
    effective_display_width, effective_display_height = _effective_display_dimensions(
        width,
        height,
        rotation,
        sample_aspect_ratio,
    )
    derived_width, derived_height = _derived_dimensions(
        effective_display_width, effective_display_height
    )
    nominal_rate = _parse_fraction(stream.get("r_frame_rate"), label="nominal frame rate")
    average_rate = _parse_fraction(stream.get("avg_frame_rate"), label="average frame rate")
    _validate_nominal_frame_rate(nominal_rate)
    time_base = _parse_fraction(stream.get("time_base"), label="video time base")
    stream_start_pts = _parse_int(stream.get("start_pts", 0), label="video start PTS")
    stream_start_time = stream_start_pts * time_base
    duration_ticks_raw = stream.get("duration_ts")
    if duration_ticks_raw is not None:
        source_duration = _parse_int(duration_ticks_raw, label="video duration ticks") * time_base
    else:
        source_duration = _parse_fraction(format_data.get("duration"), label="container duration")
    if require_umi_whole_video:
        if clip is not None:
            raise WorkerError("whole-video extraction cannot carry caller-supplied clip bounds")
        clip_end_us = _to_microseconds(source_duration)
        if (
            video.byte_count > 16 * 1024 * 1024
            or codec_name != "h264"
            or audio_streams
            or max(source_displayed_width, source_displayed_height) > 1280
            or min(source_displayed_width, source_displayed_height) > 720
            or nominal_rate > 30
            or not MINIMUM_CLIP_US <= clip_end_us <= MAXIMUM_CLIP_US
            or not Fraction(MINIMUM_CLIP_US, 1_000_000)
            <= source_duration
            <= Fraction(MAXIMUM_CLIP_US, 1_000_000)
        ):
            raise WorkerError("source is outside the bounded UMI whole-video media profile")
        clip = ClipBounds(0, clip_end_us)
    if clip is None:
        raise WorkerError("timed extraction requires exact clip bounds")
    maximum_clip_end_overshoot = Fraction(1, 1) / nominal_rate
    clip_end_overshoot = max(Fraction(0), clip.end - source_duration)
    if (
        source_duration <= 0
        or clip.start >= source_duration
        or clip_end_overshoot > maximum_clip_end_overshoot
    ):
        raise WorkerError("requested clip interval escapes the mounted source duration")

    candidate_probe_start_us, candidate_probe_end_us = _candidate_probe_interval_us(
        clip,
        stream_start_time=stream_start_time,
        nominal_frame_rate=nominal_rate,
        has_b_frames=has_b_frames,
    )
    frames_stdout, _ = _bounded_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            (
                f"{_decimal_microseconds(candidate_probe_start_us)}"
                f"%{_decimal_microseconds(candidate_probe_end_us)}"
            ),
            "-show_frames",
            "-show_entries",
            "frame=pts",
            "-of",
            "json",
            video.proc_path,
        ],
        timeout_seconds=FFPROBE_TIMEOUT_SECONDS,
        stdout_limit=FFPROBE_FRAMES_LIMIT,
        stderr_limit=SUBPROCESS_STDERR_LIMIT,
        pass_fds=(video.descriptor,),
    )
    frames_data = _decode_json(frames_stdout, label="ffprobe clip frames")
    frames = frames_data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise WorkerError("requested clip has no decodable presentation timestamps")
    candidate_points: list[int] = []
    for index, entry in enumerate(frames):
        if not isinstance(entry, dict):
            raise WorkerError("ffprobe frame entry is malformed")
        point = _parse_int(entry.get("pts"), label=f"frame {index} PTS")
        source_time = point * time_base - stream_start_time
        if clip.start <= source_time < clip.end:
            candidate_points.append(point)
    _validate_candidate_frames(candidate_points, time_base=time_base, clip=clip)
    return MediaInfo(
        byte_count=video.byte_count,
        format_name=format_name,
        major_brand=major_brand,
        codec_name=codec_name,
        codec_profile=codec_profile,
        pixel_format=pixel_format,
        has_b_frames=has_b_frames,
        audio_codecs=tuple(audio_codecs),
        audio_stream_profiles=tuple(audio_stream_profiles),
        sample_aspect_ratio=sample_aspect_ratio,
        sample_aspect_ratio_source=sample_aspect_ratio_source,
        coded_width=width,
        coded_height=height,
        source_displayed_width=source_displayed_width,
        source_displayed_height=source_displayed_height,
        derived_width=derived_width,
        derived_height=derived_height,
        rotation_degrees=rotation,
        nominal_frame_rate=nominal_rate,
        average_frame_rate=average_rate,
        time_base=time_base,
        stream_start_pts=stream_start_pts,
        stream_start_time=stream_start_time,
        source_duration=source_duration,
        candidate_frame_pts=tuple(candidate_points),
        ffprobe_version=ffprobe_version,
        ffmpeg_version=ffmpeg_version,
    ), clip


def _container_platform() -> str:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "linux/arm64"
    if machine in {"amd64", "x86_64"}:
        return "linux/amd64"
    raise WorkerError("worker architecture is outside the pinned container platforms")


def _select_frames(media: MediaInfo, clip: ClipBounds) -> FrameSelection:
    count = _ceil_fraction(clip.duration * TARGET_RATE)
    if count < 16 or count > MAXIMUM_FRAMES:
        raise WorkerError("8-Hz selector produced an invalid frame count")
    frame_times = tuple(
        point * media.time_base - media.stream_start_time for point in media.candidate_frame_pts
    )
    requested: list[Fraction] = []
    indices: list[int] = []
    source_pts: list[int] = []
    for bin_index in range(count):
        center = clip.start + Fraction(2 * bin_index + 1, 2 * count) * clip.duration
        selected_index = min(
            range(len(frame_times)),
            key=lambda index: (
                abs(frame_times[index] - center),
                frame_times[index],
                index,
            ),
        )
        requested.append(center)
        indices.append(selected_index)
        source_pts.append(media.candidate_frame_pts[selected_index])
    if any(later < earlier for earlier, later in pairwise(indices)):
        raise WorkerError("frame selector violated monotonic source ordering")
    requested_us = tuple(_to_microseconds(value) for value in requested)
    requested_ms = tuple(value // 1000 for value in requested_us)
    if any(later <= earlier for earlier, later in pairwise(requested_ms)):
        raise WorkerError("selected MediaPipe VIDEO timestamps are not strictly increasing")
    return FrameSelection(
        requested_times=tuple(requested),
        source_frame_indices=tuple(indices),
        source_pts=tuple(source_pts),
    )


def _select_filter(points: list[int]) -> str:
    if not points or len(points) > MAXIMUM_FRAMES:
        raise WorkerError("ffmpeg selection PTS set is invalid")
    if points != sorted(set(points)):
        raise WorkerError("ffmpeg selection PTS values must be sorted and unique")
    terms = [f"eq(pts\\,{point})" for point in points]
    while len(terms) > 1:
        terms = [
            f"({terms[index]}+{terms[index + 1]})" if index + 1 < len(terms) else terms[index]
            for index in range(0, len(terms), 2)
        ]
    return "select=" + terms[0]


def _decode_command(
    *,
    video_proc_path: str,
    video_descriptor: int,
    media: MediaInfo,
    selection: FrameSelection,
    clip: ClipBounds,
    input_mirrored: bool,
) -> list[str]:
    del video_descriptor
    unique_points = sorted(set(selection.source_pts))
    filters = [_select_filter(unique_points)]
    if input_mirrored:
        filters.append("hflip")
    if (
        media.derived_width != media.source_displayed_width
        or media.derived_height != media.source_displayed_height
    ):
        filters.append(f"scale={media.derived_width}:{media.derived_height}:flags={SCALE_FLAGS}")
    filters.append("setsar=1/1")
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-hwaccel",
        "none",
        "-copyts",
        "-ss",
        _decimal_microseconds(clip.start_us),
        "-i",
        video_proc_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(len(unique_points)),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _decode_selected_rgb(
    video: MountedInput,
    media: MediaInfo,
    selection: FrameSelection,
    clip: ClipBounds,
    *,
    input_mirrored: bool,
) -> Path:
    unique_points = sorted(set(selection.source_pts))
    raw_path = Path("/tmp/selected-rgb.bin")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(raw_path, flags, 0o600)
    except OSError as exc:
        raise WorkerError("temporary RGB output cannot be created safely") from exc
    expected = len(unique_points) * media.derived_width * media.derived_height * 3
    try:
        _bounded_decode_to_file(
            _decode_command(
                video_proc_path=video.proc_path,
                video_descriptor=video.descriptor,
                media=media,
                selection=selection,
                clip=clip,
                input_mirrored=input_mirrored,
            ),
            output_descriptor=descriptor,
            expected_bytes=expected,
            timeout_seconds=FFMPEG_TIMEOUT_SECONDS,
            stderr_limit=SUBPROCESS_STDERR_LIMIT,
            pass_fds=(video.descriptor,),
        )
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if metadata.st_size != expected:
        raise WorkerError("ffmpeg RGB byte count disagrees with selected displayed frames")
    return raw_path


def _empty_track(frame_count: int, point_count: int) -> dict[str, np.ndarray]:
    return {
        "coordinates": np.zeros((frame_count, point_count, 3), dtype=np.float32),
        "coordinate_available": np.zeros((frame_count, point_count), dtype=np.bool_),
        "presence": np.zeros((frame_count, point_count), dtype=np.float32),
        "presence_available": np.zeros((frame_count, point_count), dtype=np.bool_),
        "visibility": np.zeros((frame_count, point_count), dtype=np.float32),
        "visibility_available": np.zeros((frame_count, point_count), dtype=np.bool_),
    }


def _finite_float32(value: object, *, label: str) -> np.float32:
    if isinstance(value, bool):
        raise WorkerError(f"{label} is not a finite coordinate")
    try:
        result = np.float32(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkerError(f"{label} is not a finite coordinate") from exc
    if not np.isfinite(result):
        raise WorkerError(f"{label} is not a finite coordinate")
    return result


def _record_track(
    destination: dict[str, np.ndarray],
    frame_index: int,
    landmarks: object,
    *,
    expected_points: int,
    label: str,
) -> None:
    if not isinstance(landmarks, list):
        raise WorkerError(f"{label} result is not a landmark list")
    if not landmarks:
        return
    if len(landmarks) != expected_points:
        raise WorkerError(f"{label} returned an unexpected landmark count")
    for point_index, landmark in enumerate(landmarks):
        for coordinate_index, field in enumerate(("x", "y", "z")):
            if not hasattr(landmark, field):
                raise WorkerError(f"{label} landmark omits {field}")
            destination["coordinates"][frame_index, point_index, coordinate_index] = (
                _finite_float32(getattr(landmark, field), label=f"{label} {field}")
            )
        destination["coordinate_available"][frame_index, point_index] = True
        for score in ("presence", "visibility"):
            value = getattr(landmark, score, None)
            if value is None:
                continue
            destination[score][frame_index, point_index] = _finite_float32(
                value, label=f"{label} {score}"
            )
            destination[f"{score}_available"][frame_index, point_index] = True


def _track_arrays(frame_count: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: _empty_track(frame_count, point_count)
        for name, _, point_count in (*IMAGE_TRACKS, *WORLD_TRACKS)
    }


def _extract_landmarks(
    model: MountedInput,
    media: MediaInfo,
    selection: FrameSelection,
    clip: ClipBounds,
    raw_rgb_path: Path,
) -> tuple[dict[str, np.ndarray], list[str], str]:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception as exc:
        raise WorkerError("MediaPipe 1.0.1 could not be imported") from exc

    frame_count = len(selection.requested_times)
    tracks = _track_arrays(frame_count)
    requested_us = [_to_microseconds(value) for value in selection.requested_times]
    source_times = [
        point * media.time_base - media.stream_start_time for point in selection.source_pts
    ]
    source_us = [_to_microseconds(value) for value in source_times]
    arrays: dict[str, np.ndarray] = {
        "timestamps_us": np.asarray(requested_us, dtype=np.int64),
        "requested_timestamps_us": np.asarray(requested_us, dtype=np.int64),
        "source_timestamps_us": np.asarray(source_us, dtype=np.int64),
        "displayed_timestamps_us": np.asarray(source_us, dtype=np.int64),
        "source_pts": np.asarray(selection.source_pts, dtype=np.int64),
        "source_time_base": np.asarray(
            [media.time_base.numerator, media.time_base.denominator], dtype=np.int64
        ),
        "source_frame_indices": np.asarray(selection.source_frame_indices, dtype=np.int32),
        "clip_bounds_us": np.asarray([clip.start_us, clip.end_us], dtype=np.int64),
        "displayed_rgb_sha256": np.zeros((frame_count, 32), dtype=np.uint8),
    }
    for track_name, fields in tracks.items():
        for field_name, values in fields.items():
            arrays[f"{track_name}_{field_name}"] = values

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model.proc_path,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_suppression_threshold=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    unique_points = sorted(set(selection.source_pts))
    occurrences: dict[int, list[int]] = {point: [] for point in unique_points}
    for output_index, source_point in enumerate(selection.source_pts):
        occurrences[source_point].append(output_index)
    frame_bytes = media.derived_width * media.derived_height * 3
    displayed_hashes: list[str] = [""] * frame_count
    try:
        with vision.HolisticLandmarker.create_from_options(options) as landmarker:
            with raw_rgb_path.open("rb", buffering=0) as stream:
                for source_point in unique_points:
                    raw_frame = stream.read(frame_bytes)
                    if len(raw_frame) != frame_bytes:
                        raise WorkerError("temporary RGB stream ended before all selected frames")
                    digest = hashlib.sha256(raw_frame).digest()
                    image_data = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        media.derived_height, media.derived_width, 3
                    )
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
                    for output_index in occurrences[source_point]:
                        arrays["displayed_rgb_sha256"][output_index] = np.frombuffer(
                            digest, dtype=np.uint8
                        )
                        displayed_hashes[output_index] = digest.hex()
                        timestamp_ms = requested_us[output_index] // 1000
                        result = landmarker.detect_for_video(image, timestamp_ms)
                        for name, attribute, point_count in (*IMAGE_TRACKS, *WORLD_TRACKS):
                            _record_track(
                                tracks[name],
                                output_index,
                                getattr(result, attribute, None),
                                expected_points=point_count,
                                label=name,
                            )
                if stream.read(1):
                    raise WorkerError("temporary RGB stream contains unexpected trailing bytes")
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError("MediaPipe Holistic VIDEO inference failed") from exc
    return arrays, displayed_hashes, mp.__version__


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _encode_raw_npz(
    *, sample_id: str, arrays: dict[str, np.ndarray]
) -> tuple[bytes, list[dict[str, object]]]:
    metadata = {
        "schema": RAW_SCHEMA,
        "sample_id": sample_id,
        "status": "ex-203-candidate-only",
        "canonical_or_release_quality": False,
        "hand_slot_semantics": "opaque-observation-slots-mapper-assigns-anatomical-side",
        "track_prefixes": [name for name, _, _ in (*IMAGE_TRACKS, *WORLD_TRACKS)],
    }
    materialized = dict(arrays)
    materialized["metadata_json_utf8"] = np.frombuffer(
        rfc8785.dumps(metadata), dtype=np.uint8
    ).copy()
    inventory: list[dict[str, object]] = []
    members: dict[str, bytes] = {}
    for name in sorted(materialized):
        array = materialized[name]
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise WorkerError("raw artifact array contract contains an invalid array")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise WorkerError("raw artifact contains a non-finite floating-point value")
        if array.dtype.kind == "f" and np.signbit(array[array == 0]).any():
            raise WorkerError("raw artifact contains negative zero")
        inventory.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "tensor_sha256": _array_sha256(array),
            }
        )
        members[f"{name}.npy"] = _npy_bytes(array)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for member_name in sorted(members):
            info = zipfile.ZipInfo(member_name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, members[member_name])
    payload = output.getvalue()
    if not payload or len(payload) > MAXIMUM_RAW_ARTIFACT_BYTES:
        raise WorkerError("raw Holistic NPZ violates its byte ceiling")
    return payload, inventory


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("file publication made no progress")
        offset += written


def _write_private_create_once(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("publication target is not a regular file")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        if os.fstat(descriptor).st_size != len(payload):
            raise OSError("published file size changed")
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _publish(*, sample_id: str, raw_payload: bytes, receipt_payload: bytes) -> tuple[Path, Path]:
    try:
        root_metadata = OUTPUT_ROOT.lstat()
    except OSError as exc:
        raise WorkerError("output mount is unavailable") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or OUTPUT_ROOT.is_symlink():
        raise WorkerError("output mount must be a non-symlink directory")
    stage = OUTPUT_ROOT / f".{sample_id}.stage-{uuid.uuid4().hex}"
    final = OUTPUT_ROOT / sample_id
    try:
        os.mkdir(stage, 0o700)
        _write_private_create_once(stage / RAW_FILENAME, raw_payload)
        _write_private_create_once(stage / RECEIPT_FILENAME, receipt_payload)
        stage_descriptor = os.open(
            stage,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        _rename_directory_no_replace(stage, final)
        root_descriptor = os.open(
            OUTPUT_ROOT,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
    except FileExistsError as exc:
        raise WorkerError("sample output already exists; publication is create-once") from exc
    except OSError as exc:
        raise WorkerError("atomic owner-only output publication failed") from exc
    return final / RAW_FILENAME, final / RECEIPT_FILENAME


def _receipt_content_sha256(receipt_without_digest: dict[str, object]) -> str:
    return hashlib.sha256(
        b"umi-raw-holistic-extraction-receipt-v3\0" + rfc8785.dumps(receipt_without_digest)
    ).hexdigest()


def _build_receipt(
    *,
    sample_id: str,
    container_image_id: str,
    video: MountedInput,
    model: MountedInput,
    clip: ClipBounds,
    input_mirrored: bool,
    media: MediaInfo,
    selection: FrameSelection,
    displayed_hashes: list[str],
    mediapipe_version: str,
    raw_payload: bytes,
    inventory: list[dict[str, object]],
) -> dict[str, object]:
    source_times = [
        point * media.time_base - media.stream_start_time for point in selection.source_pts
    ]
    effective_display_width, effective_display_height = _effective_display_dimensions(
        media.coded_width,
        media.coded_height,
        media.rotation_degrees,
        media.sample_aspect_ratio,
    )
    candidate_probe_interval_us = _candidate_probe_interval_us(
        clip,
        stream_start_time=media.stream_start_time,
        nominal_frame_rate=media.nominal_frame_rate,
        has_b_frames=media.has_b_frames,
    )
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "sample_id": sample_id,
        "status": "ex-203-candidate-only",
        "claim_boundary": {
            "canonical_or_release_quality": False,
            "cross_runtime_equivalence_established": False,
            "source_is_umi_challenge_media": False,
            "purpose": "bootstrap-timed-source-isolated-raw-holistic-research",
        },
        "container": {
            "image_id": container_image_id,
            "platform": _container_platform(),
            "network": "none",
            "root_filesystem": "read-only",
            "worker_uid": os.geteuid(),
        },
        "input_video": {
            "sha256": video.sha256,
            "byte_count": video.byte_count,
            "mounted_read_only": True,
            "source_ceiling_bytes": MAXIMUM_SOURCE_VIDEO_BYTES,
        },
        "model": {
            "sha256": model.sha256,
            "byte_count": model.byte_count,
            "mounted_read_only": True,
        },
        "source_media": {
            "profile": "bounded-bootstrap-source-not-umi-challenge-media",
            "container": "mp4",
            "format_name": media.format_name,
            "major_brand": media.major_brand,
            "codec": media.codec_name,
            "codec_profile": media.codec_profile,
            "pixel_format": media.pixel_format,
            "has_b_frames": media.has_b_frames,
            "audio_stream_count": len(media.audio_codecs),
            "audio_codecs": list(media.audio_codecs),
            "audio_streams": [
                {
                    "codec": codec,
                    "profile": profile,
                    "sample_format": sample_format,
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                }
                for codec, profile, sample_format, sample_rate, channels in (
                    media.audio_stream_profiles
                )
            ],
            "audio_policy": "ignored-by-ffmpeg-an",
            "sample_aspect_ratio": _fraction_string(media.sample_aspect_ratio),
            "sample_aspect_ratio_source": media.sample_aspect_ratio_source,
            "derived_sample_aspect_ratio": "1/1",
            "coded_dimensions": [media.coded_width, media.coded_height],
            "displayed_dimensions": [
                media.source_displayed_width,
                media.source_displayed_height,
            ],
            "display_aspect_ratio": _fraction_string(
                effective_display_width / effective_display_height
            ),
            "display_rotation_degrees": media.rotation_degrees,
            "display_autorotation": "ffmpeg-default-enabled",
            "input_mirrored": input_mirrored,
            "canonical_unmirroring": (
                "ffmpeg-hflip-before-extraction" if input_mirrored else "not-required"
            ),
            "duration": _fraction_string(media.source_duration),
            "duration_us": _to_microseconds(media.source_duration),
            "nominal_frame_rate": _fraction_string(media.nominal_frame_rate),
            "average_frame_rate": _fraction_string(media.average_frame_rate),
            "clip_end_tolerance_policy": "source-metadata-rounding-at-most-one-nominal-frame-v1",
            "clip_end_overshoot": _fraction_string(
                max(Fraction(0), clip.end - media.source_duration)
            ),
            "maximum_clip_end_overshoot": _fraction_string(
                Fraction(1, 1) / media.nominal_frame_rate
            ),
            "maximum_source_frame_rate": _fraction_string(MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE),
            "source_time_base": _fraction_string(media.time_base),
            "stream_start_pts": media.stream_start_pts,
            "stream_start_time": _fraction_string(media.stream_start_time),
            "candidate_frame_count": len(media.candidate_frame_pts),
            "candidate_frame_limit": _candidate_frame_limit(clip),
            "candidate_frame_pts": list(media.candidate_frame_pts),
            "candidate_probe_policy": (
                "absolute-stream-time-with-one-frame-start-and-bframe-plus-one-end-guard-v1"
            ),
            "candidate_probe_interval_us": list(candidate_probe_interval_us),
        },
        "clip": {
            "clip_start_us": clip.start_us,
            "clip_end_us": clip.end_us,
            "duration_us": clip.duration_us,
            "interval": "half-open-[start,end)",
        },
        "derived_view": {
            "dimensions": [media.derived_width, media.derived_height],
            "orientation_neutral_bound": [1280, 720],
            "pixel_format": "rgb24",
            "aspect_policy": "apply-source-sar-then-nearest-bounded-even-per-axis",
            "pixel_aspect_filter": "setsar=1/1",
            "scale_filter": (
                "identity"
                if (
                    media.derived_width == media.source_displayed_width
                    and media.derived_height == media.source_displayed_height
                )
                else f"scale={media.derived_width}:{media.derived_height}:flags={SCALE_FLAGS}"
            ),
        },
        "selection": {
            "algorithm": "8hz-equal-clip-bin-center-nearest-ffprobe-pts-v2",
            "tie_break": "earlier-presentation-timestamp-then-earlier-interval-frame-index",
            "clip_bounds_us": [clip.start_us, clip.end_us],
            "frame_count": len(selection.requested_times),
            "requested_times": [_fraction_string(value) for value in selection.requested_times],
            "requested_timestamps_us": [
                _to_microseconds(value) for value in selection.requested_times
            ],
            "source_frame_indices": list(selection.source_frame_indices),
            "source_frame_index_basis": "ffprobe-filtered-clip-candidate-sequence-zero-based",
            "source_pts": list(selection.source_pts),
            "source_times": [_fraction_string(value) for value in source_times],
            "source_timestamps_us": [_to_microseconds(value) for value in source_times],
            "displayed_rgb_sha256": displayed_hashes,
        },
        "extractor": {
            "mediapipe_version": mediapipe_version,
            "task": "HolisticLandmarker",
            "running_mode": "VIDEO",
            "invocation": "synchronous-detect_for_video",
            "delegate": "CPU",
            "thresholds": {
                "min_face_detection_confidence": "1/2",
                "min_face_suppression_threshold": "1/2",
                "min_face_landmarks_confidence": "1/2",
                "min_pose_detection_confidence": "1/2",
                "min_pose_suppression_threshold": "1/2",
                "min_pose_landmarks_confidence": "1/2",
                "min_hand_landmarks_confidence": "1/2",
            },
            "ffprobe_version": media.ffprobe_version,
            "ffmpeg_version": media.ffmpeg_version,
        },
        "raw_artifact": {
            "schema": RAW_SCHEMA,
            "filename": RAW_FILENAME,
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
            "byte_count": len(raw_payload),
            "array_inventory": inventory,
            "hand_observation_slots": [
                {
                    "slot": 0,
                    "image_source_attribute": "left_hand_landmarks",
                    "world_source_attribute": "left_hand_world_landmarks",
                },
                {
                    "slot": 1,
                    "image_source_attribute": "right_hand_landmarks",
                    "world_source_attribute": "right_hand_world_landmarks",
                },
            ],
            "anatomical_side_assignment": "deferred-to-mapping-layer",
        },
        "publication": {
            "directory": sample_id,
            "raw_filename": RAW_FILENAME,
            "receipt_filename": RECEIPT_FILENAME,
            "mode": "atomic-directory-renameat2-noreplace",
            "owner_only": True,
        },
    }
    receipt["content_sha256"] = _receipt_content_sha256(receipt)
    return receipt


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--expected-video-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--clip-start-us")
    parser.add_argument("--clip-end-us")
    parser.add_argument("--whole-video", action="store_true")
    parser.add_argument("--input-mirrored", choices=("false", "true"), required=True)
    parser.add_argument("--help", action="help")
    arguments = parser.parse_args()
    if SAFE_SAMPLE_ID.fullmatch(arguments.sample_id) is None:
        raise WorkerError("sample_id is not a safe lowercase identifier")
    for name in ("expected_video_sha256", "expected_model_sha256"):
        if HEX_SHA256.fullmatch(getattr(arguments, name)) is None:
            raise WorkerError(f"{name} is not a lowercase SHA-256 digest")
    if arguments.expected_model_sha256 != EXPECTED_MODEL_SHA256:
        raise WorkerError("model digest is not the pinned Holistic task model")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", arguments.container_image_id) is None:
        raise WorkerError("container_image_id is not a local immutable image ID")
    if arguments.whole_video:
        if arguments.clip_start_us is not None or arguments.clip_end_us is not None:
            raise WorkerError("whole-video extraction refuses caller-supplied clip bounds")
        arguments.clip = None
    else:
        if arguments.clip_start_us is None or arguments.clip_end_us is None:
            raise WorkerError("timed extraction requires both clip bounds")
        clip_start_us = _parse_int(arguments.clip_start_us, label="clip_start_us")
        clip_end_us = _parse_int(arguments.clip_end_us, label="clip_end_us")
        if (
            clip_start_us < 0
            or not MINIMUM_CLIP_US <= clip_end_us - clip_start_us <= MAXIMUM_CLIP_US
        ):
            raise WorkerError("clip bounds must name a 2-through-15-second interval")
        arguments.clip = ClipBounds(clip_start_us, clip_end_us)
    arguments.input_mirrored_bool = arguments.input_mirrored == "true"
    if os.geteuid() == 0:
        raise WorkerError("Holistic worker refuses to run as root")
    return arguments


def main() -> int:
    arguments = _arguments()
    video = _open_mounted_input(VIDEO_PATH, maximum_bytes=MAXIMUM_SOURCE_VIDEO_BYTES)
    try:
        model = _open_mounted_input(MODEL_PATH, maximum_bytes=MAXIMUM_MODEL_BYTES)
        try:
            if video.sha256 != arguments.expected_video_sha256:
                raise WorkerError("mounted video digest changed after host validation")
            if model.sha256 != arguments.expected_model_sha256:
                raise WorkerError("mounted model digest changed after host validation")
            media, clip = _probe_media(
                video,
                arguments.clip,
                require_umi_whole_video=arguments.whole_video,
            )
            selection = _select_frames(media, clip)
            raw_rgb_path = _decode_selected_rgb(
                video,
                media,
                selection,
                clip,
                input_mirrored=arguments.input_mirrored_bool,
            )
            arrays, displayed_hashes, mediapipe_version = _extract_landmarks(
                model, media, selection, clip, raw_rgb_path
            )
            if mediapipe_version != "1.0.1":
                raise WorkerError("loaded MediaPipe version does not match the pinned worker")
            _verify_mounted_input_unchanged(video, maximum_bytes=MAXIMUM_SOURCE_VIDEO_BYTES)
            _verify_mounted_input_unchanged(model, maximum_bytes=MAXIMUM_MODEL_BYTES)
            raw_payload, inventory = _encode_raw_npz(sample_id=arguments.sample_id, arrays=arrays)
            receipt = _build_receipt(
                sample_id=arguments.sample_id,
                container_image_id=arguments.container_image_id,
                video=video,
                model=model,
                clip=clip,
                input_mirrored=arguments.input_mirrored_bool,
                media=media,
                selection=selection,
                displayed_hashes=displayed_hashes,
                mediapipe_version=mediapipe_version,
                raw_payload=raw_payload,
                inventory=inventory,
            )
            receipt_payload = rfc8785.dumps(receipt)
            raw_path, receipt_path = _publish(
                sample_id=arguments.sample_id,
                raw_payload=raw_payload,
                receipt_payload=receipt_payload,
            )
        finally:
            os.close(model.descriptor)
    finally:
        os.close(video.descriptor)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "sample_id": arguments.sample_id,
        "container_image_id": arguments.container_image_id,
        "input_video_sha256": arguments.expected_video_sha256,
        "model_sha256": arguments.expected_model_sha256,
        "clip_start_us": clip.start_us,
        "clip_end_us": clip.end_us,
        "input_mirrored": arguments.input_mirrored_bool,
        "raw_artifact_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "receipt_content_sha256": receipt["content_sha256"],
        "raw_relative_path": f"{arguments.sample_id}/{raw_path.name}",
        "receipt_relative_path": f"{arguments.sample_id}/{receipt_path.name}",
    }
    encoded_completion = rfc8785.dumps(completion) + b"\n"
    if len(encoded_completion) > 4096:
        raise WorkerError("completion record unexpectedly exceeds its byte ceiling")
    os.write(sys.stdout.fileno(), encoded_completion)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkerError as error:
        message = str(error)
        if len(message) > 512:
            message = "bounded Holistic worker failure"
        os.write(sys.stderr.fileno(), f"worker error: {message}\n".encode())
        raise SystemExit(2) from None
