from __future__ import annotations

import copy
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fetch_c5_broad_how2sign as fetch


class Remote(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.requests = 0
        self.bytes_downloaded = 0

    def read(self, size=-1):
        block = super().read(size)
        self.requests += 1
        self.bytes_downloaded += len(block)
        return block


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Actual tiny ZIP writes; reservation admission and video probes are synthetic."""
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(
        fetch, "identity", lambda pid, argv: {"pid": pid, "argv": argv, "synthetic": True}
    )
    monkeypatch.setattr(fetch, "now", lambda: "2026-10-19T05:00:00+00:00")
    (tmp_path / "tools").mkdir()
    for name in fetch.SOURCES:
        (tmp_path / "tools" / name).write_text("# synthetic frozen source\n")
    reservation = tmp_path / "runs/reservation/plan.json"
    reservation.parent.mkdir(parents=True)
    reservation.write_text('{"synthetic":true}')
    reservation.with_name("review.json").write_text("{}")
    buffer = io.BytesIO()
    clips = []
    source_info = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as z:
        for signer in (1, 2, 3, 5):
            for group in range(3):
                name = f"source{signer}_{group}-rgb_front"
                z.writestr("raw/" + name + ".mp4", (name * 4).encode())
                source_info.append((signer, name))
                clips.extend(
                    {
                        "family": "how2sign",
                        "metadata": {
                            "video_name": name,
                            "video_id": name,
                            "signer": signer,
                            "sentence_name": f"{name}-{i}",
                            "start_sec": float(i * 2),
                            "end_sec": float(i * 2 + 2),
                        },
                    }
                    for i in range(2)
                )
    payload = buffer.getvalue()
    sources = []
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        infos = z.infolist()
        for signer, name in source_info:
            info = z.getinfo("raw/" + name + ".mp4")
            sources.append(
                {
                    "video_name": name,
                    "video_id": name,
                    "signer": signer,
                    "archive_member": info.filename,
                    "crc32": info.CRC,
                    "file_size": info.file_size,
                    "compressed_size": info.compress_size,
                    "header_offset": info.header_offset,
                    "compression_method": info.compress_type,
                    "candidate_clips": [f"{name}-{i}" for i in range(2)],
                }
            )
        fingerprint = fetch.archive_fingerprint(infos)
    record = {
        "how2sign_primary_recordings": sources,
        "primary_records": clips,
        "planned_How2Sign_raw_source_bytes": sum(r["file_size"] for r in sources),
    }
    archive = {
        "file_id": "synthetic",
        "archive_name": "synthetic.zip",
        "archive_size_bytes": len(payload),
        "archive_fingerprint_sha256": fingerprint,
    }
    monkeypatch.setattr(fetch, "admit", lambda _: (copy.deepcopy(record), copy.deepcopy(archive)))
    calls = []

    def remote(**kwargs):
        calls.append(kwargs)
        return Remote(payload)

    monkeypatch.setattr(fetch, "How2SignRemoteZip", remote)
    monkeypatch.setattr(fetch, "probe_source", lambda path, clips: {"synthetic_probe": True})
    job = tmp_path / "runs/job/plan.json"
    data = tmp_path / "data/raw"
    fetch.freeze(job, reservation, data)
    return SimpleNamespace(
        root=tmp_path,
        job=job,
        data=data,
        reservation=reservation,
        payload=payload,
        sources=sources,
        record=record,
        archive=archive,
        calls=calls,
    )


def test_serial_zip_integrity_timestamps_and_terminal_receipts(fixture):
    f = fixture
    fetch.run(f.job)
    complete = fetch.read(f.job.with_name("complete.json"))
    assert complete["sources_verified"] == 12 and complete["selected_intervals_verified"] == 24
    assert complete["confirmation_consumed"] is False and complete["training_allowed"] is False
    assert complete["model_predictions_performed"] is False
    assert len(complete["source_receipts"]) == 12 and len(f.calls) == 1
    for index, row in enumerate(f.sources):
        receipt = fetch.read(f.job.with_name(f"source-{index:02d}.json"))
        path = f.root / receipt["video"]["path"]
        assert fetch.sha256(path) == receipt["video"]["sha256"]
        assert path.stat().st_size == row["file_size"]
        assert receipt["CRC_and_bytes_verified"] and receipt["timestamps_fit_actual_recording"]
        assert len(receipt["selected_clips"]) == 2
    assert fetch.read(f.job.with_name("exit.json"))["exit_code"] == 0
    assert not f.job.with_name("failure.json").exists()
    assert not list(f.data.rglob("*.partial.mp4"))
    with pytest.raises(ValueError, match="replay"):
        fetch.run(f.job)
    assert len(f.calls) == 1


def test_corrupt_member_retains_partial_and_cannot_publish_source_or_completion(
    fixture, monkeypatch
):
    f = fixture
    payload = bytearray(f.payload)
    row = f.sources[0]
    offset = row["header_offset"] + 30 + len(row["archive_member"].encode())
    payload[offset] ^= 1
    monkeypatch.setattr(fetch, "How2SignRemoteZip", lambda **kwargs: Remote(bytes(payload)))
    with pytest.raises(ValueError, match="CRC"):
        fetch.run(f.job)
    assert list(f.data.rglob("*.partial.mp4"))
    assert not f.job.with_name("source-00.json").exists()
    assert not f.job.with_name("complete.json").exists()
    assert fetch.read(f.job.with_name("failure.json"))["partials_retained"] is True
    assert fetch.read(f.job.with_name("exit.json"))["exit_code"] == 1


def test_failed_timestamp_probe_retains_crc_checked_bytes_without_admitting_them(
    fixture, monkeypatch
):
    f = fixture

    def bad(*args):
        raise ValueError("synthetic interval outside source")

    monkeypatch.setattr(fetch, "probe_source", bad)
    with pytest.raises(ValueError, match="interval outside"):
        fetch.run(f.job)
    assert list(f.data.rglob("*.partial.mp4")) and not f.job.with_name("source-00.json").exists()
    assert not f.job.with_name("complete.json").exists()


def test_changed_archive_directory_stops_before_source_download(fixture, monkeypatch):
    f = fixture
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("other", "bad")
    monkeypatch.setattr(fetch, "How2SignRemoteZip", lambda **kwargs: Remote(buf.getvalue()))
    with pytest.raises(ValueError, match="directory identity"):
        fetch.run(f.job)
    assert not list(f.data.rglob("*.mp4"))
    assert fetch.read(f.job.with_name("exit.json"))["exit_code"] == 1


def test_low_disk_refuses_before_worker_or_data_creation(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage reserve"):
        fetch.run(f.job)
    assert not f.job.with_name("worker.json").exists() and not f.data.exists()


def test_modified_frozen_source_stops_before_network_or_data(fixture):
    f = fixture
    (f.root / "tools" / fetch.SOURCES[0]).write_text("# altered\n")
    with pytest.raises(ValueError, match="plan/source changed"):
        fetch.run(f.job)
    assert not f.calls and not f.data.exists()


def test_actual_probe_parses_metadata_without_frame_decoding(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [{"width": 640, "height": 480, "duration": "N/A"}],
                    "format": {"duration": "5.0"},
                }
            )
        )

    monkeypatch.setattr(fetch.subprocess, "run", run)
    actual = fetch.probe_source(tmp_path / "video.mp4", [{"start_sec": 1, "end_sec": 4}])
    assert actual["format"]["duration"] == "5.0"
    assert calls[0][0] == "ffprobe" and calls[0][calls[0].index("-threads") + 1] == "1"
    assert "-show_frames" not in calls[0]
    with pytest.raises(ValueError, match="interval exceeds"):
        fetch.probe_source(tmp_path / "video.mp4", [{"start_sec": 1, "end_sec": 6}])


def test_missing_reservation_fails_before_network_or_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="owned reservation"):
        fetch.admit(tmp_path / "runs/missing/plan.json")
