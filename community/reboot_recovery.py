"""Host-local model service recovery under its persistent exclusive lock.

The host and boot identities come from the OS, never configuration or a PID.
Only a previous boot can authorize quarantining an unfinished service's files.
This journal must remain outside model-writable paths on host-local storage.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import plistlib
import secrets
import stat
import subprocess
import sys
import uuid
from pathlib import Path

from native_runtime import canonical, digest

LIMIT = 256 * 1024
LEGACY_SCHEMA = "umi-model-reboot-journal/1"
SCHEMA = "umi-model-reboot-journal/2"


def kernel_identity() -> dict[str, str]:
    if sys.platform == "darwin":
        boot = (
            subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                capture_output=True,
                check=True,
                timeout=5,
                env={},
            )
            .stdout.decode()
            .strip()
        )
        platform = subprocess.run(
            ["/usr/sbin/ioreg", "-a", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            check=True,
            timeout=5,
            env={},
        ).stdout
        host = plistlib.loads(platform)[0]["IOPlatformUUID"]
    elif sys.platform == "linux":
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        host = Path("/etc/machine-id").read_text().strip()
    else:
        raise RuntimeError("OS boot identity unavailable")
    result = {"platform": sys.platform, "host": uuid.UUID(host).hex, "boot": uuid.UUID(boot).hex}
    validate_identity(result)
    return result


def validate_identity(identity: dict) -> None:
    if not isinstance(identity, dict) or set(identity) != {"platform", "host", "boot"}:
        raise ValueError("invalid OS identity")
    if identity["platform"] not in {"darwin", "linux"}:
        raise ValueError("unsupported OS identity")
    for field in ("host", "boot"):
        value = identity[field]
        if not isinstance(value, str) or uuid.UUID(value).hex != value or int(value, 16) == 0:
            raise ValueError("invalid host or boot UUID")


def private_directory(path: Path) -> None:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("recovery directories must have no aliases")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("recovery directories must be owner-private")


def darwin_volume_uuid(path: Path) -> str:
    """Read the persistent volume ID from Darwin, without resolving symlinks.

    ATTR_VOL_UUID and the required ATTR_VOL_INFO come from Apple's sys/attr.h.
    Device numbers alone are not persistent across an APFS reboot.
    """

    class Attributes(ctypes.Structure):
        _fields_ = (
            ("bitmapcount", ctypes.c_uint16),
            ("reserved", ctypes.c_uint16),
            ("commonattr", ctypes.c_uint32),
            ("volattr", ctypes.c_uint32),
            ("dirattr", ctypes.c_uint32),
            ("fileattr", ctypes.c_uint32),
            ("forkattr", ctypes.c_uint32),
        )

    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    getter = library.getattrlist
    getter.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(Attributes),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
    ]
    getter.restype = ctypes.c_int
    attributes = Attributes(5, 0, 0, 0x80040000, 0, 0, 0)
    # Four-byte returned length followed by the sixteen-byte volume UUID.
    result = ctypes.create_string_buffer(20)
    if getter(os.fsencode(path), ctypes.byref(attributes), result, len(result), 1) != 0:
        raise OSError(ctypes.get_errno(), "persistent volume identity unavailable")
    if int.from_bytes(result.raw[:4], sys.byteorder) != len(result):
        raise ValueError("invalid volume identity response")
    identity = uuid.UUID(bytes=result.raw[4:20])
    if identity.int == 0:
        raise ValueError("persistent volume identity unavailable")
    return identity.hex


def validate_metadata(value: object, *, volume: bool) -> None:
    if not isinstance(value, list) or len(value) != 2 or type(value[1]) is not int or value[1] <= 0:
        raise ValueError("invalid retained artifact identity")
    if volume:
        if (
            not isinstance(value[0], str)
            or uuid.UUID(value[0]).hex != value[0]
            or uuid.UUID(value[0]).int == 0
        ):
            raise ValueError("invalid retained volume identity")
    elif type(value[0]) is not int or value[0] <= 0:
        raise ValueError("invalid retained device identity")


def metadata(path: Path, kind: str, *, volume: bool = False) -> list[int | str] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    permitted_type = {"directory": stat.S_ISDIR, "socket": stat.S_ISSOCK, "file": stat.S_ISREG}[
        kind
    ]
    expected_mode = 0o700 if kind == "directory" else 0o600
    if (
        not permitted_type(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != expected_mode
        or (kind != "directory" and info.st_nlink != 1)
    ):
        raise ValueError("retained model artifact is unsafe")
    if not volume:
        return [info.st_dev, info.st_ino]
    identity = darwin_volume_uuid(path)
    after = path.lstat()
    if any(
        getattr(info, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink")
    ):
        raise ValueError("retained model artifact changed during volume lookup")
    result = [identity, info.st_ino]
    validate_metadata(result, volume=True)
    return result


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RebootRecovery:
    def __init__(self, socket_path: Path, config_path: Path, config_sha256: str, document: dict):
        self.socket = socket_path
        self.path = Path(f"{socket_path}.reboot.json")
        self.config_sha256 = config_sha256
        self.identity = kernel_identity()
        validate_identity(self.identity)
        self.scratches = [Path(worker["scratch_directory"]) for worker in document["workers"]]
        for scratch in self.scratches:
            private_directory(scratch.parent)
            # Never move control files, the configuration, or another worker.
            if not scratch.is_absolute() or scratch.parent.resolve(strict=True) != scratch.parent:
                raise ValueError("scratch parent is aliased")
            if config_path.is_relative_to(scratch) or socket_path.is_relative_to(scratch):
                raise ValueError("control files must be outside worker scratch")
            if any(other != scratch and other.is_relative_to(scratch) for other in self.scratches):
                raise ValueError("scratch directories overlap")
        if len(set(self.scratches)) != len(self.scratches):
            raise ValueError("scratch directories repeat")
        self.state = self._read()

    def _read(self) -> dict | None:
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
                or not 0 < info.st_size <= LIMIT
            ):
                raise ValueError("unsafe reboot journal")
            raw = stream.read(LIMIT + 1)
            after = os.fstat(stream.fileno())
        linked = self.path.lstat()
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if len(raw) != info.st_size or any(
            getattr(info, field) != getattr(after, field)
            or getattr(after, field) != getattr(linked, field)
            for field in fields
        ):
            raise ValueError("reboot journal changed")
        state = json.loads(raw)
        if (
            canonical(state) != raw
            or not isinstance(state, dict)
            or set(state) != {"schema", "config_sha256", "identity", "phase", "scratches", "plan"}
            or state["schema"] not in {LEGACY_SCHEMA, SCHEMA}
            or state["phase"] not in {"active", "clean"}
            or not isinstance(state["scratches"], list)
            or not 1 <= len(state["scratches"]) <= 256
        ):
            raise ValueError("invalid reboot journal")
        validate_identity(state["identity"])
        digest(state["config_sha256"])
        for scratch in state["scratches"]:
            if (
                not isinstance(scratch, dict)
                or set(scratch) != {"path", "metadata"}
                or not isinstance(scratch["path"], str)
                or not Path(scratch["path"]).is_absolute()
            ):
                raise ValueError("invalid retained scratch identity")
            validate_metadata(
                scratch["metadata"],
                volume=state["schema"] == SCHEMA and state["identity"]["platform"] == "darwin",
            )
        return state

    def _metadata(self, path: Path, kind: str) -> list[int | str] | None:
        # Old journals keep their exact device check. Never invent a missing
        # historical volume identity; a clean restart can write the new schema.
        return metadata(
            path,
            kind,
            volume=self.state["schema"] == SCHEMA and self.identity["platform"] == "darwin",
        )

    def _write(self, state: dict) -> None:
        raw = canonical(state)
        if len(raw) > LIMIT:
            raise ValueError("reboot journal exceeds limit")
        temporary = self.path.parent / f".umi-journal-{secrets.token_hex(16)}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            sync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        self.state = state

    def _no_socket_artifacts(self) -> None:
        for path in (self.socket, Path(f"{self.socket}.capacity.json")):
            if path.exists() or path.is_symlink():
                raise FileExistsError("retained model socket requires recovery")

    def _clean_scratches(self) -> None:
        for scratch in self.state["scratches"]:
            path = Path(scratch["path"])
            private_directory(path)
            if self._metadata(path, "directory") != scratch["metadata"] or any(path.iterdir()):
                raise ValueError("retained scratch is not clean")

    def _entries(self) -> list[dict]:
        entries = [
            (self.socket, "socket", None),
            (Path(f"{self.socket}.capacity.json"), "file", None),
        ]
        entries += [(Path(s["path"]), "directory", s["metadata"]) for s in self.state["scratches"]]
        plan = []
        for source, kind, retained in entries:
            current = self._metadata(source, kind)
            if kind == "directory" and current != retained:
                raise ValueError("retained scratch identity changed")
            token = hashlib.sha256(
                (self.state["identity"]["boot"] + str(source)).encode()
            ).hexdigest()
            target = source.parent / f".umi-reboot-{token}"
            if target.exists() or target.is_symlink():
                raise FileExistsError("quarantine target already exists")
            plan.append(
                {"source": str(source), "kind": kind, "metadata": current, "target": str(target)}
            )
        return plan

    def _apply(self, entry: dict, *, execute: bool) -> None:
        source, target, kind = Path(entry["source"]), Path(entry["target"]), entry["kind"]
        old = entry["metadata"]
        present, quarantined = self._metadata(source, kind), self._metadata(target, kind)
        if old is None:
            if present is not None or quarantined is not None:
                raise ValueError("unexpected model artifact during recovery")
            return
        if present == old and quarantined is None:
            if not execute:
                return
            os.rename(source, target)
            present, quarantined = None, old
        if quarantined != old:
            raise ValueError("recovery artifact identity differs")
        if execute:
            # A previous attempt may have renamed the artifact but failed to
            # sync its parent. Reopening the plan must finish that barrier.
            sync_directory(source.parent)
        if kind == "directory":
            if present is not None and (present == old or any(source.iterdir())):
                raise ValueError("replacement scratch is occupied")
            if present is None and execute:
                source.mkdir(mode=0o700)
            if execute:
                # This also covers a mkdir that succeeded before interruption.
                sync_directory(source)
                sync_directory(source.parent)
        elif present is not None:
            raise ValueError("model socket reappeared during recovery")

    def prepare(self) -> None:
        state = self.state
        if state is None:
            self._no_socket_artifacts()
            return
        if any(state["identity"][key] != self.identity[key] for key in ("platform", "host")):
            raise ValueError("reboot journal belongs to another host")
        if state["phase"] == "clean":
            if state["plan"] is not None:
                raise ValueError("clean journal contains pending recovery")
            self._no_socket_artifacts()
            self._clean_scratches()
            return
        if state["identity"]["boot"] == self.identity["boot"]:
            raise RuntimeError("unfinished service in the current boot requires review")
        if state["config_sha256"] != self.config_sha256 or [
            s["path"] for s in state["scratches"]
        ] != [str(p) for p in self.scratches]:
            raise ValueError("unfinished deployment configuration changed")
        if state["plan"] is None:
            self._write({**state, "plan": self._entries()})
        plan = self.state["plan"]
        # Do not trust stored paths independently of the reviewed configuration.
        expected = [(str(self.socket), "socket"), (f"{self.socket}.capacity.json", "file")]
        expected += [(str(p), "directory") for p in self.scratches]
        if not isinstance(plan, list) or len(plan) != len(expected):
            raise ValueError("invalid recovery plan")
        for index, (entry, (source, kind)) in enumerate(zip(plan, expected, strict=True)):
            token = hashlib.sha256((state["identity"]["boot"] + source).encode()).hexdigest()
            target = str(Path(source).parent / f".umi-reboot-{token}")
            if (
                not isinstance(entry, dict)
                or set(entry) != {"source", "target", "kind", "metadata"}
                or entry["source"] != source
                or entry["kind"] != kind
                or entry["target"] != target
            ):
                raise ValueError("recovery plan binding differs")
            if entry["metadata"] is not None:
                validate_metadata(
                    entry["metadata"],
                    volume=state["schema"] == SCHEMA and self.identity["platform"] == "darwin",
                )
            if (
                kind == "directory"
                and entry["metadata"] != state["scratches"][index - 2]["metadata"]
            ):
                raise ValueError("recovery scratch binding differs")
            self._apply(entry, execute=False)
        for entry in plan:
            self._apply(entry, execute=True)

    def begin(self) -> None:
        self._no_socket_artifacts()
        scratches = []
        for path in self.scratches:
            private_directory(path)
            if any(path.iterdir()):
                raise ValueError("worker scratch must start empty")
            scratches.append(
                {
                    "path": str(path),
                    "metadata": metadata(
                        path, "directory", volume=self.identity["platform"] == "darwin"
                    ),
                }
            )
        self._write(
            {
                "schema": SCHEMA,
                "config_sha256": self.config_sha256,
                "identity": self.identity,
                "phase": "active",
                "scratches": scratches,
                "plan": None,
            }
        )

    def finish(self) -> None:
        # Called only after the sidecar's graceful shutdown returned successfully.
        self._no_socket_artifacts()
        self._clean_scratches()
        self._write({**self.state, "phase": "clean"})
