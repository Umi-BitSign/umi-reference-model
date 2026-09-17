"""Stage a private CPU-only MediaPipe package without changing an environment.

Only a completed, reviewed ARM64 build is accepted. The original Python package
is copied without caches; its extension is replaced in that copy. All private
Mach-O dependencies are relocated next to the extension and ad-hoc signed.
This stages executable artifacts; it does not qualify model accuracy or rights.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

LIBRARIES = frozenset(
    f"libopencv_{name}.3.4.dylib"
    for name in (
        "calib3d",
        "core",
        "features2d",
        "highgui",
        "imgcodecs",
        "imgproc",
        "video",
        "videoio",
    )
)
EXTENSION = "_framework_bindings.cpython-310-darwin.so"
F_GETPATH = 50
RENAME_EXCL = 0x00000004
ROOT = Path(__file__).resolve().parent


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("native build input must be a lowercase SHA-256 digest")
    return value


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def build_inputs() -> dict:
    document = json.loads((ROOT / "inputs.json").read_bytes())
    native = document.get("unrelocated_native_sha256")
    package = document.get("installed_mediapipe_inventory")
    expected_names = {f"mediapipe/python/{EXTENSION}"} | {
        f"mediapipe/python/{name}" for name in LIBRARIES
    }
    if (
        not isinstance(document, dict)
        or document.get("schema") != "umi-community-native-build-inputs/1"
        or not isinstance(package, dict)
        or set(package)
        != {"algorithm", "schema", "sha256", "file_count", "content_bytes", "original_binding"}
        or package.get("algorithm") != "sha256-canonical-json-v1"
        or package.get("schema") != "umi-mediapipe-installed-package-inventory/1"
        or not isinstance(package.get("original_binding"), dict)
        or set(package["original_binding"]) != {"path", "sha256", "size_bytes"}
        or package["original_binding"].get("path") != f"python/{EXTENSION}"
        or document.get("patch") != "mediapipe-v0.10.14-macos.patch"
        or not isinstance(native, dict)
        or set(native) != expected_names
        or any(_digest(value) != value for value in native.values())
    ):
        raise ValueError("native build input inventory differs")
    for name in (
        "overlay_manifest_sha256",
        "patch_sha256",
        "relocated_binding_sha256",
        "unrelocated_binding_sha256",
    ):
        _digest(document.get(name))
    _digest(package.get("sha256"))
    _digest(package["original_binding"].get("sha256"))
    if native[f"mediapipe/python/{EXTENSION}"] != document["unrelocated_binding_sha256"]:
        raise ValueError("binding identities disagree")
    for name in (
        "overlay_content_bytes",
        "overlay_file_count",
    ):
        if type(document.get(name)) is not int or document[name] < 0:
            raise ValueError("native build count or size differs")
    for name in ("content_bytes", "file_count"):
        if type(package.get(name)) is not int or package[name] < 0:
            raise ValueError("installed package count or size differs")
    if (
        type(package["original_binding"].get("size_bytes")) is not int
        or package["original_binding"]["size_bytes"] < 0
    ):
        raise ValueError("installed binding size differs")
    return document


def _identity(info) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def canonical_directory(path: Path, *, label: str) -> tuple[Path, tuple[int, int]]:
    """Return an opened directory's exact Darwin path and stable identity."""
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute directory without aliases")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"{label} must be an absolute directory without aliases") from error
    try:
        metadata = os.fstat(descriptor)
        if platform.system() == "Darwin":
            raw = fcntl.fcntl(descriptor, F_GETPATH, b"\0" * 1024)
            actual = raw.split(b"\0", 1)[0]
            canonical = actual == os.fsencode(path)
        else:
            canonical = os.fsencode(path.resolve(strict=True)) == os.fsencode(path)
        linked = path.stat(follow_symlinks=False)
        if (
            not canonical
            or not stat.S_ISDIR(metadata.st_mode)
            or (
                metadata.st_dev,
                metadata.st_ino,
            )
            != (linked.st_dev, linked.st_ino)
        ):
            raise ValueError(f"{label} must be an absolute directory without aliases")
        return path, (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def file_hash(path: Path) -> tuple[str, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("native build input must be a regular file")
        checksum = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
        linked = path.lstat()
        if (
            size != before.st_size
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(linked)
        ):
            raise ValueError("native build input changed while hashing")
    return checksum.hexdigest(), size


def sha256(path: Path) -> str:
    return file_hash(path)[0]


def copy_file(source: Path, destination: Path, *, replace: bool = False) -> tuple[str, int]:
    """Copy and hash one opened source; never reopen it between review and use."""
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if replace else os.O_EXCL)
    try:
        destination_fd = os.open(destination, flags, 0o600)
    except BaseException:
        os.close(source_fd)
        raise
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("native build input must be a regular file")
        checksum = hashlib.sha256()
        size = 0
        while chunk := os.read(source_fd, 1024 * 1024):
            checksum.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        linked = source.lstat()
        if (
            size != before.st_size
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(linked)
        ):
            raise ValueError("native build input changed while copying")
        return checksum.hexdigest(), size
    finally:
        os.close(source_fd)
        os.close(destination_fd)


def snapshot_package(package: Path, destination: Path) -> dict:
    destination.mkdir(mode=0o700)
    records = []
    for source in sorted(package.rglob("*")):
        relative = source.relative_to(package)
        if "__pycache__" in relative.parts or source.name.endswith(".pyc"):
            continue
        metadata = source.lstat()
        target = destination / relative
        if stat.S_ISDIR(metadata.st_mode):
            target.mkdir(mode=0o700)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            checksum, size = copy_file(source, target)
            records.append({"path": relative.as_posix(), "sha256": checksum, "size_bytes": size})
        else:
            raise ValueError("installed package contains a link or special file")
    return {
        "sha256": hashlib.sha256(canonical(records)).hexdigest(),
        "file_count": len(records),
        "content_bytes": sum(record["size_bytes"] for record in records),
        "records": records,
    }


def installed_package_identity(package: Path) -> dict:
    records = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.name.endswith(".pyc"):
            continue
        metadata = path.lstat()
        if path.is_symlink() or not (
            stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
        ):
            raise ValueError("installed package contains a link or special file")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        checksum, size = file_hash(path)
        records.append({"path": relative.as_posix(), "sha256": checksum, "size_bytes": size})
    return {
        "sha256": hashlib.sha256(canonical(records)).hexdigest(),
        "file_count": len(records),
        "content_bytes": sum(record["size_bytes"] for record in records),
    }


def publish_exclusive(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameatx_np
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-2, os.fsencode(source), -2, os.fsencode(destination), RENAME_EXCL) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def run(*command: str) -> str:
    return subprocess.check_output(command, text=True, timeout=60).strip()


def dependencies(path: Path) -> tuple[str, list[str]]:
    identity = run("/usr/bin/otool", "-D", str(path)).splitlines()
    if len(identity) != 2:
        raise ValueError("expected one Mach-O dylib identity")
    records = run("/usr/bin/otool", "-L", str(path)).splitlines()[1:]
    names = []
    for record in records:
        match = re.fullmatch(r"\s*(\S+) \(compatibility version [^)]+\)", record)
        if match is None:
            raise ValueError("unexpected Mach-O dependency record")
        names.append(match[1])
    if not names or names[0] != identity[1]:
        raise ValueError("Mach-O identity differs from load commands")
    return identity[1], names[1:]


def private_dependencies(names: list[str], *, relocated: bool) -> list[str]:
    expected = "@loader_path/" if relocated else "@rpath/"
    result = []
    for name in names:
        if name.startswith(("/usr/lib/", "/System/Library/")):
            continue
        if not name.startswith(expected) or name.removeprefix(expected) not in LIBRARIES:
            raise ValueError(f"unexpected non-system native dependency: {name}")
        result.append(name)
    return result


def rpaths(path: Path) -> list[str]:
    commands = run("/usr/bin/otool", "-l", str(path))
    return re.findall(r"cmd LC_RPATH\n\s*cmdsize \d+\n\s*path (.+) \(offset \d+\)", commands)


def relocate(path: Path) -> None:
    if run("/usr/bin/lipo", "-archs", str(path)) != "arm64":
        raise ValueError("native overlay accepts only thin ARM64 artifacts")
    _, names = dependencies(path)
    changes = []
    for name in private_dependencies(names, relocated=False):
        changes.extend(("-change", name, "@loader_path/" + name.removeprefix("@rpath/")))
    for value in rpaths(path):
        changes.extend(("-delete_rpath", value))
    run("/usr/bin/install_name_tool", "-id", f"@loader_path/{path.name}", *changes, str(path))
    run("/usr/bin/codesign", "--force", "--sign", "-", "--timestamp=none", str(path))
    run("/usr/bin/codesign", "--verify", "--strict", str(path))
    identity, names = dependencies(path)
    if identity != f"@loader_path/{path.name}" or rpaths(path):
        raise ValueError("native relocation left an unexpected search path")
    for name in private_dependencies(names, relocated=True):
        sibling = path.parent / name.removeprefix("@loader_path/")
        if not sibling.is_file() or sibling.is_symlink():
            raise ValueError("relocated native dependency is missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-bin", type=Path, required=True)
    parser.add_argument("--installed-package", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("this packager runs on the ARM64 build host")
    build, build_identity = canonical_directory(args.build_bin, label="build root")
    package, package_identity = canonical_directory(
        args.installed_package, label="installed package"
    )
    if (
        build_identity == package_identity
        or build.is_relative_to(package)
        or package.is_relative_to(build)
    ):
        raise ValueError("native build inputs must be distinct and non-overlapping")
    reviewed = build_inputs()
    target = args.destination
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise ValueError("destination must be a new absolute path")
    parent, _ = canonical_directory(target.parent, label="destination parent")
    parent_info = parent.stat()
    if parent_info.st_uid != os.getuid() or stat.S_IMODE(parent_info.st_mode) & 0o077:
        raise ValueError("destination parent must be owner-private")
    if target.is_relative_to(build) or target.is_relative_to(package):
        raise ValueError("destination must not overlap native build inputs")
    if package.name != "mediapipe" or not (package / "python" / EXTENSION).is_file():
        raise ValueError("expected the Python 3.10 MediaPipe package")
    patch = ROOT / reviewed["patch"]
    if sha256(patch) != reviewed["patch_sha256"]:
        raise ValueError("native compatibility patch differs from reviewed input")
    source_binding = build / "mediapipe/python/_framework_bindings.so"
    inputs = {EXTENSION: source_binding}
    for name in sorted(LIBRARIES):
        inputs[name] = build / "third_party/opencv_cmake/lib" / name
    os.umask(0o077)
    temporary = Path(tempfile.mkdtemp(prefix=".native-overlay-pending-", dir=parent))
    try:
        stage = temporary / "overlay"
        stage.mkdir(mode=0o700)
        package_snapshot = snapshot_package(package, stage / "mediapipe")
        expected_package = reviewed["installed_mediapipe_inventory"]
        if {key: package_snapshot[key] for key in ("sha256", "file_count", "content_bytes")} != {
            key: expected_package[key] for key in ("sha256", "file_count", "content_bytes")
        }:
            raise ValueError("installed MediaPipe package differs from reviewed input")
        original = next(
            (
                record
                for record in package_snapshot["records"]
                if record["path"] == expected_package["original_binding"]["path"]
            ),
            None,
        )
        if original != {
            "path": expected_package["original_binding"]["path"],
            "sha256": expected_package["original_binding"]["sha256"],
            "size_bytes": expected_package["original_binding"]["size_bytes"],
        }:
            raise ValueError("installed MediaPipe binding differs from reviewed input")

        native = stage / "mediapipe/python"
        sources = {}
        for name, source in inputs.items():
            destination = native / name
            if destination.exists():
                if name != EXTENSION or not destination.is_file() or destination.is_symlink():
                    raise ValueError("unexpected file in overlay")
                destination.chmod(0o600)
            checksum, _ = copy_file(source, destination, replace=name == EXTENSION)
            key = f"mediapipe/python/{name}"
            if checksum != reviewed["unrelocated_native_sha256"][key]:
                raise ValueError("native build output differs from reviewed input")
            sources[key] = checksum
            private_dependencies(dependencies(destination)[1], relocated=False)
        for name in sorted(inputs):
            relocate(native / name)

        records = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                path.chmod(0o400)
                checksum, size = file_hash(path)
                records.append(
                    {
                        "path": path.relative_to(stage).as_posix(),
                        "sha256": checksum,
                        "size_bytes": size,
                    }
                )
        document = {
            "schema": "umi-mediapipe-cpu-overlay/1",
            "platform": "macos/arm64",
            "python_abi": "cp310",
            "mediapipe_version": "0.10.14",
            "files": records,
            "unrelocated_native_sha256": sources,
            "quality_evaluated": False,
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
        manifest_sha256 = hashlib.sha256(payload).hexdigest()
        binding_sha256 = sha256(native / EXTENSION)
        if (
            len(records) != reviewed["overlay_file_count"]
            or sum(record["size_bytes"] for record in records) != reviewed["overlay_content_bytes"]
            or manifest_sha256 != reviewed["overlay_manifest_sha256"]
            or binding_sha256 != reviewed["relocated_binding_sha256"]
        ):
            raise ValueError("staged overlay differs from the reviewed live artifact")
        with (stage / "overlay.json").open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        (stage / "overlay.json").chmod(0o400)
        for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
            directory.chmod(0o500)
        stage.chmod(0o500)
        if any(stat.S_IMODE(p.stat().st_mode) & 0o222 for p in stage.rglob("*")):
            raise ValueError("overlay is not read-only")
        publish_exclusive(stage, target)
    finally:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o700)
            else:
                path.chmod(0o600)
        shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "status": "staged",
                "destination": str(target),
                "files": len(records),
                "manifest_sha256": manifest_sha256,
                "binding_sha256": binding_sha256,
                "quality_evaluated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
