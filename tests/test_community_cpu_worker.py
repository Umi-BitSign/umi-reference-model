from __future__ import annotations

import asyncio
import errno
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker_python():
    # The supervisor stays on 3.12; CI also launches workers with real 3.10.
    return os.environ.get("UMI_TEST_WORKER_PYTHON", sys.executable)


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "community"))
    return importlib.import_module("cpu_runtime"), importlib.import_module("cpu_worker")


@pytest.fixture
def sealed(tmp_path, modules):
    runtime, _ = modules
    roots = {}
    for name in ("code", "environment", "python"):
        roots[name] = tmp_path / name
        roots[name].mkdir(mode=0o700)
        (roots[name] / "fixture.py").write_text("raise AssertionError('do not execute')\n")
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    model = bundle / "model"
    model.mkdir(mode=0o700)
    body = b"raise AssertionError('do not import')\n"
    (model / "umi_inference.py").write_bytes(body)
    (model / "umi_inference.py").chmod(0o400)
    manifest = runtime.canonical(
        {
            "schema": "umi-model-bundle/1",
            "profile": "offline_bundle/1",
            "parent_baseline_sha256": None,
            "license_id": "MIT",
            "files": [
                {
                    "path": "umi_inference.py",
                    "role": "inference",
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "size_bytes": len(body),
                }
            ],
        }
    )
    (bundle / "manifest.json").write_bytes(manifest)
    (bundle / "manifest.json").chmod(0o400)
    bundle_sha = hashlib.sha256(b"umi-open-competition-v1\0" + manifest).hexdigest()
    output = tmp_path / "runtime.json"
    revision = runtime.stage_manifest(roots, bundle=bundle, bundle_sha256=bundle_sha, output=output)
    return roots, bundle, bundle_sha, output, revision


def test_cpu_identity_binds_environment_and_execution(modules, sealed):
    runtime, _ = modules
    roots, _, bundle_sha, manifest, revision = sealed
    document = runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision)
    assert document["model_bundle_sha256"] == bundle_sha
    assert document["execution"]["device"] == "cpu"
    assert document["platform"] == "linux/x86_64"
    assert manifest.stat().st_mode & 0o777 == 0o400
    assert (
        hashlib.sha256(b"umi-community-native-runtime-v1\0" + manifest.read_bytes()).hexdigest()
        != revision
    )


@pytest.mark.parametrize("root", ["code", "environment", "python"])
def test_runtime_drift_refuses_readiness(modules, sealed, root):
    runtime, _ = modules
    roots, _, _, manifest, revision = sealed
    (roots[root] / "fixture.py").write_text("changed")
    with pytest.raises(ValueError, match="installed runtime differs"):
        runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision)


@pytest.mark.parametrize(
    "field,value",
    [
        ("platform", "macos/arm64"),
        ("python_abi", "cp312"),
        ("execution", {"device": "cuda"}),
        ("schema", "umi-community-native-runtime/1"),
    ],
)
def test_unsupported_profile_refused_even_with_matching_hash(modules, sealed, field, value):
    runtime, _ = modules
    roots, _, _, manifest, _ = sealed
    document = json.loads(manifest.read_bytes())
    document[field] = value
    raw = runtime.canonical(document)
    manifest.chmod(0o600)
    manifest.write_bytes(raw)
    with pytest.raises(ValueError, match="unsupported CPU runtime"):
        runtime.verify_runtime(
            roots,
            manifest=manifest,
            expected_revision=hashlib.sha256(runtime.DOMAIN + raw).hexdigest(),
        )


def test_wrong_manifest_and_bundle_identity_refused(modules, sealed, tmp_path):
    runtime, _ = modules
    roots, bundle, _, manifest, _ = sealed
    with pytest.raises(ValueError, match="differs from reviewed"):
        runtime.verify_runtime(roots, manifest=manifest, expected_revision="00" * 32)
    with pytest.raises(ValueError, match="caller-pinned identity"):
        runtime.stage_manifest(
            roots, bundle=bundle, bundle_sha256="00" * 32, output=tmp_path / "must-not-exist"
        )
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_clip_is_private_deadline_forwarded_and_removed(modules, tmp_path, failure):
    _, worker = modules
    scratch = tmp_path.resolve()
    deadline = time.time_ns() + 10**9

    class Model:
        def translate_path(self, path, *, deadline_unix_ns):
            assert path.parent == scratch
            assert path.read_bytes() == b"video"
            assert path.stat().st_mode & 0o777 == 0o600
            assert deadline_unix_ns == deadline
            if failure:
                raise TimeoutError("expired")
            return "hello"

    if failure:
        with pytest.raises(TimeoutError):
            worker.translate_video(Model(), scratch, b"video", deadline)
    else:
        assert worker.translate_video(Model(), scratch, b"video", deadline) == "hello"
    assert list(scratch.iterdir()) == []


def test_cpu_loader_does_not_import_native_overlay_or_select_gpu(modules, tmp_path, monkeypatch):
    runtime, worker = modules
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            set_num_threads=lambda count: calls.append(("threads", count)),
            set_num_interop_threads=lambda count: calls.append(("interop", count)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "runtime",
        types.SimpleNamespace(
            SHuBERTInferenceRuntime=lambda root, **kwargs: (root, kwargs),
        ),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The loader's environment changes are confined to this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    root, arguments = worker.load_model(tmp_path / "bundle", tmp_path / "scratch")
    assert root == tmp_path / "bundle/model"
    assert arguments == {k: v for k, v in runtime.EXECUTION.items() if k != "cpu_threads"} | {
        "verify_assets": True,
    }
    assert calls == [("threads", 4), ("interop", 1)]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TMPDIR"] == str(tmp_path / "scratch")


def test_directory_aliases_overlap_and_public_scratch_refused(modules, tmp_path):
    _, worker = modules
    root = tmp_path.resolve()
    code, scratch = root / "code", root / "scratch"
    code.mkdir(mode=0o700)
    scratch.mkdir(mode=0o700)
    assert worker.checked_directories({"code": code, "scratch": scratch})["scratch"] == scratch
    with pytest.raises(ValueError, match="non-overlapping"):
        worker.checked_directories({"code": root, "scratch": scratch})
    link = root / "link"
    link.symlink_to(code)
    with pytest.raises(ValueError, match="aliases"):
        worker.checked_directories({"code": link, "scratch": scratch})
    scratch.chmod(0o755)
    with pytest.raises(ValueError, match="owner-private"):
        worker.checked_directories({"code": code, "scratch": scratch})


@pytest.mark.parametrize(
    "network_errno,accepted",
    [
        (errno.ENETUNREACH, True),
        (errno.EPERM, True),
        (errno.EACCES, True),
        (errno.ETIMEDOUT, False),
        (errno.ECONNREFUSED, False),
        (None, False),
    ],
)
def test_sandbox_network_failures_are_not_all_treated_as_isolation(
    modules, tmp_path, monkeypatch, network_errno, accepted
):
    _, worker = modules

    def deny_chmod(*args):
        raise OSError(errno.EROFS, "read-only")

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def connect(self, target):
            if network_errno is not None:
                raise OSError(network_errno, "fixture")

    monkeypatch.setattr(worker.os, "chmod", deny_chmod)
    monkeypatch.setattr(worker.socket, "socket", Connection)
    if accepted:
        worker.sandbox_probes([tmp_path], tmp_path / "hidden-marker")
    else:
        with pytest.raises(RuntimeError, match="sandbox"):
            worker.sandbox_probes([tmp_path], tmp_path / "hidden-marker")


def test_worker_refuses_writable_runtime_and_visible_marker(modules, tmp_path, monkeypatch):
    _, worker = modules
    marker = tmp_path / "marker"
    marker.write_text("harmless")
    with pytest.raises(RuntimeError, match="deny writes"):
        worker.sandbox_probes([tmp_path], marker)

    def deny_chmod(*args):
        raise OSError(errno.EROFS, "read-only")

    monkeypatch.setattr(worker.os, "chmod", deny_chmod)
    with pytest.raises(RuntimeError, match="hide the out-of-scope"):
        worker.sandbox_probes([tmp_path], marker)


def test_actual_cpu_worker_loop_reuses_model_and_reaps_on_timeout(modules, tmp_path, worker_python):
    transport = importlib.import_module("worker_transport")
    script = tmp_path / "inert_cpu_loader.py"
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    script.write_text(
        "import os, sys, time\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'community')!r})\nimport cpu_worker\n"
        "class Model:\n"
        " def translate_path(self, path, *, deadline_unix_ns):\n"
        "  video = path.read_bytes()\n"
        "  if video == b'hang': time.sleep(30)\n"
        "  if video == b'fail': raise RuntimeError('inference failed')\n"
        "  os.write(1, b'native inference diagnostic\\n')\n"
        "  return video.decode()\n"
        "def load_model(bundle, scratch):\n"
        " os.write(1, b'native model loading diagnostic\\n')\n"
        " return Model()\n"
        "cpu_worker.load_model = load_model\n"
        f"cpu_worker.run_worker(Path('.'), Path({str(scratch)!r}), {'ab' * 32!r})\n"
    )

    async def run():
        backend = transport.WarmModelProcess(
            [worker_python, "-B", "-s", str(script)],
            environment={},
            cwd=tmp_path,
            model_revision="ab" * 32,
            startup_seconds=5,
            inference_seconds=0.4,
            scratch_directory=scratch.resolve(),
        )
        try:
            await backend.startup()
            pid = backend._process.pid
            assert await backend.translate(b"hello") == "hello"
            assert await backend.translate(b"again") == "again"
            assert backend._process.pid == pid
            with pytest.raises(TimeoutError):
                await backend.translate(b"hang")
            assert not list(scratch.iterdir())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert await backend.translate(b"recovered") == "recovered"
            with pytest.raises((EOFError, asyncio.IncompleteReadError)):
                await backend.translate(b"fail")
            assert not list(scratch.iterdir())
        finally:
            await backend.close()

    asyncio.run(run())


def test_loading_failure_emits_no_readiness_or_diagnostic_on_protocol(tmp_path, worker_python):
    script = tmp_path / "failed_loader.py"
    script.write_text(
        "import os, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'community')!r})\nimport cpu_worker\n"
        "def load_model(*args):\n"
        " os.write(1, b'native load diagnostic\\n')\n"
        " raise RuntimeError('loading failed')\n"
        "cpu_worker.load_model = load_model\n"
        f"cpu_worker.run_worker(Path('.'), Path('.'), {'ab' * 32!r})\n"
    )
    result = subprocess.run(
        [worker_python, "-B", "-s", str(script)], capture_output=True, check=False, timeout=5
    )
    assert result.returncode != 0
    assert result.stdout == b""
    assert b"native load diagnostic" in result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mount/network namespaces")
def test_linux_bubblewrap_sandbox_probes(modules, tmp_path, worker_python):
    import shutil

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is required by the Linux CPU CI job")
    marker = tmp_path / "outside-marker"
    marker.write_text("harmless host file")
    script = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT / 'community')!r}); "
        "from cpu_worker import sandbox_probes; "
        f"sandbox_probes([Path({str(ROOT / 'community')!r})], Path({str(marker)!r}))"
    )
    python_roots = json.loads(
        subprocess.check_output(
            [
                worker_python,
                "-B",
                "-s",
                "-c",
                "import json, sys; print(json.dumps([sys.base_prefix, sys.prefix]))",
            ],
            timeout=5,
        )
    )
    command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    # Only the reviewed interpreter, standard shared libraries and code are visible.
    for path in dict.fromkeys(
        [
            Path("/usr"),
            Path("/lib"),
            Path("/lib64"),
            *(Path(path) for path in python_roots),
            ROOT / "community",
        ]
    ):
        if path.exists():
            command += ["--ro-bind", str(path), str(path)]
    command += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/",
        "--",
        worker_python,
        "-B",
        "-s",
        "-c",
        script,
    ]
    result = subprocess.run(
        command, env={"PATH": "/usr/bin:/bin"}, capture_output=True, check=False, timeout=15
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_manifest_cannot_be_written_inside_an_inventoried_tree(modules, sealed):
    runtime, _ = modules
    roots, bundle, bundle_sha, _, _ = sealed
    for directory in [*roots.values(), bundle]:
        with pytest.raises(ValueError, match="outside inventoried"):
            runtime.stage_manifest(
                roots, bundle=bundle, bundle_sha256=bundle_sha, output=directory / "runtime.json"
            )
        assert not (directory / "runtime.json").exists()


def test_runtime_sealing_cli_in_worker_interpreter(modules, sealed, tmp_path, worker_python):
    runtime, _ = modules
    roots, bundle, bundle_sha, _, expected_revision = sealed
    manifest = tmp_path / "cli-runtime.json"
    result = subprocess.run(
        [
            worker_python,
            "-B",
            "-s",
            str(ROOT / "community/cpu_runtime.py"),
            "--code",
            str(roots["code"]),
            "--environment",
            str(roots["environment"]),
            "--python",
            str(roots["python"]),
            "--bundle",
            str(bundle),
            "--expected-bundle-sha256",
            bundle_sha,
            "--output",
            str(manifest),
        ],
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert json.loads(result.stdout)["model_revision"] == expected_revision
    assert (
        runtime.verify_runtime(roots, manifest=manifest, expected_revision=expected_revision)[
            "model_bundle_sha256"
        ]
        == bundle_sha
    )


def add_bundle_metadata(bundle, runtime, metadata):
    manifest = bundle / "manifest.json"
    document = json.loads(manifest.read_bytes())
    for name, content in metadata.items():
        raw = content if isinstance(content, bytes) else runtime.canonical(content)
        target = bundle / "model" / name
        target.write_bytes(raw)
        target.chmod(0o400)
        document["files"].append(
            {
                "path": name,
                "role": "dependency" if name == "runtime.py" else "config",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    document["files"].sort(key=lambda row: row["path"])
    manifest.chmod(0o600)
    raw = runtime.canonical(document)
    manifest.write_bytes(raw)
    manifest.chmod(0o400)
    return hashlib.sha256(b"umi-open-competition-v1\0" + raw).hexdigest()


def test_sealing_retained_rc5_declarations_preserves_bundle(modules, sealed, tmp_path):
    runtime, worker = modules
    roots, bundle, _, _, old_revision = sealed
    retained = ROOT / "tests/fixtures/community-public-derivative"
    verifier = importlib.import_module("bundle_verification")
    actual_manifest = (retained / "manifest.json").read_bytes()
    assert hashlib.sha256(verifier.DOMAIN + actual_manifest).hexdigest() == (
        "132d59d4692f32c29ff92b7ada5f392c69adc8a14302eae124ceb92f3f685224"
    )
    records = verifier._records(actual_manifest)
    metadata = {
        name: (retained / name).read_bytes() for name in ("MODEL_IDENTITY.json", "fs_router.json")
    }
    for name, raw in metadata.items():
        row = next(record for record in records if record["path"] == name)
        assert len(raw) == row["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    bundle_sha = add_bundle_metadata(bundle, runtime, metadata)
    before = {str(p): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    output = tmp_path / "rc5-runtime.json"
    revision = runtime.stage_manifest(roots, bundle=bundle, bundle_sha256=bundle_sha, output=output)
    assert revision != old_revision
    for _ in range(2):
        document = runtime.verify_runtime(roots, manifest=output, expected_revision=revision)
        assert document["schema"] == "umi-community-cpu-runtime/2"
        assert document["generation"] == {
            "num_beams": 5,
            "max_length": 2048,
            "length_penalty": 0.6,
            "no_repeat_ngram_size": 16,
        }
        assert (
            worker.declared_generation(bundle, bundle_sha256=bundle_sha) == document["generation"]
        )
    assert before == {str(p): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "settings",
    [
        {"length_penalty": True},
        {"length_penalty": float("inf")},
        {"no_repeat_ngram_size": -1},
        {"no_repeat_ngram_size": True},
        {"num_beams": 0},
        {"max_length": 0},
        {"num_beams": 5.0},
        {"do_sample": True},
    ],
)
def test_invalid_generation_cannot_authorize_runtime(modules, sealed, settings):
    runtime, _ = modules
    roots, _, _, manifest, _ = sealed
    document = json.loads(manifest.read_bytes())
    document["generation"].update(settings)
    # Non-finite JSON is rejected by canonical sealing too; exercise the reader.
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    manifest.chmod(0o600)
    manifest.write_bytes(raw)
    with pytest.raises(ValueError):
        runtime.verify_runtime(
            roots,
            manifest=manifest,
            expected_revision=hashlib.sha256(runtime.DOMAIN + raw).hexdigest(),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "MODEL_IDENTITY.json": {
                "generation": {"length_penalty": 0.6},
                "baseline_generation_overrides": {"length_penalty": 1.0},
            }
        },
        {
            "MODEL_IDENTITY.json": {"generation": {"no_repeat_ngram_size": 16}},
            "fs_router.json": {"baseline_generate_overrides": {"no_repeat_ngram_size": 8}},
        },
        {"MODEL_IDENTITY.json": b'{"generation":{"length_penalty":0.6,"length_penalty":1}}'},
        {"MODEL_IDENTITY.json": {"generation": {"temperature": 1}}},
        {"MODEL_IDENTITY.json": {"generation": None}},
    ],
)
def test_invalid_or_disagreeing_declarations_fail_before_seal(modules, sealed, tmp_path, metadata):
    runtime, _ = modules
    roots, bundle, _, _, _ = sealed
    bundle_sha = add_bundle_metadata(bundle, runtime, metadata)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        runtime.stage_manifest(roots, bundle=bundle, bundle_sha256=bundle_sha, output=output)
    assert not output.exists()


def test_generation_metadata_tamper_and_unsealed_runtime_fail(modules, sealed):
    runtime, worker = modules
    _, bundle, _, _, _ = sealed
    bundle_sha = add_bundle_metadata(
        bundle, runtime, {"MODEL_IDENTITY.json": {"generation": {"length_penalty": 0.6}}}
    )
    assert worker.declared_generation(bundle, bundle_sha256=bundle_sha)["length_penalty"] == 0.6
    path = bundle / "model/MODEL_IDENTITY.json"
    path.chmod(0o600)
    path.write_text('{"generation":{"length_penalty":.7}}')
    path.chmod(0o400)
    with pytest.raises(ValueError):
        worker.declared_generation(bundle, bundle_sha256=bundle_sha)


def test_version_one_manifest_remains_readable_without_rewriting(modules, sealed):
    runtime, worker = modules
    roots, bundle, bundle_sha, manifest, _ = sealed
    document = json.loads(manifest.read_bytes())
    document["schema"] = "umi-community-cpu-runtime/1"
    document["execution"] = runtime.EXECUTION
    del document["generation"]
    raw = runtime.canonical(document)
    manifest.chmod(0o600)
    manifest.write_bytes(raw)
    revision = hashlib.sha256(runtime.DOMAIN + raw).hexdigest()
    assert runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision) == document
    assert worker.declared_generation(bundle, bundle_sha256=bundle_sha) == worker.DEFAULT_GENERATION
    assert manifest.read_bytes() == raw


@pytest.mark.parametrize("beams,length", [(5, 2048), (3, 1024)])
def test_real_bundle_runtime_import_and_generate_kwargs(
    modules, sealed, tmp_path, monkeypatch, beams, length
):
    runtime, worker = modules
    roots, bundle, _, _, _ = sealed
    # An inert historical runtime implements the same narrow warm boundary:
    # its constructor predates length_penalty/no_repeat_ngram_size arguments.
    model_source = """from types import SimpleNamespace
class Translator:
    def generate(self, **kwargs):
        return kwargs
class SHuBERTInferenceRuntime:
    def __init__(self, root, *, device, generation_num_beams, generation_max_length,
                 dino_batch_size, model_execution_concurrency, verify_assets):
        assert device == "cpu" and verify_assets is True
        self._translator = Translator()
        self.beams, self.length = generation_num_beams, generation_max_length
    def translate_path(self, path, *, deadline_unix_ns):
        result = self._translator.generate(num_beams=self.beams, max_length=self.length,
                    video=path.read_bytes(), deadline=deadline_unix_ns,
                    early_stopping=True, length_penalty=1.0)
        return repr(result)
"""
    generation = {
        "num_beams": beams,
        "max_length": length,
        "length_penalty": 0.6,
        "no_repeat_ngram_size": 16,
    }
    bundle_sha = add_bundle_metadata(
        bundle,
        runtime,
        {"runtime.py": model_source.encode(), "MODEL_IDENTITY.json": {"generation": generation}},
    )
    manifest = tmp_path / "derivative-runtime.json"
    revision = runtime.stage_manifest(
        roots, bundle=bundle, bundle_sha256=bundle_sha, output=manifest
    )
    document = runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision)
    assert worker.verified_generation(bundle, document) == generation
    monkeypatch.delitem(sys.modules, "runtime", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None
        ),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    try:
        for _ in range(2):  # fresh loading after restart applies exactly the same settings
            model = worker.load_model(bundle, tmp_path, generation)
            import ast

            result = ast.literal_eval(worker.translate_video(model, tmp_path, b"video", 123))
            assert result == generation | {
                "video": b"video",
                "deadline": 123,
                "early_stopping": True,
            }
        assert not list(tmp_path.glob("request-*"))
    finally:
        sys.modules.pop("runtime", None)


def test_legacy_manifest_cannot_silently_ignore_declared_decoding(modules, sealed):
    runtime, worker = modules
    _, bundle, _, manifest, _ = sealed
    sha = add_bundle_metadata(
        bundle, runtime, {"MODEL_IDENTITY.json": {"generation": {"length_penalty": 0.6}}}
    )
    document = json.loads(manifest.read_bytes())
    document.update(schema="umi-community-cpu-runtime/1", model_bundle_sha256=sha)
    del document["generation"]
    with pytest.raises(ValueError, match="reseal before serving"):
        worker.verified_generation(bundle, document)
