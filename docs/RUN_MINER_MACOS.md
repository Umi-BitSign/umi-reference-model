# Run the UMI S1 reference miner on Apple Silicon macOS

This procedure runs the public S1 bootstrap on an Apple Silicon Mac. PyTorch runs
natively on the host through MPS or CPU. MediaPipe extraction remains inside the
pinned Linux/AMD64 Docker worker. This profile does not use the older ARM64 Docker
worker and makes no native ARM64 extractor equivalence claim.

The current model is a low-accuracy no-weight bootstrap. A Mac deployment may join
inactive public calibration after it passes the signed UMI release checks and the
load test below. It is not UMI activation evidence or a capacity guarantee.

## 1. Check the host

Use an Apple Silicon Mac with macOS 14 or newer, Python 3.12, Docker Desktop, `uv`,
Git, `jq`, `curl`, and `shasum`. Docker Desktop must be running in the same login
session as the miner. Configure macOS and Docker Desktop to restart after reboot,
and prevent automatic system sleep while the miner is announced.

Install missing command-line dependencies with Homebrew:

```bash
brew install python@3.12 uv jq git
```

Docker Desktop is installed separately. Open a Bash shell and run the remaining
command blocks in that same shell so their fail-fast setting and variables remain
in effect. First, check the tools and host:

```bash
set -euo pipefail
umask 077
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
python3.12 -c 'import platform; assert platform.machine() == "arm64"'
test "$(uv --version | awk '{print $2}')" = 0.12.9
DOCKER_EXECUTABLE="$(command -v docker)"
test -x "$DOCKER_EXECUTABLE"
docker version --format 'client={{.Client.Version}};server={{.Server.Version}}'
docker info >/dev/null
```

Stop if Homebrew installs a `uv` version other than 0.12.9. Do not run the miner
from a Rosetta shell.

## 2. Check out trusted source and install the locked environment

The sealed model payload remains release `umi-s1-public-finetune-v1-r2`. The macOS
host support lives in a later source revision and does not change any file under
`release/`. Obtain `MAC_SUPPORT_GIT_REVISION` from the trusted release announcement.
The signed inactive UMI release supplies its own exact UMI source identity and must
include miner finality assets for `aarch64-apple-darwin`.

```bash
export MODEL_RELEASE_TAG=umi-s1-public-finetune-v1-r2
export MODEL_RELEASE_GIT_REVISION=20307ea05684e098ab362fa6bfc174c2aced3b9e
export MAC_SUPPORT_GIT_REVISION=40_LOWERCASE_HEX_FROM_TRUSTED_ANNOUNCEMENT
export UMI_INACTIVE_RELEASE=/absolute/path/to/public-inactive-release
export UMI_RELEASE_MANIFEST_SHA256=64_LOWERCASE_HEX_FROM_TRUSTED_ANNOUNCEMENT
export UMI_RELEASE_AUTHORITY=EXPECTED_RELEASE_AUTHORITY_SS58
export TRUSTED_UMI_RELEASE_RESOLVER=/absolute/path/to/trusted/umi-shadow-release-resolve-miner
export RESOLVED_UMI_PARENT=/absolute/private/umi-resolved-releases
export RESOLVED_UMI_DIRECTORY="$RESOLVED_UMI_PARENT/release-001"

mkdir -p "$HOME/umi-miner"
cd "$HOME/umi-miner"
git clone https://github.com/Umi-BitSign/umi-reference-model.git
git clone https://github.com/Umi-BitSign/umi.git

cd umi-reference-model
git verify-tag "$MODEL_RELEASE_TAG"
test "$(git rev-parse "$MODEL_RELEASE_TAG^{commit}")" = "$MODEL_RELEASE_GIT_REVISION"
git checkout --detach "$MAC_SUPPORT_GIT_REVISION"
test "$(git rev-parse HEAD)" = "$MAC_SUPPORT_GIT_REVISION"
git diff --exit-code "$MODEL_RELEASE_TAG" -- release

test -x "$TRUSTED_UMI_RELEASE_RESOLVER"
test "$(shasum -a 256 "$UMI_INACTIVE_RELEASE/release-manifest.json" | awk '{print $1}')" = \
  "$UMI_RELEASE_MANIFEST_SHA256"
install -d -m 700 "$RESOLVED_UMI_PARENT"
test ! -e "$RESOLVED_UMI_DIRECTORY"
"$TRUSTED_UMI_RELEASE_RESOLVER" "$UMI_INACTIVE_RELEASE" \
  --expected-authority-hotkey "$UMI_RELEASE_AUTHORITY" \
  --target-triple aarch64-apple-darwin \
  --output-dir "$RESOLVED_UMI_DIRECTORY"
export RESOLVED_UMI_MINER="$RESOLVED_UMI_DIRECTORY/resolved-miner-release.json"
jq -e \
  '.schema == "umi-resolved-miner-release/1" and
   .target_triple == "aarch64-apple-darwin" and
   .validator_runtime_supported == false' \
  "$RESOLVED_UMI_MINER"
UMI_GIT_REVISION="$(jq -er .umi_git_revision "$RESOLVED_UMI_MINER")"
UMI_SOURCE_TREE_SHA256="$(jq -er .umi_source_tree_sha256 "$RESOLVED_UMI_MINER")"
UMI_PYTHON_WHEEL="$(jq -er .python_wheel "$RESOLVED_UMI_MINER")"
UMI_SCORING_POLICY="$(jq -er .policy_path "$RESOLVED_UMI_MINER")"
UMI_SCORING_POLICY_SHA256="$(jq -er .scoring_policy_sha256 "$RESOLVED_UMI_MINER")"
test "$(jq -er .umi_revision "$RESOLVED_UMI_MINER")" = \
  "git:$UMI_GIT_REVISION;source-tree-sha256:$UMI_SOURCE_TREE_SHA256"
test "$(shasum -a 256 "$UMI_SCORING_POLICY" | awk '{print $1}')" = \
  "$UMI_SCORING_POLICY_SHA256"
git -C ../umi checkout --detach "$UMI_GIT_REVISION"
test "$(git -C ../umi rev-parse HEAD)" = "$UMI_GIT_REVISION"
test -z "$(git status --short)"
test -z "$(git -C ../umi status --short)"

install -d -m 700 "$HOME/umi-miner/deployment-tmp"
uv export --frozen --extra dev --no-dev --no-emit-project \
  --output-file "$HOME/umi-miner/deployment-tmp/model-requirements.txt"
(
  cd ../umi
  uv export --frozen --no-dev --no-emit-project \
    --output-file "$HOME/umi-miner/deployment-tmp/umi-requirements.txt"
)
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python \
  --require-hashes \
  --requirement "$HOME/umi-miner/deployment-tmp/model-requirements.txt" \
  --requirement "$HOME/umi-miner/deployment-tmp/umi-requirements.txt"
uv pip install --python .venv/bin/python --no-deps "$UMI_PYTHON_WHEEL"
PURELIB="$(.venv/bin/python -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$HOME/umi-miner/umi-reference-model/src" > \
  "$PURELIB/umi-reference-model-source.pth"
chmod 600 "$PURELIB/umi-reference-model-source.pth"
unset PYTHONPATH
.venv/bin/python - "$HOME/umi-miner/umi" "$UMI_SOURCE_TREE_SHA256" <<'PY'
import sys
from pathlib import Path
from umi.policy import umi_source_tree_sha256

checkout = (Path(sys.argv[1]) / "src").resolve(strict=True)
import umi
origin = Path(umi.__file__).resolve(strict=True)
assert not origin.is_relative_to(checkout), (origin, checkout)
assert umi_source_tree_sha256() == sys.argv[2]
PY
.venv/bin/python -c \
  'import platform, torch; assert platform.machine() == "arm64"; print(torch.__version__)'
```

Keep both lockfiles unchanged. Stop if the sealed release directory differs from
the r2 tag, the signed UMI release fails verification, or either checkout is dirty.

## 3. Verify, build, and bind the extractor

Verify the unchanged r2 artifact set from its sealed commit:

```bash
cd "$HOME/umi-miner/umi-reference-model"
(
  cd release
  shasum -a 256 -c SHA256SUMS
)
BASE_MODEL_SHA256="$(jq -er \
  '.artifacts[] | select(.label == "model") | .sha256' \
  release/release-manifest.json)"
BASE_MODEL_FILENAME="$(jq -er \
  '.artifacts[] | select(.label == "model") | .filename' \
  release/release-manifest.json)"
BASE_MODEL_PATH="$PWD/release/$BASE_MODEL_FILENAME"
BASE_INFERENCE_REVISION="$(jq -er .inference_revision \
  release/release-manifest.json)"
.venv/bin/python tools/release_artifacts.py \
  --verify release/release-manifest.json \
  --artifact-directory release \
  --repository . \
  --release-git-revision "$MODEL_RELEASE_GIT_REVISION"
```

Build the checked-in Linux/AMD64 image through Docker Desktop. The build record
binds `host_platform: darwin/arm64`, `container_platform: linux/amd64`, the immutable
image ID, exact packages, and checked-in source digests.

```bash
install -d -m 700 "$HOME/umi-miner/local-build"
EXTRACTOR_BUILD_RECORD="$HOME/umi-miner/local-build/local-extractor.json"
test ! -e "$EXTRACTOR_BUILD_RECORD"
DOCKER_EXECUTABLE="$(command -v docker)"
.venv/bin/python -m bitsign_motion.local_extractor_release \
  --docker "$DOCKER_EXECUTABLE" \
  --output "$EXTRACTOR_BUILD_RECORD"
jq -e \
  '.schema == "umi-local-extractor-build/2" and
   .host_platform == "darwin/arm64" and
   .container_platform == "linux/amd64"' \
  "$EXTRACTOR_BUILD_RECORD"
EXTRACTOR_IMAGE="$(jq -er .image_id "$EXTRACTOR_BUILD_RECORD")"
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' \
  "$EXTRACTOR_IMAGE")" = linux/amd64

.venv/bin/python -m bitsign_motion.local_bundle_rebind \
  --base-archive "$BASE_MODEL_PATH" \
  --base-sha256 "$BASE_MODEL_SHA256" \
  --base-inference-revision "$BASE_INFERENCE_REVISION" \
  --build-record "$EXTRACTOR_BUILD_RECORD" \
  --docker "$DOCKER_EXECUTABLE" \
  --output "$HOME/umi-miner/local-build/model-bundle"
MODEL_BUNDLE="$HOME/umi-miner/local-build/model-bundle"
MODEL_REVISION="$(jq -er .inference_revision \
  "$MODEL_BUNDLE/inference-identity.json")"
test "$MODEL_REVISION" != "$BASE_INFERENCE_REVISION"
```

The Docker worker uses AMD64 emulation on Apple Silicon. Do not change
`UMI_S1_EXTRACTOR_PLATFORM` to `linux/arm64`; that worker does not implement the
release miner's whole-video input contract.

## 4. Select MPS or CPU and probe

Download the MediaPipe task directly from Google and verify it:

```bash
install -d -m 700 "$HOME/umi-miner/runtime-assets"
MEDIAPIPE_TASK="$HOME/umi-miner/runtime-assets/holistic_landmarker.task"
curl --fail --location --proto '=https' --tlsv1.2 \
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task?generation=1703178474695092' \
  --output "$MEDIAPIPE_TASK"
test "$(shasum -a 256 "$MEDIAPIPE_TASK" | awk '{print $1}')" = \
  e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8
chmod 600 "$MEDIAPIPE_TASK"
```

Select MPS when the pinned PyTorch build reports it as available. CPU is the
supported fallback. MPS fast math and automatic CPU fallback stay disabled so an
operator can identify the device that executed the model.

```bash
UMI_S1_DEVICE="$(.venv/bin/python - <<'PY'
import torch
print("mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu")
PY
)"
test "$UMI_S1_DEVICE" = mps || test "$UMI_S1_DEVICE" = cpu
export UMI_S1_DEVICE
unset PYTORCH_ENABLE_MPS_FALLBACK PYTORCH_MPS_FAST_MATH PYTORCH_MPS_PREFER_METAL

export UMI_S1_BUNDLE="$MODEL_BUNDLE"
export UMI_S1_INFERENCE_REVISION="$MODEL_REVISION"
export UMI_S1_EXTRACTOR_IMAGE="$EXTRACTOR_IMAGE"
export UMI_S1_EXTRACTOR_MODEL="$MEDIAPIPE_TASK"
export UMI_S1_EXTRACTOR_PLATFORM=linux/amd64
export UMI_S1_DOCKER_EXECUTABLE="$DOCKER_EXECUTABLE"
export UMI_S1_TEMP_ROOT="$HOME/umi-miner/s1-jobs"
export UMI_S1_HARD_DEADLINE_SECONDS=150
install -d -m 700 "$UMI_S1_TEMP_ROOT"

PROBE="$(.venv/bin/python -m bitsign_motion.umi_reference_backend probe)"
printf '%s\n' "$PROBE" | jq -e \
  --arg revision "$MODEL_REVISION" \
  '.status == "ready" and
   .claim_status == "component_test_no_weight" and
   .inference_revision == $revision'
test -z "$(find "$UMI_S1_TEMP_ROOT" -mindepth 1 -print -quit)"
test -z "$(docker ps --all --filter name=bitsign-holistic- --format '{{.ID}}')"
```

## 5. Run the Apple Silicon request-to-reveal E2E

Before public announcement, run the complete E2E with one rights-cleared private
video. The test performs one valid and one invalid UMI request, verifies signed
timelock envelopes, opens them after reveal, and checks job and container cleanup.
It writes a private report outside the repository.

```bash
export BITSIGN_RUN_UMI_RELEASE_E2E=1
export BITSIGN_UMI_RELEASE_PROFILE=public-s1-finetune/1
export BITSIGN_UMI_DEPLOYMENT_PROFILE=macos-apple-silicon/1
export BITSIGN_UMI_REPOSITORY="$HOME/umi-miner/umi"
export BITSIGN_UMI_RESOLVED_MINER_RELEASE="$RESOLVED_UMI_MINER"
export BITSIGN_UMI_RELEASE_VIDEO=/absolute/private/path/to/eligible-test.mp4
export BITSIGN_UMI_RELEASE_VIDEO_RIGHTS_CLEARED=1
export BITSIGN_UMI_EXTRACTOR_BUILD_RECORD="$EXTRACTOR_BUILD_RECORD"
E2E_ROOT=/absolute/owner-only/e2e
install -d -m 700 "$E2E_ROOT"
export BITSIGN_UMI_RELEASE_E2E_REPORT="$E2E_ROOT/macos-private-run.json"
export UMI_S1_BASE_INFERENCE_REVISION="$BASE_INFERENCE_REVISION"
test ! -e "$BITSIGN_UMI_RELEASE_E2E_REPORT"

.venv/bin/python -m pytest -q tests/test_umi_release_e2e.py
test "$(jq -er .schema "$BITSIGN_UMI_RELEASE_E2E_REPORT")" = \
  umi-s1-macos-miner-e2e-run/1
test "$(jq -er .runtime.host_operating_system \
  "$BITSIGN_UMI_RELEASE_E2E_REPORT")" = Darwin
test "$(jq -er .runtime.host_architecture \
  "$BITSIGN_UMI_RELEASE_E2E_REPORT")" = arm64
test "$(jq -er .runtime.container_platform \
  "$BITSIGN_UMI_RELEASE_E2E_REPORT")" = linux/amd64
```

Project an aggregate public record when the private report will be shared as
deployment evidence:

```bash
PUBLIC_E2E="$E2E_ROOT/macos-public-e2e.json"
test ! -e "$PUBLIC_E2E"
.venv/bin/python -m bitsign_motion.s1_release_evidence release-e2e \
  --run-report "$BITSIGN_UMI_RELEASE_E2E_REPORT" \
  --output "$PUBLIC_E2E"
test "$(jq -er .schema "$PUBLIC_E2E")" = umi-s1-macos-miner-e2e/1
```

The public projection excludes the private video digest and response plaintext.
It is a deployment artifact, not an addition to the sealed r2 release inventory.
The E2E reserves four concurrent model slots only to exercise one request path per
launch validator. Four is not a production-capacity result.

## 6. Capacity gate and supervision

One observed M3 Ultra run on macOS 26.5.2 used PyTorch 2.7.0 MPS and the
Linux/AMD64 Docker extractor. A 14.1-second, 1280x720, 30 fps, silent H.264 clip
completed in about 30 seconds, returned 42 UTF-8 bytes, and passed cleanup. This is
one machine observation. At the launch minima, four validators can assign up to 112
clips to one selected miner in a window. The observed single-clip run does not prove
that this host can meet that burst or its response deadlines.

Three actual workers per validator would require ten 30-second waves to serve 28
clips and would consume a full 300 seconds before overhead. The first plausible
backend setting is therefore 16 actual workers. Set the UMI outer admission limit
to 64 so every validator can place 16 requests into the backend at once. Requests
for the same verified video in the same window share one shielded backend job;
successful hypotheses remain cached only for that active window. UMI still seals
and signs a distinct validator-bound response for every request.

The signed policy provides a 300-second issue interval followed by a 300-second
response interval. Each request is also bounded by its signed block deadline, and
the earlier boundary controls. The 150-second backend deadline and 180-second UMI
inference timeout are only local ceilings. A validator's signed 90-second transport
timeout starts before it waits for a miner slot. A retry may join the shielded job
started by the first attempt, but production readiness requires every initial
response in the burst to finish within the shortest signed transport timeout. Do
not use the local ceilings to extend a signed protocol boundary.

Before announcing the Mac as a production miner, run the opt-in test in
`tests/test_umi_macos_capacity.py`. It requires exactly 28 distinct rights-cleared
eligible clip paths and the resolved Darwin miner release. That resolved document
binds the minimum concurrency and transport timeout across all four signed validator
templates. The test sends 112 real `btauth/1`
requests through the UMI ASGI miner, requires 112 valid sealed envelopes and exactly
28 backend jobs, enforces the shortest template transport timeout, verifies job and
container cleanup, and writes aggregate evidence without video bytes or plaintext.
It is an in-process ASGI inference-burst gate, not a public-network, TLS, or
chain-finality rehearsal.

Create a canonical input document with this exact shape. Every path is absolute,
and the clips array contains 28 entries. `outer_inference_concurrency` must be a
positive multiple of four.

```json
{"clips":[{"eligibility_attested":true,"path":"/absolute/private/clips/01.mp4","sha256":"64_lowercase_hex","size_bytes":123456}],"expected_actual_workers":16,"outer_inference_concurrency":64,"resolved_miner_release":"/absolute/private/umi-resolved-releases/release-001/resolved-miner-release.json","rights_cleared_for_private_testing":true,"schema":"umi-s1-macos-capacity-input/1"}
```

The following command builds the complete RFC 8785 input from a private directory
containing only the 28 test MP4 files. The two attestation variables require an
explicit operator decision.

```bash
export CAPACITY_CLIP_DIRECTORY=/absolute/private/capacity/clips
export BITSIGN_UMI_MACOS_CAPACITY_INPUT=/absolute/private/capacity/input.json
export CAPACITY_CLIPS_RIGHTS_CLEARED=yes
export CAPACITY_CLIPS_ELIGIBLE=yes
test ! -e "$BITSIGN_UMI_MACOS_CAPACITY_INPUT"
.venv/bin/python - \
  "$CAPACITY_CLIP_DIRECTORY" \
  "$RESOLVED_UMI_MINER" \
  "$BITSIGN_UMI_MACOS_CAPACITY_INPUT" <<'PY'
import hashlib
import os
import sys
from pathlib import Path
from bitsign_motion.canonical import canonical_json_bytes

assert os.environ["CAPACITY_CLIPS_RIGHTS_CLEARED"] == "yes"
assert os.environ["CAPACITY_CLIPS_ELIGIBLE"] == "yes"
root = Path(sys.argv[1])
resolved = Path(sys.argv[2])
output = Path(sys.argv[3])
assert all(path.is_absolute() for path in (root, resolved, output))
paths = sorted(root.glob("*.mp4"))
assert len(paths) == 28
clips = []
for path in paths:
    assert path == path.resolve(strict=True) and path.is_file()
    payload = path.read_bytes()
    assert 1 <= len(payload) <= 16 * 1024 * 1024
    clips.append({
        "eligibility_attested": True,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    })
document = {
    "clips": clips,
    "expected_actual_workers": 16,
    "outer_inference_concurrency": 64,
    "resolved_miner_release": str(resolved),
    "rights_cleared_for_private_testing": True,
    "schema": "umi-s1-macos-capacity-input/1",
}
descriptor = os.open(
    output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o600,
)
try:
    payload = canonical_json_bytes(document)
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        assert written > 0
        offset += written
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
```

Then run the burst:

```bash
export BITSIGN_RUN_UMI_MACOS_CAPACITY=1
export BITSIGN_UMI_MACOS_CAPACITY_REPORT=/absolute/private/capacity/run.json
test ! -e "$BITSIGN_UMI_MACOS_CAPACITY_REPORT"
.venv/bin/python -m pytest -q tests/test_umi_macos_capacity.py
test "$(jq -er .schema "$BITSIGN_UMI_MACOS_CAPACITY_REPORT")" = \
  umi-s1-macos-capacity-run/1
test "$(jq -er .status "$BITSIGN_UMI_MACOS_CAPACITY_REPORT")" = passed
test "$(jq -er .execution.request_count "$BITSIGN_UMI_MACOS_CAPACITY_REPORT")" = 112
test "$(jq -er .execution.worker_counters.translation_jobs_started \
  "$BITSIGN_UMI_MACOS_CAPACITY_REPORT")" = 28
```

A skipped test is not a pass. Capture host and Docker peak-memory measurements
alongside the generated report. A failed run keeps the Mac in component-test use.

From the sibling `umi` checkout at the exact `UMI_GIT_REVISION` in the signed
inactive release, follow the sibling file
`../umi/docs/MACOS_MINER_OPERATOR.md`. It resolves the signed Darwin miner template
with `umi-shadow-release-resolve-miner`, starts the foreground process, and supplies
the requirements for supervising it with `launchd`. Its miner command must enable
UMI's explicit sharing layer with `--coalesce-window-video-inference` and set
`--max-backend-workers 16` alongside `--max-inference-concurrency 64`. Those values
remain provisional until the capacity gate above passes on the deployed host. The
service must:

- run as the dedicated login user, never as root;
- use the absolute Python, policy, finality binary, chain-spec, state, and wallet
  paths verified above;
- load the model environment from an owner-only file;
- wrap the foreground miner in `/usr/bin/caffeinate -dimsu`;
- write bounded stdout and stderr logs outside the repository;
- restart after failure with a delay;
- depend operationally on Docker Desktop being ready; and
- stop public ingress if the local `/healthz` check fails.

Complete registration, signed-policy configuration, TLS ingress, `serve-axon`, and
remote discovery with that UMI procedure. Do not adapt commands from a Linux release
whose manifest lacks `aarch64-apple-darwin` finality assets. The current UMI live
validator target remains Linux; this Mac profile enables mining only.

## 7. Failure handling

A missing MPS device, model mismatch, Docker failure, timeout, invalid video, or
unsupported operation must yield an encrypted error response or fail startup. Do
not add a text fallback. Stop the miner and close public ingress if a generated job
directory or `bitsign-holistic-*` container survives cancellation.
