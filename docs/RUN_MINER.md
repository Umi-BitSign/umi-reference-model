# Run the UMI S1 reference miner

This profile serves a real model through the no-weight UMI component miner. UMI
translation weights are inactive, so registration and successful responses do not
guarantee mining rewards.

The release does not distribute a Docker image. Each operator builds the extractor
from source, validates the resulting local image, and derives a new model revision
that binds that image's immutable ID.

## Host requirements

Use a Linux/AMD64 host with:

- Python 3.12 and `uv`;
- Docker with permission to build and run Linux/AMD64 images;
- `curl`, `jq`, Git, unzip, and SHA-256 tooling;
- outbound HTTPS access to package sources, the challenge host, and drand Quicknet;
- at least 8 GiB of free RAM and enough disk for PyTorch and the local image.

CPU inference is supported. A miner may set `UMI_S1_DEVICE=cuda:0` when its pinned
PyTorch installation and host driver expose CUDA. The MediaPipe extractor runs in the
Linux/AMD64 container either way.

Docker socket access is usually equivalent to root access on the host. Run the miner
under a dedicated account and never mount wallet directories into the extractor
container.

## 1. Check out exact code

Clone this repository and the public UMI repository beside each other. Check out the
release tag for this repository and the exact UMI commit recorded in
`release/release-manifest.json`.

Run every block below in the same Bash session. The first command enables fail-fast
handling so a failed digest, revision, import, or cleanup check stops setup.

```bash
set -euo pipefail
mkdir -p "$HOME/umi-miner"
cd "$HOME/umi-miner"
git clone https://github.com/Umi-BitSign/umi-reference-model.git
git clone https://github.com/Umi-BitSign/umi.git
cd umi-reference-model
git checkout RELEASE_TAG
SOURCE_GIT_REVISION="$(jq -r .source_git_revision release/release-manifest.json)"
UMI_GIT_REVISION="$(jq -r .umi_git_revision release/release-manifest.json)"
RELEASE_GIT_REVISION="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD^)" = "$SOURCE_GIT_REVISION"
test "$(git rev-list --parents -n 1 HEAD | wc -w | tr -d ' ')" = 2
test "$(git diff --name-only --no-renames "$SOURCE_GIT_REVISION" HEAD | LC_ALL=C sort)" = \
  "$(printf '%s\n' release/SHA256SUMS release/release-manifest.json)"
git -C ../umi checkout --detach "$UMI_GIT_REVISION"
test "$(git -C ../umi rev-parse HEAD)" = "$UMI_GIT_REVISION"
test -z "$(git status --short)"
test -z "$(git -C ../umi status --short)"
```

Replace `RELEASE_TAG` with the published immutable tag. Stop if either checkout is
dirty or a revision check fails.

## 2. Install the locked environment

Export both frozen lockfiles, then install them into one environment. Add two fixed,
path-only `.pth` files for the verified source trees; do not perform editable installs
or invoke either build backend during deployment.

```bash
cd "$HOME/umi-miner/umi-reference-model"
install -d -m 700 deployment-tmp
uv export --frozen --no-dev --no-emit-project \
  --output-file deployment-tmp/model-requirements.txt
(
  cd ../umi
  uv export --frozen --no-dev --no-emit-project \
    --output-file ../umi-reference-model/deployment-tmp/umi-requirements.txt
)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --require-hashes \
  --requirement deployment-tmp/model-requirements.txt \
  --requirement deployment-tmp/umi-requirements.txt
PURELIB="$(.venv/bin/python -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$HOME/umi-miner/umi-reference-model/src" > \
  "$PURELIB/umi-reference-model-source.pth"
printf '%s\n' "$HOME/umi-miner/umi/src" > "$PURELIB/umi-source.pth"
chmod 600 "$PURELIB/umi-reference-model-source.pth" "$PURELIB/umi-source.pth"
unset PYTHONPATH
.venv/bin/python -c 'import bitsign_motion, umi'
```

Keep `uv.lock` from both repositories unchanged. A dependency conflict or lock drift
blocks startup. The portable identity pins Python 3.12, NumPy, PyTorch, Safetensors,
RFC 8785, and every bound source module.

## 3. Download the base model

Download `umi-s1-baseline-v0-portable.zip`, `SHA256SUMS`, and
`release-manifest.json` from the same GitHub release into an owner-only directory.

```bash
install -d -m 700 "$HOME/umi-miner/release-assets"
cd "$HOME/umi-miner/release-assets"
shasum -a 256 -c SHA256SUMS
BASE_MODEL_SHA256="$(
  jq -er '.artifacts[] | select(.label == "model") | .sha256' release-manifest.json
)"
test "$(shasum -a 256 umi-s1-baseline-v0-portable.zip | cut -d ' ' -f 1)" = \
  "$BASE_MODEL_SHA256"
cmp release-manifest.json \
  "$HOME/umi-miner/umi-reference-model/release/release-manifest.json"
cmp SHA256SUMS "$HOME/umi-miner/umi-reference-model/release/SHA256SUMS"
"$HOME/umi-miner/umi-reference-model/.venv/bin/python" \
  "$HOME/umi-miner/umi-reference-model/tools/release_artifacts.py" \
  --verify "$HOME/umi-miner/umi-reference-model/release/release-manifest.json" \
  --artifact-directory "$HOME/umi-miner/release-assets" \
  --repository "$HOME/umi-miner/umi-reference-model" \
  --release-git-revision "$RELEASE_GIT_REVISION"
```

No extractor archive or MediaPipe task binary belongs in this release.

## 4. Build and bind the local extractor

Build from the checked-in Linux/AMD64 Docker source. The command resolves the
immutable image ID, verifies platform and non-root configuration, checks dependency
labels, starts the image without network access, verifies every package in the locked
requirements file, and checks the installed FFmpeg version.

```bash
cd "$HOME/umi-miner/umi-reference-model"
install -d -m 700 "$HOME/umi-miner/local-build"
DOCKER_EXECUTABLE="$(command -v docker)"
.venv/bin/python -m bitsign_motion.local_extractor_release \
  --docker "$DOCKER_EXECUTABLE" \
  --output "$HOME/umi-miner/local-build/local-extractor.json"
EXTRACTOR_IMAGE="$(
  jq -er .image_id "$HOME/umi-miner/local-build/local-extractor.json"
)"
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$EXTRACTOR_IMAGE"
```

The observed platform must be `linux/amd64`. Do not tag or substitute another image
after the build record is written.

Bind the local image ID into a derived bundle:

```bash
.venv/bin/python -m bitsign_motion.local_bundle_rebind \
  --base-archive "$HOME/umi-miner/release-assets/umi-s1-baseline-v0-portable.zip" \
  --base-sha256 "$BASE_MODEL_SHA256" \
  --build-record "$HOME/umi-miner/local-build/local-extractor.json" \
  --docker "$DOCKER_EXECUTABLE" \
  --output "$HOME/umi-miner/local-build/model-bundle"
MODEL_BUNDLE="$HOME/umi-miner/local-build/model-bundle"
MODEL_REVISION="$(jq -er .inference_revision "$MODEL_BUNDLE/inference-identity.json")"
test "$MODEL_REVISION" != "$(jq -r .inference_revision release/release-manifest.json)"
```

The rebinder verifies the published base bundle, copies its model, tokenizer, config,
rights, runtime, and postprocessing bytes unchanged, replaces only
`preprocessing.supported_oci_images["linux/amd64"]`, reseals the identity and manifest,
and loads the derived result again. The new revision identifies this local image. It
does not claim byte equivalence with another operator's image.

## 5. Download the MediaPipe task and probe

Download the task model directly from Google and verify its fixed digest. UMI does not
redistribute this file.

```bash
MEDIAPIPE_TASK="$HOME/umi-miner/release-assets/holistic_landmarker.task"
curl --fail --location --proto '=https' --tlsv1.2 \
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task?generation=1703178474695092' \
  --output "$MEDIAPIPE_TASK"
printf '%s  %s\n' \
  'e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8' \
  "$MEDIAPIPE_TASK" | shasum -a 256 -c -
chmod 600 "$MEDIAPIPE_TASK"
```

Set absolute paths and probe the full derived identity before starting the miner:

```bash
export UMI_S1_BUNDLE="$MODEL_BUNDLE"
export UMI_S1_INFERENCE_REVISION="$MODEL_REVISION"
export UMI_S1_EXTRACTOR_IMAGE="$EXTRACTOR_IMAGE"
export UMI_S1_EXTRACTOR_MODEL="$MEDIAPIPE_TASK"
export UMI_S1_EXTRACTOR_PLATFORM='linux/amd64'
export UMI_S1_DOCKER_EXECUTABLE="$DOCKER_EXECUTABLE"
export UMI_S1_TEMP_ROOT="$HOME/umi-miner/s1-jobs"
export UMI_S1_DEVICE='cpu'
export UMI_S1_HARD_DEADLINE_SECONDS='120'
install -d -m 700 "$UMI_S1_TEMP_ROOT"
"$HOME/umi-miner/umi-reference-model/.venv/bin/python" \
  -m bitsign_motion.umi_reference_backend probe
test -z "$(find "$UMI_S1_TEMP_ROOT" -mindepth 1 -print -quit)"
test -z "$(docker ps --all --filter name=bitsign-holistic- --format '{{.ID}}')"
```

The probe must report `status: ready`, `claim_status: component_test_no_weight`, and
the derived revision. A package, image, task, source, or identity mismatch blocks
startup.

## 6. Start the UMI miner

Create or select a Bittensor wallet and hotkey using the public UMI instructions. Keep
the video-host allowlist narrow. This example listens on all interfaces and admits one
validator and one HTTPS challenge host.

```bash
install -d -m 700 "$HOME/umi-miner/state"
"$HOME/umi-miner/umi-reference-model/.venv/bin/python" -m umi.miner \
  --wallet-name umi \
  --hotkey miner \
  --translator bitsign_motion.umi_reference_backend:translator \
  --model-revision "$UMI_S1_INFERENCE_REVISION" \
  --validator-hotkey VALIDATOR_SS58 \
  --video-host challenges.example.org \
  --nonce-db "$HOME/umi-miner/state/nonces.sqlite3" \
  --max-inference-concurrency 1 \
  --inference-timeout 120 \
  --listen-host 0.0.0.0 \
  --port 8091
```

Replace `VALIDATOR_SS58` and the challenge hostname with values supplied for the
component test. Put TLS and public ingress in front of port 8091. The endpoint trusts
only `btauth/1` requests from the explicit validator allowlist.

Check health locally:

```bash
curl --fail --silent http://127.0.0.1:8091/healthz | jq .
```

Expected health includes `translation_weights_active: false`,
`protocol_conformance: false`, and the derived model revision. Those false values are
intentional for this component-test release.

## Registration and failure handling

Registration spends the live SN78 registration cost. Review the current cost and UMI
instructions before submitting a transaction. The reference backend performs no chain
write; wallet signing stays in the UMI miner and `btcli` processes.

A missing model, hash mismatch, extractor failure, timeout, or invalid video produces
an explicit encrypted error response and scores zero. Do not add a text fallback. After
a timeout or cancellation, stop the miner if any generated job directory or
`bitsign-holistic-*` container remains, preserve bounded logs, and report the incident
with the derived model revision.
