# Run the UMI S1 reference miner

This profile serves the low-accuracy S1 public bootstrap through a no-weight UMI
miner. It is a starter model for miners to inspect and improve. It is not an ASL
translator, accessibility tool, production service, or activation result. UMI
translation weights are inactive, so registration and successful responses do not
guarantee mining rewards.

The release does not distribute a Docker image. Each operator builds the extractor
from source, validates the resulting local image, and derives a new model revision
that binds that image's immutable ID.

Live calibration also requires a finalized, signed UMI inactive release. Obtain its
location, canonical `release-manifest.json` SHA-256, and release-authority hotkey
through a trusted channel. The `umi_git_revision` in this model's manifest records
the UMI revision used for its published component E2E run. It does not authorize a
live policy or replace the signed inactive release.

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

## 1. Verify the inactive release and check out exact code

Verify the UMI release with a verifier from an independently trusted checkout or a
previously verified wheel. Then clone this repository and the public UMI repository
beside each other. Check out the model release tag and the UMI commit named by the
signed inactive release.

The values below identify the reviewed public-finetune v1 model release. Confirm
both through the trusted release announcement before executing code from the
checkout. The signed `umi-s1-public-finetune-v1` tag must resolve to the exact
reviewed release commit; a newer development `main` is not a substitute.

Run Sections 1 through 5 in the same Bash session. The first command enables
fail-fast handling so a failed digest, revision, import, or cleanup check stops
setup. Section 6 saves the non-secret runtime configuration in an owner-only file,
then deliberately splits the foreground miner and the verification commands between
two terminals.

```bash
set -euo pipefail
export UMI_INACTIVE_RELEASE=/absolute/path/to/public-inactive-release
export UMI_RELEASE_MANIFEST_SHA256=64_LOWERCASE_HEX_CHARACTERS
export UMI_RELEASE_AUTHORITY=EXPECTED_RELEASE_AUTHORITY_SS58
export TRUSTED_UMI_RELEASE_VERIFY=/absolute/path/to/trusted/umi-shadow-release-verify
export MODEL_RELEASE_TAG=umi-s1-public-finetune-v1
export EXPECTED_MODEL_RELEASE_GIT_REVISION=cd31402c451f5da3af87fcd8c10f9ed0fbe43d27
test -x "$TRUSTED_UMI_RELEASE_VERIFY"
test "$(sha256sum "$UMI_INACTIVE_RELEASE/release-manifest.json" | cut -d ' ' -f 1)" = \
  "$UMI_RELEASE_MANIFEST_SHA256"
"$TRUSTED_UMI_RELEASE_VERIFY" "$UMI_INACTIVE_RELEASE" \
  --expected-authority-hotkey "$UMI_RELEASE_AUTHORITY"
test "$(jq -r .translation_weights_active \
  "$UMI_INACTIVE_RELEASE/release-manifest.json")" = false
UMI_GIT_REVISION="$(jq -er .umi_git_revision \
  "$UMI_INACTIVE_RELEASE/release-manifest.json")"

mkdir -p "$HOME/umi-miner"
cd "$HOME/umi-miner"
git clone https://github.com/Umi-BitSign/umi-reference-model.git
git clone https://github.com/Umi-BitSign/umi.git
cd umi-reference-model
test "$(git rev-parse "$MODEL_RELEASE_TAG^{commit}")" = \
  "$EXPECTED_MODEL_RELEASE_GIT_REVISION"
git checkout --detach "$EXPECTED_MODEL_RELEASE_GIT_REVISION"
test "$(git rev-parse HEAD)" = "$EXPECTED_MODEL_RELEASE_GIT_REVISION"
SOURCE_GIT_REVISION="$(jq -r .source_git_revision release/release-manifest.json)"
RELEASE_GIT_REVISION="$(git rev-parse HEAD)"
test "$RELEASE_GIT_REVISION" = "$EXPECTED_MODEL_RELEASE_GIT_REVISION"
test "$(git rev-parse HEAD^)" = "$SOURCE_GIT_REVISION"
test "$(git rev-list --parents -n 1 HEAD | wc -w | tr -d ' ')" = 2
test "$(git diff --name-only --no-renames "$SOURCE_GIT_REVISION" HEAD | LC_ALL=C sort)" = \
  "$(printf '%s\n' release/SHA256SUMS release/release-manifest.json)"
git -C ../umi checkout --detach "$UMI_GIT_REVISION"
test "$(git -C ../umi rev-parse HEAD)" = "$UMI_GIT_REVISION"
test -z "$(git status --short)"
test -z "$(git -C ../umi status --short)"
```

Replace the UMI release placeholders with values received through the trusted
release channel. Stop if either model tag or commit differs from the trusted
announcement, release verification fails, either checkout is dirty, or a revision
check fails. Follow UMI's `docs/SHADOW_CALIBRATION_OPERATOR.md` if the trusted
verifier is not already installed.

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

## 3. Verify the sealed release artifacts

The immutable release tag contains one manifest-defined model archive, aggregate
evidence, and its license companions under `release/`. Verify the complete set before
using the model. `release/SHA256SUMS` must contain one line for every manifest artifact.

```bash
cd "$HOME/umi-miner/umi-reference-model"
(
  cd release
  sha256sum -c SHA256SUMS
)
BASE_MODEL_SHA256="$(
  jq -er '.artifacts[] | select(.label == "model") | .sha256' \
    release/release-manifest.json
)"
BASE_MODEL_FILENAME="$(
  jq -er '.artifacts[] | select(.label == "model") | .filename' \
    release/release-manifest.json
)"
test "$BASE_MODEL_FILENAME" = "$(basename "$BASE_MODEL_FILENAME")"
BASE_MODEL_PATH="$HOME/umi-miner/umi-reference-model/release/$BASE_MODEL_FILENAME"
test "$(sha256sum "$BASE_MODEL_PATH" | cut -d ' ' -f 1)" = "$BASE_MODEL_SHA256"
BASE_INFERENCE_REVISION="$(jq -er .inference_revision release/release-manifest.json)"
"$HOME/umi-miner/umi-reference-model/.venv/bin/python" \
  "$HOME/umi-miner/umi-reference-model/tools/release_artifacts.py" \
  --verify "$HOME/umi-miner/umi-reference-model/release/release-manifest.json" \
  --artifact-directory "$HOME/umi-miner/umi-reference-model/release" \
  --repository "$HOME/umi-miner/umi-reference-model" \
  --release-git-revision "$RELEASE_GIT_REVISION"
```

The verifier checks the A -> B -> C commit history, the aggregate-only evidence,
license copies, source closure, artifact names, and every digest. No extractor archive
or MediaPipe task binary belongs in the release.

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
  --base-archive "$BASE_MODEL_PATH" \
  --base-sha256 "$BASE_MODEL_SHA256" \
  --base-inference-revision "$BASE_INFERENCE_REVISION" \
  --build-record "$HOME/umi-miner/local-build/local-extractor.json" \
  --docker "$DOCKER_EXECUTABLE" \
  --output "$HOME/umi-miner/local-build/model-bundle"
MODEL_BUNDLE="$HOME/umi-miner/local-build/model-bundle"
MODEL_REVISION="$(jq -er .inference_revision "$MODEL_BUNDLE/inference-identity.json")"
test "$MODEL_REVISION" != "$BASE_INFERENCE_REVISION"
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
install -d -m 700 "$HOME/umi-miner/runtime-assets"
MEDIAPIPE_TASK="$HOME/umi-miner/runtime-assets/holistic_landmarker.task"
curl --fail --location --proto '=https' --tlsv1.2 \
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task?generation=1703178474695092' \
  --output "$MEDIAPIPE_TASK"
printf '%s  %s\n' \
  'e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8' \
  "$MEDIAPIPE_TASK" | sha256sum -c -
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
export UMI_S1_HARD_DEADLINE_SECONDS='150'
install -d -m 700 "$UMI_S1_TEMP_ROOT"
"$HOME/umi-miner/umi-reference-model/.venv/bin/python" \
  -m bitsign_motion.umi_reference_backend probe
test -z "$(find "$UMI_S1_TEMP_ROOT" -mindepth 1 -print -quit)"
test -z "$(docker ps --all --filter name=bitsign-holistic- --format '{{.ID}}')"
```

The probe must report `status: ready` and the derived revision. Check that the
reported no-weight claim status agrees with the signed release manifest. A package,
image, task, source, or identity mismatch blocks startup.

Save the runtime variables for the two terminals used below. This file contains no
wallet seed, but keep it owner-only because it describes the local deployment:

```bash
install -d -m 700 "$HOME/umi-miner/state"
RUNTIME_ENV="$HOME/umi-miner/state/reference-miner.env"
umask 077
for NAME in \
  UMI_INACTIVE_RELEASE \
  UMI_RELEASE_MANIFEST_SHA256 \
  UMI_RELEASE_AUTHORITY \
  TRUSTED_UMI_RELEASE_VERIFY \
  UMI_S1_BUNDLE \
  UMI_S1_INFERENCE_REVISION \
  UMI_S1_EXTRACTOR_IMAGE \
  UMI_S1_EXTRACTOR_MODEL \
  UMI_S1_EXTRACTOR_PLATFORM \
  UMI_S1_DOCKER_EXECUTABLE \
  UMI_S1_TEMP_ROOT \
  UMI_S1_DEVICE \
  UMI_S1_HARD_DEADLINE_SECONDS
do
  printf 'export %s=%q\n' "$NAME" "${!NAME}"
done > "$RUNTIME_ENV"
chmod 600 "$RUNTIME_ENV"
```

## 6. Register, start, and announce the UMI miner

Create or select a Bittensor wallet and hotkey. Register that hotkey on SN78 once,
after reviewing the live registration cost:

```bash
cd "$HOME/umi-miner/umi-reference-model"
.venv/bin/btcli subnets register \
  --netuid 78 \
  --network finney \
  --wallet umi \
  --wallet-hotkey miner \
  --mev-shield
```

Registration is coldkey-signed and spends the live cost. Stop if the client cannot
use the required MEV-shielded path.

The miner takes its validator registry, protocol limits, finality pin, and activation
clock from the exact signed policy. Do not edit or reserialize that policy. In
terminal A, resolve the signed artifact paths, derive the delivery-host allowlist,
and start the foreground backend on loopback port 8091. Leave this process running.

```bash
set -euo pipefail
cd "$HOME/umi-miner/umi-reference-model"
source "$HOME/umi-miner/state/reference-miner.env"
RELEASE_MANIFEST="$UMI_INACTIVE_RELEASE/release-manifest.json"
test "$(sha256sum "$RELEASE_MANIFEST" | cut -d ' ' -f 1)" = \
  "$UMI_RELEASE_MANIFEST_SHA256"
test -x "$TRUSTED_UMI_RELEASE_VERIFY"
"$TRUSTED_UMI_RELEASE_VERIFY" "$UMI_INACTIVE_RELEASE" \
  --expected-authority-hotkey "$UMI_RELEASE_AUTHORITY"
SCORING_POLICY="$UMI_INACTIVE_RELEASE/scoring-policy.json"
SCORING_POLICY_SHA256="$(jq -er .scoring_policy_sha256 "$RELEASE_MANIFEST")"
test "$(sha256sum "$SCORING_POLICY" | cut -d ' ' -f 1)" = \
  "$SCORING_POLICY_SHA256"
test "$(jq -r .translation_weights_active "$SCORING_POLICY")" = false
TARGET_TRIPLE="$(jq -er .release_authority.intent.target_triple "$RELEASE_MANIFEST")"

release_artifact() {
  local relative
  relative="$(jq -er --arg label "$1" \
    '.external_artifacts[] | select(.label == $label) | .relative_path' \
    "$RELEASE_MANIFEST")"
  printf '%s/%s\n' "$UMI_INACTIVE_RELEASE" "$relative"
}

FINALITY_VERIFIER="$(release_artifact finality_verifier_binary)"
FINALITY_CHAIN_SPEC="$(release_artifact finality_chain_spec)"
MIRROR_DISCOVERY="$(release_artifact mirror_discovery_rule)"
VIDEO_HOST_ARGS=()
VIDEO_PORT_ARGS=()
while IFS=$'\t' read -r HOST PORT; do
  VIDEO_HOST_ARGS+=(--video-host "$HOST")
  VIDEO_PORT_ARGS+=(--video-port "$PORT")
done < <("$HOME/umi-miner/umi-reference-model/.venv/bin/python" \
  - "$MIRROR_DISCOVERY" <<'PY'
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

document = json.loads(Path(sys.argv[1]).read_bytes())
for origin in document["delivery_origins"]:
    parsed = urlsplit(origin)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise SystemExit("invalid signed delivery origin")
    print(parsed.hostname, parsed.port or 443, sep="\t")
PY
)
test "${#VIDEO_HOST_ARGS[@]}" -gt 0
test "${#VIDEO_HOST_ARGS[@]}" = "${#VIDEO_PORT_ARGS[@]}"

install -d -m 700 "$HOME/umi-miner/state"
"$HOME/umi-miner/umi-reference-model/.venv/bin/python" -m umi.miner \
  --wallet-name umi \
  --hotkey miner \
  --policy "$SCORING_POLICY" \
  --target-triple "$TARGET_TRIPLE" \
  --finality-verifier-binary "$FINALITY_VERIFIER" \
  --finality-chain-spec "$FINALITY_CHAIN_SPEC" \
  --finality-state "$HOME/umi-miner/state/miner-finality.sqlite3" \
  --translator bitsign_motion.umi_reference_backend:translator \
  --model-revision "$UMI_S1_INFERENCE_REVISION" \
  "${VIDEO_HOST_ARGS[@]}" \
  "${VIDEO_PORT_ARGS[@]}" \
  --nonce-db "$HOME/umi-miner/state/nonces.sqlite3" \
  --assignment-db "$HOME/umi-miner/state/assignments.sqlite3" \
  --inference-timeout 180 \
  --inference-admission-timeout 10 \
  --backend-lifecycle-timeout 60 \
  --listen-host 127.0.0.1 \
  --port 8091
```

The default inference concurrency is the number of validators in the signed policy;
a lower value is rejected because each validator needs one schedulable slot. The
endpoint trusts only `btauth/1` requests from that policy registry. The host and port
allowlists come from every signed delivery origin, including IPv6 literals and
non-default HTTPS ports.

Open terminal B, load the same configuration, and check health locally:

```bash
set -euo pipefail
cd "$HOME/umi-miner/umi-reference-model"
source "$HOME/umi-miner/state/reference-miner.env"
HEALTH="$(curl --fail --silent http://127.0.0.1:8091/healthz)"
printf '%s\n' "$HEALTH" | jq .
SCORING_POLICY_SHA256="$(jq -er .scoring_policy_sha256 \
  "$UMI_INACTIVE_RELEASE/release-manifest.json")"
printf '%s\n' "$HEALTH" | jq -e \
  --arg policy "$SCORING_POLICY_SHA256" \
  --arg revision "$UMI_S1_INFERENCE_REVISION" \
  '.ok == true and
   .translation_weights_active == false and
   .protocol_conformance == false and
   .runtime_mode == "inactive_shadow" and
   .scoring_policy_sha256 == $policy and
   .model_revision == $revision and
   .window_authority == "ProofBackedMinerWindowAuthority" and
   .finality_service == "running"'
```

The false weight and conformance values are intentional. A different policy hash,
model revision, authority mode, or finality status blocks announcement.

Expose the loopback service through a TLS edge or reverse proxy at a stable public IP
and port, then announce that exact endpoint on SN78. The proxy must preserve the raw
request target, authentication headers, and body bytes. It must also enforce finite
header and body timeouts plus the signed 16 KiB header ceiling. The announced port is
the ingress or proxy's public port; it may differ from backend port 8091.

```bash
cd "$HOME/umi-miner/umi-reference-model"
source "$HOME/umi-miner/state/reference-miner.env"
PUBLIC_IP=203.0.113.10
PUBLIC_PORT=8091

.venv/bin/btcli tx serve-axon \
  --netuid 78 \
  --ip "$PUBLIC_IP" \
  --port "$PUBLIC_PORT" \
  --network finney \
  --wallet umi \
  --wallet-hotkey miner \
  --no-mev-shield
```

`serve-axon` is hotkey-signed. The reference miner itself still performs no chain
write. Verify discovery from another host with the same pinned environment:

```bash
export MINER_HOTKEY_SS58=YOUR_MINER_HOTKEY_SS58
.venv/bin/python -c \
  'import asyncio, os; from umi.chain import discover_miner; print(asyncio.run(discover_miner(os.environ["MINER_HOTKEY_SS58"])))'
```

The returned hotkey, UID, and endpoint must match the registration and public
ingress. Then fetch `/healthz` through that public endpoint and have the validator run
once without an explicit `--miner-url`; this exercises read-only metagraph discovery.

## 7. Failure handling

A missing model, hash mismatch, extractor failure, timeout, or invalid video produces
an explicit encrypted error response and scores zero. Do not add a text fallback. After
a timeout or cancellation, stop the miner if any generated job directory or
`bitsign-holistic-*` container remains, preserve bounded logs, and report the incident
with the derived model revision.
