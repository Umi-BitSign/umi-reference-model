# Public S1 release procedure

Use this procedure for a release produced by the `public-s1-finetune/1` training
path. It accepts only an intake that passes the reviewed external policy. The public
release contains aggregate evidence and a deterministic six-file model archive. It
must not contain source rows, videos, annotations, references, predictions, or
training checkpoints.

The release history has three commits:

1. Commit A contains the tested source, intake archive, intake evidence, intake
   policy, licenses, and source attribution files.
2. Commit B is the sole child of A and adds only
   `release/umi-s1-public-finetune-v1-release-e2e-evidence.json`.
3. Commit C is the sole child of B and changes only
   `release/release-manifest.json` and `release/SHA256SUMS`.

Do not push, tag, or upload the release until the project owner has reviewed commit C.

## 1. Run the public intake

The source release and intake policy come from the private training repository. The
policy must be reviewed and copied to its fixed public filename before intake. Run
every command block in this document in the same Bash session so the fail-fast
settings and exact revision, archive, and extractor variables remain in force.

```bash
set -euo pipefail
cd /absolute/path/to/umi-reference-model

SOURCE_RELEASE=/absolute/private/path/to/public-s1-release
SOURCE_POLICY=/absolute/private/path/to/intake-policy.json
INTAKE_TMP=/absolute/owner-only/path/to/new-intake-directory

uv run --frozen --extra dev python -m bitsign_motion.public_s1_release_intake \
  intake \
  --source "$SOURCE_RELEASE" \
  --policy "$SOURCE_POLICY" \
  --output "$INTAKE_TMP"

install -m 0644 \
  "$INTAKE_TMP/umi-s1-public-finetune-v1-portable.zip" \
  release/umi-s1-public-finetune-v1-portable.zip
install -m 0644 \
  "$INTAKE_TMP/umi-s1-public-finetune-v1-evidence.json" \
  release/umi-s1-public-finetune-v1-evidence.json
install -m 0644 "$SOURCE_POLICY" \
  release/umi-s1-public-finetune-v1-intake-policy.json
```

Run the intake verifier against the staged files. It requires an intake-only
directory, so verify `INTAKE_TMP` before removing it:

```bash
uv run --frozen --extra dev python -m bitsign_motion.public_s1_release_intake \
  verify \
  --root "$INTAKE_TMP" \
  --policy release/umi-s1-public-finetune-v1-intake-policy.json
```

Copy `LICENSE`, `NOTICE`, and `licenses/CC-BY-SA-4.0.txt` to `release/` under their
existing names. The artifact set is closed: each reviewed source also requires its
fixed license and attribution companion:

| Source | License companion | Attribution companion |
|---|---|---|
| 2M-Flores-ASL | `2M-FLORES-ASL-LICENSE.txt` | `2M-FLORES-ASL-ATTRIBUTION.txt` |
| FLEURS-ASL | `FLEURS-ASL-LICENSE.txt` | `FLEURS-ASL-ATTRIBUTION.txt` |
| FSboard | `FSBOARD-LICENSE.txt` | `FSBOARD-ATTRIBUTION.txt` |
| Taskmaster-1 | `TASKMASTER-LICENSE.txt` | `TASKMASTER-ATTRIBUTION.txt` |

Copy each exact license byte stream from the owner-reviewed source evidence whose
SHA-256 is pinned as that source's `license_sha256`. Generate each attribution file
as the exact UTF-8 encoding of the corresponding `attribution_notice` string in the
intake evidence, with no added newline. The release generator rejects a missing,
additional, renamed, or digest-mismatched companion.

## 2. Create and test commit A

Remove the previous v0 payloads and metadata. Keep the reviewed `README.md` and
`runtime-files.txt`; the public-S1 history uses an exact release-directory inventory
at every stage. Retaining or repurposing any other legacy filename makes
verification fail.

```bash
git rm -- \
  release/CC-BY-4.0.txt \
  release/FLEURS-ATTRIBUTION.txt \
  release/FSBOARD-ATTRIBUTION.txt \
  release/SHA256SUMS \
  release/release-manifest.json \
  release/umi-s1-baseline-v0-motion-ablation-evidence.json \
  release/umi-s1-baseline-v0-portable.zip \
  release/umi-s1-baseline-v0-release-e2e-evidence.json \
  release/umi-s1-baseline-v0-rights-decision.json \
  release/umi-s1-baseline-v0-selection-ledger.json
```

Run the locked checks before commit A:

```bash
uv lock --check
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev ruff format --check .
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev python tools/repo_guard.py
```

Commit the reviewed source, intake artifacts, policy, licenses, and attribution
files. Leave the public E2E file and generated metadata files out of commit A. Record
the resulting commit as `A`.

```bash
git status --short
git add --all
git diff --cached --check
git commit -m "Prepare public S1 release source"
A="$(git rev-parse HEAD)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

## 3. Bind the local extractor and run E2E

On the Linux/AMD64 E2E host, build the extractor from A, record its immutable image
identity outside the repository, and read the public base revision and archive
digest:

```bash
DOCKER_EXECUTABLE="$(command -v docker)"
EXTRACTOR_BUILD_RECORD=/absolute/owner-only/local-build/local-extractor.json
install -d -m 700 "$(dirname "$EXTRACTOR_BUILD_RECORD")"
test ! -e "$EXTRACTOR_BUILD_RECORD"
uv run --frozen --extra dev python -m bitsign_motion.local_extractor_release \
  --docker "$DOCKER_EXECUTABLE" \
  --output "$EXTRACTOR_BUILD_RECORD"

BASE_ARCHIVE=release/umi-s1-public-finetune-v1-portable.zip
BASE_INFERENCE_REVISION="$(
  unzip -p "$BASE_ARCHIVE" inference-identity.json | jq -er .inference_revision
)"
BASE_ARCHIVE_SHA256="$(sha256sum "$BASE_ARCHIVE" | cut -d ' ' -f 1)"

uv run --frozen --extra dev python -m bitsign_motion.local_bundle_rebind \
  --base-archive "$BASE_ARCHIVE" \
  --base-sha256 "$BASE_ARCHIVE_SHA256" \
  --base-inference-revision "$BASE_INFERENCE_REVISION" \
  --build-record "$EXTRACTOR_BUILD_RECORD" \
  --docker "$DOCKER_EXECUTABLE" \
  --output /absolute/owner-only/model-bundle
```

Run the clean Linux/AMD64 request-to-reveal E2E against A. Build the test environment
from both frozen lockfiles without editable installs:

```bash
A="$(git rev-parse HEAD)"
REFERENCE_ROOT="$(pwd -P)"
UMI_ROOT="$(cd ../umi && pwd -P)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$UMI_ROOT" status --porcelain=v1 --untracked-files=all)"
install -d -m 700 deployment-tmp

uv export --frozen --extra dev --no-dev --no-emit-project \
  --output-file deployment-tmp/release-model-requirements.txt
(
  cd "$UMI_ROOT"
  uv export --frozen --no-dev --no-emit-project \
    --output-file "$REFERENCE_ROOT/deployment-tmp/release-umi-requirements.txt"
)
uv venv --clear --python 3.12 deployment-tmp/release-e2e-venv
uv pip install --python deployment-tmp/release-e2e-venv/bin/python \
  --require-hashes \
  --requirement deployment-tmp/release-model-requirements.txt \
  --requirement deployment-tmp/release-umi-requirements.txt
E2E_PURELIB="$(
  deployment-tmp/release-e2e-venv/bin/python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"
printf '%s\n' "$REFERENCE_ROOT/src" > "$E2E_PURELIB/umi-reference-model-source.pth"
printf '%s\n' "$UMI_ROOT/src" > "$E2E_PURELIB/umi-source.pth"
chmod 600 \
  "$E2E_PURELIB/umi-reference-model-source.pth" \
  "$E2E_PURELIB/umi-source.pth"
unset PYTHONPATH
deployment-tmp/release-e2e-venv/bin/python -c \
  'import bitsign_motion, pytest, umi; print(bitsign_motion.__file__, pytest.__file__, umi.__file__)'
```

Set the public profile and every private runtime path. The report and task model stay
outside the repository, and the report directory must already exist with owner-only
permissions:

```bash
RUNTIME_ASSET_DIR=/absolute/owner-only/runtime-assets
install -d -m 700 "$RUNTIME_ASSET_DIR"
MEDIAPIPE_TASK="$RUNTIME_ASSET_DIR/holistic_landmarker.task"
test ! -e "$MEDIAPIPE_TASK"
curl --fail --location --proto '=https' --tlsv1.2 \
  'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task?generation=1703178474695092' \
  --output "$MEDIAPIPE_TASK"
printf '%s  %s\n' \
  'e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8' \
  "$MEDIAPIPE_TASK" | sha256sum -c -
chmod 600 "$MEDIAPIPE_TASK"

export BITSIGN_UMI_RELEASE_PROFILE=public-s1-finetune/1
export BITSIGN_RUN_UMI_RELEASE_E2E=1
export BITSIGN_UMI_REPOSITORY="$UMI_ROOT"
export BITSIGN_UMI_RELEASE_VIDEO=/absolute/private/path/to/eligible-test.mp4
export BITSIGN_UMI_RELEASE_VIDEO_RIGHTS_CLEARED=1
export BITSIGN_UMI_EXTRACTOR_BUILD_RECORD="$EXTRACTOR_BUILD_RECORD"
export BITSIGN_UMI_RELEASE_E2E_REPORT=/absolute/owner-only/e2e/private-run.json

export UMI_S1_BUNDLE=/absolute/owner-only/model-bundle
export UMI_S1_BASE_INFERENCE_REVISION="$BASE_INFERENCE_REVISION"
export UMI_S1_INFERENCE_REVISION="$(
  jq -er .inference_revision "$UMI_S1_BUNDLE/inference-identity.json"
)"
export UMI_S1_EXTRACTOR_IMAGE="$(
  jq -er .image_id "$BITSIGN_UMI_EXTRACTOR_BUILD_RECORD"
)"
export UMI_S1_EXTRACTOR_MODEL="$MEDIAPIPE_TASK"
export UMI_S1_EXTRACTOR_PLATFORM=linux/amd64
export UMI_S1_DOCKER_EXECUTABLE="$DOCKER_EXECUTABLE"
export UMI_S1_TEMP_ROOT=/absolute/owner-only/s1-jobs
export UMI_S1_DEVICE=cpu
export UMI_S1_HARD_DEADLINE_SECONDS=150

install -d -m 700 "$UMI_S1_TEMP_ROOT"
deployment-tmp/release-e2e-venv/bin/python -m pytest -q \
  tests/test_umi_release_e2e.py
```

A skipped test is not a pass. Project the owner-only run into the aggregate public
schema, install it under the public-S1 filename, verify its source binding, and
commit only that file as B:

```bash
PUBLIC_E2E_STAGING=/absolute/owner-only/e2e/public-s1-release-e2e.json
test ! -e "$PUBLIC_E2E_STAGING"
uv run --frozen --extra dev python -m bitsign_motion.s1_release_evidence \
  release-e2e \
  --run-report "$BITSIGN_UMI_RELEASE_E2E_REPORT" \
  --output "$PUBLIC_E2E_STAGING"
install -m 0644 "$PUBLIC_E2E_STAGING" \
  release/umi-s1-public-finetune-v1-release-e2e-evidence.json
test "$(jq -er .schema \
  release/umi-s1-public-finetune-v1-release-e2e-evidence.json)" = \
  "umi-s1-public-finetune-release-e2e/1"
test "$(jq -er .release_profile \
  release/umi-s1-public-finetune-v1-release-e2e-evidence.json)" = \
  "public-s1-finetune/1"
test "$(jq -er .tested_reference_model_git_revision \
  release/umi-s1-public-finetune-v1-release-e2e-evidence.json)" = "$A"

git add release/umi-s1-public-finetune-v1-release-e2e-evidence.json
test "$(git diff --cached --name-only)" = \
  "release/umi-s1-public-finetune-v1-release-e2e-evidence.json"
test -z "$(git diff --name-only)"
test -z "$(git ls-files --others --exclude-standard)"
git diff --cached --check
git commit -m "Record public S1 release E2E evidence"
B="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD^)" = "$A"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

## 4. Generate commit C

Set the exact revisions and list the complete fixed release inventory. Additional
artifact arguments are rejected.

```bash
B="$(git rev-parse HEAD)"
PUBLIC_E2E=release/umi-s1-public-finetune-v1-release-e2e-evidence.json
UMI_GIT_REVISION="$(jq -er .umi_git_revision "$PUBLIC_E2E")"
test "$(git -C ../umi rev-parse HEAD)" = "$UMI_GIT_REVISION"
test -z "$(git -C ../umi status --porcelain=v1 --untracked-files=all)"

uv run --frozen --extra dev python tools/release_artifacts.py \
  --release-id umi-s1-public-finetune-v1 \
  --inference-revision "$BASE_INFERENCE_REVISION" \
  --public-s1-policy release/umi-s1-public-finetune-v1-intake-policy.json \
  --source-git-revision "$B" \
  --umi-git-revision "$UMI_GIT_REVISION" \
  --artifact model=release/umi-s1-public-finetune-v1-portable.zip \
  --artifact intake-evidence=release/umi-s1-public-finetune-v1-evidence.json \
  --artifact intake-policy=release/umi-s1-public-finetune-v1-intake-policy.json \
  --artifact release-e2e-evidence=release/umi-s1-public-finetune-v1-release-e2e-evidence.json \
  --artifact code-license=release/LICENSE \
  --artifact notice=release/NOTICE \
  --artifact model-license=release/CC-BY-SA-4.0.txt \
  --artifact two-m-flores-license=release/2M-FLORES-ASL-LICENSE.txt \
  --artifact two-m-flores-attribution=release/2M-FLORES-ASL-ATTRIBUTION.txt \
  --artifact fleurs-license=release/FLEURS-ASL-LICENSE.txt \
  --artifact fleurs-attribution=release/FLEURS-ASL-ATTRIBUTION.txt \
  --artifact fsboard-license=release/FSBOARD-LICENSE.txt \
  --artifact fsboard-attribution=release/FSBOARD-ATTRIBUTION.txt \
  --artifact taskmaster-license=release/TASKMASTER-LICENSE.txt \
  --artifact taskmaster-attribution=release/TASKMASTER-ATTRIBUTION.txt \
  --output-directory release
```

Review the manifest and checksum file, then commit only those two files as commit C.
Verify the complete tree and its history:

```bash
git add release/release-manifest.json release/SHA256SUMS
test "$(git diff --cached --name-only | LC_ALL=C sort)" = \
  "$(printf '%s\n' release/SHA256SUMS release/release-manifest.json)"
test -z "$(git diff --name-only)"
test -z "$(git ls-files --others --exclude-standard)"
git diff --cached --check
git commit -m "Seal public S1 release metadata"
C="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD^)" = "$B"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
uv run --frozen --extra dev python tools/release_artifacts.py \
  --verify release/release-manifest.json \
  --artifact-directory release \
  --repository . \
  --release-git-revision "$C"
(
  cd release
  sha256sum -c SHA256SUMS
)
```

The tag points to C. Any change to the runtime, model, policy, evidence, licenses,
or attribution files requires a new commit A and another E2E run.
