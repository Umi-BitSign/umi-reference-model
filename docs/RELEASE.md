# Release checklist

This checklist preserves the legacy `umi-s1-baseline-v0` release procedure. New
`public-s1-finetune/1` candidates use [the public S1 release procedure](PUBLIC_S1_RELEASE.md).
Both profiles use the same strict release verifier and the same A -> B -> C history.

A release may be labeled `component_test_no_weight` only after every item below
passes. The released model remains a low-accuracy integration fixture. This process
does not activate UMI translation weights or establish useful ASL translation.

The release history has three commits:

1. commit A contains the tested source, base model ZIP, three pre-E2E aggregate
   evidence files, and six license companions; an unpublished reseal may also
   retain the prior generated artifacts at A;
2. commit B is the sole child of A and changes only the public E2E evidence file;
3. commit C is the sole child of B and changes only `release/release-manifest.json`
   and `release/SHA256SUMS`.

An immutable release tag must point to C. Do not push, tag, or upload any release
file until the project owner approves C and its checksum set.

## 1. Freeze the model and aggregate evidence

The private training and evaluation authorities must be canonical and
content-addressed. Public release evidence is aggregate-only. It must not contain
source rows, signer or sample identifiers, references, tensor identities, token IDs,
hypotheses, or edit distances.

Set absolute paths to the reviewed private reports and final portable bundle:

```bash
set -euo pipefail
cd /absolute/path/to/umi-reference-model

MODEL_BUNDLE=/absolute/path/to/portable-base-bundle
DECODER_REPORT=/absolute/private/path/to/decoder-comparison.json
CHECKPOINT_REPORT=/absolute/private/path/to/checkpoint-sweep.json
GUIDANCE_REPORT=/absolute/private/path/to/motion-guidance-sweep.json
MOTION_REPORT=/absolute/private/path/to/motion-ablation.json
FINAL_RIGHTS_REVIEW=/absolute/private/path/to/final-rights-review.json
BASE_INFERENCE_REVISION="$(
  jq -er .inference_revision "$MODEL_BUNDLE/inference-identity.json"
)"
RIGHTS_DECISION_SHA256="$(
  jq -er .rights.release_rights_decision_sha256 \
    "$MODEL_BUNDLE/inference-identity.json"
)"
```

Project the reviewed private reports to the three public pre-E2E records. The output
paths must not already exist; archive or remove an earlier candidate before running
these commands.

```bash
uv run --frozen --extra dev python -m bitsign_motion.s1_release_evidence \
  selection \
  --decoder-report "$DECODER_REPORT" \
  --checkpoint-report "$CHECKPOINT_REPORT" \
  --guidance-report "$GUIDANCE_REPORT" \
  --release-inference-revision "$BASE_INFERENCE_REVISION" \
  --output release/umi-s1-baseline-v0-selection-ledger.json

uv run --frozen --extra dev python -m bitsign_motion.s1_release_evidence \
  motion-ablation \
  --report "$MOTION_REPORT" \
  --release-inference-revision "$BASE_INFERENCE_REVISION" \
  --output release/umi-s1-baseline-v0-motion-ablation-evidence.json

uv run --frozen --extra dev python -m bitsign_motion.s1_release_evidence \
  rights \
  --review "$FINAL_RIGHTS_REVIEW" \
  --output release/umi-s1-baseline-v0-rights-decision.json
```

Check these authority conditions:

- the selection ledger binds the selected model state, tokenizer, decoder contract,
  private report digests, and 285-sample aggregate validation results, and names the
  predecessor evaluation revision when a source-closure rebind changes the release
  revision;
- the motion-ablation evidence states that zero motion outscored real motion and that
  useful motion grounding is not established;
- the rights evidence permits public weight redistribution under CC BY-SA 4.0 and
  forbids redistribution of source data and the MediaPipe task binary;
- the private report files remain outside the repository and release directory;
- no row-level validation evidence is present. Row-level evaluation formats are not
  release artifacts.

## 2. Assemble commit A

Package the final base bundle and copy the exact license companions from their
canonical repository locations:

```bash
uv run --frozen --extra dev python tools/package_bundle.py \
  --bundle "$MODEL_BUNDLE" \
  --output release/umi-s1-baseline-v0-portable.zip

install -m 0644 LICENSE release/LICENSE
install -m 0644 NOTICE release/NOTICE
install -m 0644 licenses/CC-BY-SA-4.0.txt release/CC-BY-SA-4.0.txt
install -m 0644 licenses/FLEURS-ATTRIBUTION.txt release/FLEURS-ATTRIBUTION.txt
install -m 0644 licenses/CC-BY-4.0.txt release/CC-BY-4.0.txt
install -m 0644 licenses/FSBOARD-ATTRIBUTION.txt release/FSBOARD-ATTRIBUTION.txt
```

The complete runtime closure must also be frozen:

- the portable identity names every transitive inference and backend module and
  binds Python, PyTorch, NumPy, Safetensors, RFC 8785, and Bittensor where used;
- `release/runtime-files.txt` matches the public source closure;
- the source staging checks in `docs/SOURCE_STAGING.md` pass;
- every runtime and preprocessing digest in `inference-identity.json` matches the
  checked-in source;
- loading the portable bundle requires its exact base inference revision;
- the model card reports the aggregate results and failed motion-grounding diagnostic
  without a production, accessibility, mobile, activation, or reward claim.

Run the local gates:

```bash
uv lock --check
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev ruff format --check .
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev python tools/repo_guard.py
```

Run the corresponding locked gates in the sibling UMI repository. Its working tree
must be clean. Record its full commit as `UMI_GIT_REVISION`.

```bash
(
  cd ../umi
  uv lock --check
  uv run --frozen --extra dev ruff check .
  uv run --frozen --extra dev ruff format --check .
  uv run --frozen --extra dev pytest -q
)
```

Commit all reviewed source and release inputs without regenerating the two metadata
files. This is commit A. For an initial seal, the public E2E evidence and metadata
are absent at A. When resealing the same release ID before it has been published,
the prior tracked E2E evidence and metadata may remain at A; commits B and C replace
them respectively. Never reuse a published release ID.

```bash
git status --short
git add --all
git diff --cached --check
git commit -m "Prepare UMI S1 baseline release source"
A="$(git rev-parse HEAD)"
UMI_GIT_REVISION="$(git -C ../umi rev-parse HEAD)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C ../umi status --porcelain=v1 --untracked-files=all)"
test ! -e release/umi-s1-baseline-v0-release-e2e-evidence.json ||
  git ls-files --error-unmatch \
    release/umi-s1-baseline-v0-release-e2e-evidence.json >/dev/null
```

Any runtime, model, evidence, dependency, or extractor-source change after A requires
a new A and another E2E run.

## 3. Build and bind the Linux/AMD64 extractor

The project does not distribute a Docker image archive. On the Linux/AMD64 E2E host,
build the extractor from commit A and bind its immutable image ID into a derived
bundle:

```bash
install -d -m 700 /absolute/owner-only/local-build
DOCKER_EXECUTABLE="$(command -v docker)"

uv run --frozen --extra dev python -m bitsign_motion.local_extractor_release \
  --docker "$DOCKER_EXECUTABLE" \
  --output /absolute/owner-only/local-build/local-extractor.json

EXTRACTOR_IMAGE="$(
  jq -er .image_id /absolute/owner-only/local-build/local-extractor.json
)"

uv run --frozen --extra dev python -m bitsign_motion.local_bundle_rebind \
  --base-archive release/umi-s1-baseline-v0-portable.zip \
  --base-sha256 "$(
    sha256sum release/umi-s1-baseline-v0-portable.zip | cut -d ' ' -f 1
  )" \
  --build-record /absolute/owner-only/local-build/local-extractor.json \
  --docker "$DOCKER_EXECUTABLE" \
  --output /absolute/owner-only/local-build/model-bundle
```

Require a new `sha256:` image ID, `linux/amd64`, a non-root image user, exact locked
dependencies, a clean package-consistency check, and the pinned FFmpeg package. The
rebinder may change only the supported local image ID, resulting inference revision,
and derived bundle manifest. Another host must build and bind its own image.

Download the MediaPipe task from the URL in `docs/RUN_MINER.md` and verify SHA-256
`e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8`.
The task binary remains outside the repository.

## 4. Run the clean Linux/AMD64 E2E test against A

Use a clean Python 3.12 venv built from the two frozen lockfiles. Export the reference
repository's `dev` extra and UMI's runtime dependencies, install both with hashes, add
path-only `.pth` files for the two clean source trees, and unset `PYTHONPATH`. Neither
project may be installed in editable mode:

```bash
REFERENCE_ROOT="$(pwd -P)"
UMI_ROOT="$(cd ../umi && pwd -P)"
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
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$UMI_ROOT" status --porcelain=v1 --untracked-files=all)"
```

The test requires a private video whose rights cover this run. It tests one valid and
one invalid request through the real local extractor and model, receives signed
timelock-encrypted envelopes, waits for reveal, validates both plaintext results, and
checks job and container cleanup. It does not measure translation quality.

Set every required value. The report path must be a new file in an existing
owner-only directory:

```bash
BASE_INFERENCE_REVISION="$(
  unzip -p release/umi-s1-baseline-v0-portable.zip inference-identity.json |
    jq -er .inference_revision
)"
export BITSIGN_RUN_UMI_RELEASE_E2E=1
export BITSIGN_UMI_REPOSITORY=/absolute/path/to/umi
export BITSIGN_UMI_RELEASE_VIDEO=/absolute/private/path/to/eligible-test.mp4
export BITSIGN_UMI_RELEASE_VIDEO_RIGHTS_CLEARED=1
export BITSIGN_UMI_EXTRACTOR_BUILD_RECORD=/absolute/owner-only/local-build/local-extractor.json
export BITSIGN_UMI_RELEASE_E2E_REPORT=/absolute/owner-only/e2e/private-run.json

export UMI_S1_BUNDLE=/absolute/owner-only/local-build/model-bundle
export UMI_S1_BASE_INFERENCE_REVISION="$BASE_INFERENCE_REVISION"
export UMI_S1_INFERENCE_REVISION="$(
  jq -er .inference_revision "$UMI_S1_BUNDLE/inference-identity.json"
)"
export UMI_S1_EXTRACTOR_IMAGE="$(
  jq -er .image_id /absolute/owner-only/local-build/local-extractor.json
)"
export UMI_S1_EXTRACTOR_MODEL=/absolute/owner-only/holistic_landmarker.task
export UMI_S1_EXTRACTOR_PLATFORM=linux/amd64
export UMI_S1_DOCKER_EXECUTABLE="$(command -v docker)"
export UMI_S1_TEMP_ROOT=/absolute/owner-only/s1-jobs
export UMI_S1_DEVICE=cpu
export UMI_S1_HARD_DEADLINE_SECONDS=150

install -d -m 700 "$UMI_S1_TEMP_ROOT"
deployment-tmp/release-e2e-venv/bin/python -m pytest -q \
  tests/test_umi_release_e2e.py
```

The outer inference, admission, and lifecycle timeouts are fixed by the test at 180,
10, and 60 seconds. Together with the 150-second model deadline, the public record
must contain the exact `150/180/10/60` timeout tuple. A skipped test is not a pass.
The private report must be mode `0600` and remain outside the repository.

Project the sealed private run to a new temporary aggregate record, then install it
at the fixed release path. The temporary path must be outside the repository and
must not already exist:

```bash
PUBLIC_E2E_STAGING=/absolute/owner-only/e2e/public-release-e2e.json
test ! -e "$PUBLIC_E2E_STAGING"
uv run --frozen --extra dev python -m bitsign_motion.s1_release_evidence \
  release-e2e \
  --run-report "$BITSIGN_UMI_RELEASE_E2E_REPORT" \
  --output "$PUBLIC_E2E_STAGING"
install -m 0644 "$PUBLIC_E2E_STAGING" \
  release/umi-s1-baseline-v0-release-e2e-evidence.json
```

The public record must name A as `tested_reference_model_git_revision`, bind the exact
UMI commit, base and derived inference revisions, local image ID, dependency versions,
test result digests, and cleanup results. It must exclude the video digest, private
response material, and fixture identity.

## 5. Commit B and generate commit C

Commit only the public E2E evidence file. It may be new for an initial seal or
modified for an unpublished reseal. This is commit B:

```bash
A="$(
  jq -er .tested_reference_model_git_revision \
    release/umi-s1-baseline-v0-release-e2e-evidence.json
)"
test "$(git rev-parse HEAD)" = "$A"
git add release/umi-s1-baseline-v0-release-e2e-evidence.json
test "$(git diff --cached --name-only)" = \
  "release/umi-s1-baseline-v0-release-e2e-evidence.json"
test -z "$(git diff --name-only)"
test -z "$(git ls-files --others --exclude-standard)"
git diff --cached --check
git commit -m "Record UMI S1 release E2E evidence"
B="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD^)" = "$A"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

Generate the schema-v2 manifest and checksum file from B. The label and filename set
is fixed by `tools/release_artifacts.py`:

```bash
RIGHTS_DECISION_SHA256="$(
  unzip -p release/umi-s1-baseline-v0-portable.zip inference-identity.json |
    jq -er .rights.release_rights_decision_sha256
)"
UMI_GIT_REVISION="$(
  jq -er .umi_git_revision release/umi-s1-baseline-v0-release-e2e-evidence.json
)"
uv run --frozen --extra dev python tools/release_artifacts.py \
  --release-id umi-s1-baseline-v0 \
  --inference-revision "$BASE_INFERENCE_REVISION" \
  --rights-decision-sha256 "$RIGHTS_DECISION_SHA256" \
  --source-git-revision "$B" \
  --umi-git-revision "$UMI_GIT_REVISION" \
  --artifact model=release/umi-s1-baseline-v0-portable.zip \
  --artifact selection-ledger=release/umi-s1-baseline-v0-selection-ledger.json \
  --artifact motion-ablation-evidence=release/umi-s1-baseline-v0-motion-ablation-evidence.json \
  --artifact rights-evidence=release/umi-s1-baseline-v0-rights-decision.json \
  --artifact release-e2e-evidence=release/umi-s1-baseline-v0-release-e2e-evidence.json \
  --artifact code-license=release/LICENSE \
  --artifact notice=release/NOTICE \
  --artifact model-license=release/CC-BY-SA-4.0.txt \
  --artifact fleurs-attribution=release/FLEURS-ATTRIBUTION.txt \
  --artifact fsboard-license=release/CC-BY-4.0.txt \
  --artifact fsboard-attribution=release/FSBOARD-ATTRIBUTION.txt \
  --output-directory release \
  --replace
```

Review both generated files. Then commit exactly those two paths as commit C:

```bash
test "$(git diff --name-only | LC_ALL=C sort)" = \
  "$(printf '%s\n' release/SHA256SUMS release/release-manifest.json)"
git add release/SHA256SUMS release/release-manifest.json
git commit -m "Seal UMI S1 baseline v0 metadata"
C="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD^)" = "$B"
```

Verify the artifacts and all three history relations from C:

```bash
uv run --frozen --extra dev python tools/release_artifacts.py \
  --verify release/release-manifest.json \
  --repository . \
  --release-git-revision "$C"
uv run --frozen --extra dev python tools/repo_guard.py
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

The verifier requires B to be C's sole parent and A to be B's sole parent. It also
requires B to differ from A only by the E2E evidence, C to differ from B only by the
two metadata files, every artifact to match its B blob, every license companion to
match its canonical B source, and the model runtime closure to match B.

## 6. Approval boundary

Give the project owner C, the manifest content digest, and the complete
`release/SHA256SUMS` for review. After explicit approval, create and push the immutable
tag and publish the exact fixed artifact set. Never publish the Docker image, private
E2E report, private evaluation reports, source data, MediaPipe task binary, secrets,
or wallet material.
