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
policy must be reviewed and copied to its fixed public filename before intake:

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

## 3. Bind the local extractor and run E2E

Build the Linux/AMD64 extractor from commit A as described in `docs/RUN_MINER.md`.
Read the base revision and archive digest from the staged intake:

```bash
BASE_ARCHIVE=release/umi-s1-public-finetune-v1-portable.zip
BASE_INFERENCE_REVISION="$(
  unzip -p "$BASE_ARCHIVE" inference-identity.json | jq -er .inference_revision
)"
BASE_ARCHIVE_SHA256="$(sha256sum "$BASE_ARCHIVE" | cut -d ' ' -f 1)"

uv run --frozen --extra dev python -m bitsign_motion.local_bundle_rebind \
  --base-archive "$BASE_ARCHIVE" \
  --base-sha256 "$BASE_ARCHIVE_SHA256" \
  --base-inference-revision "$BASE_INFERENCE_REVISION" \
  --build-record /absolute/owner-only/local-extractor.json \
  --docker "$(command -v docker)" \
  --output /absolute/owner-only/model-bundle
```

Run the clean Linux/AMD64 UMI request-to-reveal E2E test against commit A. Generate
the aggregate public E2E projection at the fixed path:

```text
release/umi-s1-public-finetune-v1-release-e2e-evidence.json
```

The E2E record must bind `A`, the public base inference revision, the locally derived
revision, the immutable extractor image ID, and the exact UMI commit. Commit only
that file as commit B.

Set this profile before running `tests/test_umi_release_e2e.py` through the locked
procedure in `docs/RELEASE.md`; it makes both the private run and public projection
use the public S1 E2E schemas and release identity:

```bash
export BITSIGN_UMI_RELEASE_PROFILE=public-s1-finetune/1
```

## 4. Generate commit C

Set the exact revisions and list the complete fixed release inventory. Additional
artifact arguments are rejected.

```bash
B="$(git rev-parse HEAD)"
UMI_GIT_REVISION="$(git -C ../umi rev-parse HEAD)"

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
C="$(git rev-parse HEAD)"
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
