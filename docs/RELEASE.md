# Release checklist

A release may be labeled `component_test_no_weight` only after every item below
passes. This checklist does not activate UMI translation weights.

## 1. Freeze authority

- The completed training report, selected model state, postprocessing selection, and
  final rights decision are canonical and content-addressed.
- The rights decision permits public weight redistribution under CC BY-SA 4.0 and
  forbids redistribution of source data and the MediaPipe task binary.
- The portable exporter consumes the final rights decision and records its digest in
  the inference identity.
- The model card reports the frozen validation and held-out results without claiming
  positive utility or production quality.

## 2. Freeze the complete runtime

- The portable identity names every transitive local inference/backend module and
  binds Python, PyTorch, NumPy, Safetensors, and RFC 8785 versions.
- `release/runtime-files.txt` matches the frozen public source closure.
- `tools/stage_runtime.py --check-only` succeeds against the frozen source snapshot.
- Every runtime-module SHA-256 in `inference-identity.json` matches the public file.
- Every preprocessing source digest matches the checked-in Docker worker, lock,
  mapping, and motion-conversion source.
- Ruff, the full public test suite, the repository guard, and `uv lock --check` pass.
- Loading the portable bundle requires the exact base inference revision.

Any closure or dependency change requires a new base export and a repeat of the real
end-to-end test.

## 3. Verify the local extractor workflow

The project does not distribute a Docker image archive.

- Build `docker/mediapipe-holistic/Dockerfile.amd64` with
  `python -m bitsign_motion.local_extractor_release` on a Linux/AMD64 Docker host.
- Require a new immutable `sha256:` image ID, `linux/amd64` platform, non-root user,
  exact dependency labels, every version in `requirements.amd64.lock`, a clean package
  consistency check, and FFmpeg `5.1.9-0+deb12u1`.
- Run `python -m bitsign_motion.local_bundle_rebind` against the published base ZIP
  and checksum.
- Verify that model, tokenizer, config, runtime, rights, postprocessing, source hashes,
  and claim boundary remain byte- or value-identical. Only
  `preprocessing.supported_oci_images["linux/amd64"]`, the resulting inference
  revision, and the bundle manifest may change.
- Reload the derived bundle using its exact new revision.
- Probe the backend with that bundle, image ID, and the separately downloaded task
  model. Require `component_test_no_weight`, an empty job directory, and no leftover
  extractor container.

No result from one host is evidence that another host's image is byte-equivalent.
Each operator binds its own immutable local image ID.

## 4. Run real inference and UMI response tests

- Download the MediaPipe task asset from the runbook and verify SHA-256
  `e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8`.
- Run an unmocked raw-video translation through the locally built image and derived
  portable bundle.
- Run the UMI release test that submits valid and invalid video requests, receives
  signed timelock-encrypted envelopes, decrypts only after the declared reveal round,
  validates the bounded hypothesis and explicit failure, and checks cleanup.

Create a clean release-test venv using Section 2 of `RUN_MINER.md`, but include the
reference repository's frozen `dev` extra when exporting its requirements. Keep the
UMI export at runtime-only. Install both exported files, add the same two path-only
`.pth` files, unset `PYTHONPATH`, and verify that `pytest`, `bitsign_motion`, and `umi`
import from that venv. The release-owner command is:

```bash
export BITSIGN_RUN_UMI_RELEASE_E2E=1
export BITSIGN_UMI_RELEASE_VIDEO=/absolute/path/to/eligible-release-test.mp4
export UMI_S1_BUNDLE=/absolute/path/to/derived-bundle
export UMI_S1_INFERENCE_REVISION=DERIVED_REVISION
export UMI_S1_EXTRACTOR_IMAGE=sha256:LOCAL_IMAGE_ID
export UMI_S1_EXTRACTOR_MODEL=/absolute/path/to/holistic_landmarker.task
export UMI_S1_EXTRACTOR_PLATFORM=linux/amd64
export UMI_S1_DOCKER_EXECUTABLE=/absolute/path/to/docker
export UMI_S1_TEMP_ROOT=/absolute/owner-only/path/to/s1-jobs
export UMI_S1_DEVICE=cpu
export UMI_S1_HARD_DEADLINE_SECONDS=120
deployment-tmp/release-e2e-venv/bin/python -m pytest -q \
  tests/test_umi_release_e2e.py
```

The venv must contain the exact reference-model and UMI locks and source checkouts
named by the release manifest. It must not contain an editable install of either
project. A skipped test is not a pass.

## 5. Seal the public model artifact

Package the final base bundle and generate model-only release metadata:

```bash
MODEL_BUNDLE=/absolute/path/to/portable-base-bundle
FINAL_RIGHTS_REVIEW=/absolute/path/to/final-rights-review.json

uv run --frozen --extra dev python tools/package_bundle.py \
  --bundle "$MODEL_BUNDLE" \
  --output release/umi-s1-baseline-v0-portable.zip

uv run --frozen --extra dev python tools/release_artifacts.py \
  --release-id umi-s1-baseline-v0 \
  --inference-revision "$(jq -r .inference_revision "$MODEL_BUNDLE/inference-identity.json")" \
  --rights-decision-sha256 "$(jq -r .content_sha256 "$FINAL_RIGHTS_REVIEW")" \
  --source-git-revision "$(git rev-parse HEAD)" \
  --umi-git-revision "$(git -C ../umi rev-parse HEAD)" \
  --artifact model=release/umi-s1-baseline-v0-portable.zip \
  --output-directory release

uv run --frozen --extra dev python tools/release_artifacts.py \
  --verify release/release-manifest.json
```

Review `release/release-manifest.json` and `release/SHA256SUMS`. The source revision
must name the clean commit containing the exact staged runtime and source-build tools.
The UMI revision must include its frozen `uv.lock`. Generate metadata only after the
source commit exists, then create one metadata-only commit whose sole parent is the
recorded source revision and whose only changed paths are
`release/release-manifest.json` and `release/SHA256SUMS`. Tag the metadata commit. The
verifier checks this history before release use. No Docker archive, Docker verification
record, task binary, or source data may appear in the release asset set.

## 6. Repository and approval boundary

Run `uv run --frozen --extra dev python tools/repo_guard.py`. It must find no source
media, annotations, task binary, model tensor payload, Docker archive, private license,
credentials, or file larger than the repository limit. The model ZIP belongs in
release hosting, with its checksum committed to Git.

Do not push a commit, create a tag, or upload an artifact until the project owner
approves the exact commit and checksum set.
