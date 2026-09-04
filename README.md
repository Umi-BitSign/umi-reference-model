# UMI S1 Reference Model

Status: public no-weight bootstrap candidate (`umi-s1-public-finetune-v1-r2`)

This repository packages a compact ASL-to-English reference model and backend for
UMI miners. It gives miners a working baseline to inspect, operate, and improve.
UMI translation weights remain inactive.

S1 is an early, low-accuracy model. It is not an ASL interpreter, an accessibility
tool, activation evidence, a production service, or a guarantee of mining rewards.

## What it does

The backend accepts raw MP4 bytes already verified by the UMI miner. It extracts
MediaPipe Holistic landmarks in an isolated Linux/AMD64 container, converts them to
the fixed 1,184-value skeletal motion representation, and runs the compact S1 PyTorch
model. Decoding uses deterministic beam search with width 2, a 24-token ceiling,
cumulative log-probability, lexicographic tie-breaking, and a no-repeat trigram
constraint. The selected output policy returns at most eight whitespace-delimited
words.

The UMI miner timelock-encrypts and signs the response. The model backend does not
handle wallet keys or submit chain calls.

## Release contents

The sealed `umi-s1-public-finetune-v1-r2` release contains the unchanged v1-named
model artifacts below. The `r2` suffix identifies the superseding release seal and
its updated UMI integration evidence; it does not claim new model weights.

- `umi-s1-public-finetune-v1-portable.zip` (`model`);
- `umi-s1-public-finetune-v1-evidence.json` (`intake-evidence`);
- `umi-s1-public-finetune-v1-intake-policy.json` (`intake-policy`);
- `umi-s1-public-finetune-v1-release-e2e-evidence.json`
  (`release-e2e-evidence`);
- `LICENSE` (`code-license`) and `NOTICE` (`notice`);
- `CC-BY-SA-4.0.txt` (`model-license`); and
- one exact license and attribution companion for each source in the training
  lineage: 2M-Flores-ASL, FLEURS-ASL, FSboard, and Taskmaster-1.

`release/release-manifest.json` records each fixed artifact name, byte length, and
SHA-256 digest. `release/SHA256SUMS` covers the same closed inventory. The extractor
image and MediaPipe task model are deliberately not distributed.

## Evaluation boundary

The selected epoch is 3 from a completed six-epoch run. On the fixed FLEURS
validation diagnostic, the exact mean normalized score under the pinned WER adapter
was 0.052609 for real motion, 0.043834 for zero motion, and 0.046340 for
deterministically deranged motion. The real-motion result exceeded both controls by
the configured 0.001 minimum. It produced 280 distinct real-motion hypotheses across
285 items, with no empty or unknown-token output. This supports publishing a
motion-grounded bootstrap candidate. It does not make the model accurate enough for
use beyond that purpose.

The release ships aggregate-only validation evidence. It does not publish source
rows, references, predictions, token IDs, per-example edit distances, training
checkpoints, or raw video. No test or devtest inference was used for candidate
selection.

The public validation result is a development diagnostic, not an estimate of
real-world translation quality. It does not establish positive miner utility, UMI
activation readiness, accessibility value, production quality, or compliance with a
UMI activation gate. The Linux/AMD64 E2E record checks the request-to-reveal release
path, not translation quality.

## Run it

Use [the miner runbook](docs/RUN_MINER.md). It builds the extractor locally,
validates its immutable image ID and installed packages, then binds that ID into a
derived model bundle before startup. The release procedure is in
[the public S1 release guide](docs/PUBLIC_S1_RELEASE.md).

## Platform support

The released serving path requires a locally built Linux/AMD64 MediaPipe extractor.
CUDA is optional for the PyTorch model when the host supports it. A native iOS or
Core ML package is not part of this release. iOS and Android inference remain future
work and need their own deterministic preprocessing and evaluation evidence.

## Licenses and attribution

Repository code is Apache-2.0. Model weights and the portable bundle are CC BY-SA
4.0. The complete source lineage and required attribution are in `NOTICE` and the
sealed release companions.

Runtime dependency and local-image distribution notes are documented in
[the third-party inventory](docs/THIRD_PARTY.md).

No source videos, annotations, derived source records, MediaPipe task binary,
secrets, or wallet material belong in this repository or its release artifacts.
