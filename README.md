# UMI S1 Reference Model

Status: `component_test_no_weight`; release candidate ready for owner approval

This repository packages the first runnable ASL-to-English reference model for UMI
miners. It gives miners a working implementation to run, inspect, and beat. Its
current accuracy is low, and UMI translation weights remain inactive.

Do not describe this release as activation evidence, interpreter-equivalent output,
or a guarantee of mining rewards. The release is useful for miner integration and
no-weight component tests.

## What it does

The reference backend accepts the raw MP4 bytes already verified by the UMI miner,
extracts MediaPipe Holistic landmarks in an isolated Linux/AMD64 container, converts
them to the fixed 1,184-value skeletal motion representation, and runs the compact S1
PyTorch translator. It returns at most eight whitespace-delimited words under the
validation-selected output policy.

The UMI miner timelock-encrypts and signs that response. The model backend never
handles wallet keys and never submits chain calls.

## Release contents

| Item | Distribution | License |
|---|---|---|
| Python runtime and Docker build source | Git repository | Apache-2.0 |
| Portable S1 model bundle | Attached release archive | CC BY-SA 4.0 |
| Linux/AMD64 extractor build source | Git repository; image built locally | Apache-2.0 plus upstream components |
| MediaPipe Holistic task model | Operator download from Google | Upstream terms |
| FLEURS-ASL and FSboard source data | Not distributed | Upstream CC licenses |

The exact artifact names, byte lengths, and SHA-256 digests are published in
`release/release-manifest.json` and `release/SHA256SUMS` when the release is sealed.

## Current quality

The selected epoch-20 FSboard-initialized model completed the frozen 40-epoch FLEURS
run. Under the exact single-reference WER adapter, its raw mean normalized score was
0.015885 on validation and 0.014881 on the test partition. The eight-word output cap,
selected from validation before test opening, raised the validation score to 0.044418.

These results are below UMI's provisional 0.10 utility floor. They show that the
training, export, extraction, and inference path works. They do not show positive
miner utility or satisfy any activation gate.

## Run it

Use [the miner runbook](docs/RUN_MINER.md). It builds the extractor locally, validates
its immutable image ID and installed packages, and binds that ID into a derived model
bundle before startup. A release operator should complete
[the release checklist](docs/RELEASE.md) first. The exact-source transfer procedure is
documented in [the staging procedure](docs/SOURCE_STAGING.md).

## Licenses and attribution

Repository code is licensed under Apache-2.0. Model weights and the portable model
bundle are licensed under CC BY-SA 4.0 because their lineage includes FLEURS-ASL.
FSboard is CC BY 4.0. Full legal text and source notices are in `licenses/`; required
attribution is in `NOTICE`.

Runtime dependency and local-image distribution notes are documented in
[the third-party inventory](docs/THIRD_PARTY.md).

No source videos, annotations, FSboard records, MediaPipe task binary, secrets, or
wallet material belong in this repository.
