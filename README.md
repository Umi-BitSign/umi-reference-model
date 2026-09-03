# UMI S1 Reference Model

Status: `component_test_no_weight`; low-accuracy integration fixture

This repository packages a compact ASL-to-English model and a reference backend for
testing the UMI miner interface. It gives miners a common implementation to run,
inspect, and replace. The model is not a usable ASL translator, an accessibility
tool, activation evidence, or a guarantee of mining rewards. UMI translation weights
remain inactive.

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

## Sealed release set

The fixed artifact set contains:

- `umi-s1-baseline-v0-portable.zip` (`model`);
- `umi-s1-baseline-v0-selection-ledger.json` (`selection-ledger`);
- `umi-s1-baseline-v0-motion-ablation-evidence.json`
  (`motion-ablation-evidence`);
- `umi-s1-baseline-v0-rights-decision.json` (`rights-evidence`);
- `umi-s1-baseline-v0-release-e2e-evidence.json` (`release-e2e-evidence`);
- `LICENSE` (`code-license`) and `NOTICE` (`notice`);
- `CC-BY-SA-4.0.txt` (`model-license`) and `FLEURS-ATTRIBUTION.txt`
  (`fleurs-attribution`);
- `CC-BY-4.0.txt` (`fsboard-license`) and `FSBOARD-ATTRIBUTION.txt`
  (`fsboard-attribution`).

`release/release-manifest.json` records each artifact's fixed name, byte length, and
SHA-256 digest. `release/SHA256SUMS` covers the same complete set. The extractor image
and MediaPipe task model are not distributed.

## Measured quality

The selected epoch-20 FSboard-initialized state completed the frozen 40-epoch FLEURS
run. Under the exact single-reference WER adapter, its raw mean normalized score was
0.015885 on validation and 0.014881 on the already-opened test partition. The
validation-selected eight-word cap raised the original greedy-decoder validation
score to 0.044418. On the same fixed 285-sample validation set, the later
beam-2/24-token/no-repeat-trigram policy scored 0.059346. This decoder choice was
made after the test partition had been opened; no beam-decoder test score is reported
as untouched evidence.

The aggregate motion diagnostic scored real motion at 0.059346, zero motion at
0.075531, and deterministically permuted motion at 0.053052. Zero motion outscoring
real motion means useful motion grounding has not been established. The public
selection and ablation records contain aggregate results and private-report digests,
not per-sample source rows, references, tensor identities, token IDs, hypotheses, or
edit distances.

The selection ledger names both the revision used for the aggregate quality run and
the source-rebound release revision. The release keeps the selected model, config,
tokenizer, decoder, and materialized-motion evaluation semantics, but it receives a
new identity because its checked-in runtime source closure changed. The ledger does
not present that transfer as a fresh quality run on the release revision.

The real-motion validation score is below UMI's provisional 0.10 utility floor.
These measurements do not establish positive miner utility, expected quality,
production quality, accessibility value, or compliance with a UMI activation gate.
The E2E record tests the release plumbing on one Linux/AMD64 host and reports no
translation-quality result.

## Run it

Use [the miner runbook](docs/RUN_MINER.md). It builds the extractor locally, validates
its immutable image ID and installed packages, and binds that ID into a derived model
bundle before startup. Release owners should complete
[the release checklist](docs/RELEASE.md). The exact-source transfer procedure is in
[the staging procedure](docs/SOURCE_STAGING.md).

## Licenses and attribution

Repository code is licensed under Apache-2.0. Model weights and the portable model
bundle are licensed under CC BY-SA 4.0 because their lineage includes FLEURS-ASL.
FSboard is CC BY 4.0. Full license texts and source notices are included in the sealed
artifact set and under `licenses/`; required attribution is also in `NOTICE`.

Runtime dependency and local-image distribution notes are documented in
[the third-party inventory](docs/THIRD_PARTY.md).

No source videos, annotations, FSboard records, MediaPipe task binary, secrets, or
wallet material belong in this repository or its release artifacts.
