# Model card: UMI S1 public bootstrap

## Model identity

`umi-s1-public-finetune-v1` is a compact skeletal-motion-to-English public bootstrap
for UMI miners. It is a starter model and replacement target. Its performance
evidence is limited to the aggregate validation record below. The sealed archive
binds the model, tokenizer, configuration, preprocessing
contract, training-lineage review, and inference identity. Each operator derives a
separate inference identity by binding the exact local extractor image it built.

Release class: no-weight public bootstrap.

## Architecture and input

S1 consumes up to 120 frames sampled at 8 Hz. Each frame contains 1,184 normalized
values derived from face, pose, and hand landmarks, plus a frame mask. The tokenizer
contains 4,096 pieces. The decoder contract permits up to 128 tokens. This release
uses deterministic width-2 beam search with a 24-token ceiling, cumulative
log-probability, lexicographic token-sequence tie-breaking, PAD/BOS suppression, and
a no-repeat trigram constraint. It then applies the validation-selected cap of eight
whitespace-delimited output words.

The reference miner uses a locally built Linux/AMD64 MediaPipe Holistic extractor
image. The derived bundle binds its exact local image ID. Independently built images
are not assumed to have identical bytes. Linux/AMD64 hosts run PyTorch on CPU or
CUDA. Apple Silicon macOS hosts use the same Linux/AMD64 worker through Docker
Desktop and run PyTorch natively on MPS or CPU. The repository's older Linux/ARM64
worker does not implement the release miner's whole-video contract.

## Training lineage

The public training lineage includes 2M-Flores-ASL, FLEURS-ASL, FSboard, and
Taskmaster-1. The completed run selected epoch 3 of 6 on FLEURS validation. It made
no test or devtest inference call.

The public release contains weights and inference metadata. It contains no training
videos, source annotations, derived source records, signer identifiers, references,
or MediaPipe task binary.

## Evaluation

| Fixed FLEURS validation condition | Mean normalized score |
|---|---:|
| Selected epoch 3, real motion | 0.052609 |
| Selected epoch 3, zero-motion control | 0.043834 |
| Selected epoch 3, deterministically deranged-motion control | 0.046340 |

Scores are exact mean normalized scores under the pinned UMI WER adapter. The
selected real-motion score exceeded each control by at least the configured 0.001
minimum. It produced 280 distinct real-motion hypotheses across 285 items, with no
empty or unknown-token output. Candidate selection used no test or devtest inference.

The public intake evidence is an aggregate-only projection of bound private reports.
It contains no per-sample source rows, references, tensor identities, token IDs,
hypotheses, or edit distances. Independent aggregate reproduction requires legally
obtained source data and the bound evaluators.

The result is a development diagnostic. It is not an untouched confirmation result,
an estimate of real-world accuracy, an accessibility certification, proof of
interpreter equivalence, or evidence that UMI translation weights should activate.

## Intended use

This public bootstrap supports:

- integration testing of the asynchronous UMI miner backend;
- a common replacement target for miners developing other models;
- measurement of raw-video extraction, model latency, resource use, response sealing,
  and post-reveal decryption.

It has no validated production, accessibility, interpreter, mobile, medical, legal,
financial, emergency, employment, safety, or high-consequence use.

## Limitations and excluded use

Outputs are often wrong, repetitive, or incomplete. The eight-word cap can truncate
meaning. Landmark extraction can fail under occlusion, unusual framing, motion blur,
or unsupported media. Skeletal features discard appearance and contextual cues that
may matter to translation.

Do not use this model as an ASL interpreter or accessibility tool, or as input to a
decision affecting a person's health, legal rights, finances, safety, employment, or
access to services. Any displayed output must be identified as machine-generated and
unreliable.

## Licenses

The model weights and portable bundle are licensed under CC BY-SA 4.0. Training
lineage includes 2M-Flores-ASL, FLEURS-ASL, FSboard, and Taskmaster-1. Required
license texts, attributions, and modification notices are shipped as separate sealed
release artifacts. Runtime code is Apache-2.0.

The MediaPipe Holistic task binary is not redistributed. Operators download it from
the pinned Google storage URL and verify its SHA-256 digest.

## Platform scope

The published runtime supports Linux/AMD64 extraction on Linux/AMD64 and Apple
Silicon macOS hosts. CUDA is optional on Linux. MPS and CPU are supported on macOS,
but extraction remains an emulated Linux/AMD64 Docker workload. macOS deployment
evidence is additive and does not alter the sealed r2 artifacts or establish burst
capacity. Native iOS, Core ML, and Android packages are not included or validated. A
future mobile release must establish a deterministic mobile preprocessing contract
and evaluate it independently.
