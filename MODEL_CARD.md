# Model card: UMI S1 baseline

## Model identity

S1 is a compact skeletal-motion-to-English model released as a low-accuracy UMI
integration fixture and miner replacement target. Each sealed release publishes its
base inference revision in the portable bundle's `inference-identity.json`. The
release manifest binds that revision to the model archive, source revision, aggregate
evidence, UMI revision, and rights decision. Each operator derives another inference
revision that binds the immutable ID of the extractor image built on that host.

Release status: `component_test_no_weight`

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
are not assumed to have identical bytes. Linux/AMD64 and Linux/ARM64 landmark tensors
are separately identified and are not bit-equivalent.

## Training lineage

The selected arm initializes the S1 encoder from FSboard fingerspelling pretraining,
then trains the ASL-to-English model on FLEURS-ASL. The frozen run used seed 17,
completed 40 epochs, selected epoch 20 from validation, and opened the held-out test
partition once after model and postprocessing selection.

The public release contains weights and inference metadata. It contains no training
videos, source annotations, derived source records, signer identifiers, or MediaPipe
task binary.

## Evaluation

| Evaluation | Mean normalized score |
|---|---:|
| Frozen validation, raw decoder | 0.015885 |
| Held-out test, raw decoder | 0.014881 |
| Frozen validation, original greedy decoder plus selected eight-word cap | 0.044418 |
| Fixed validation, beam-2/24-token/no-repeat-trigram decoder plus selected eight-word cap | 0.059346 |
| Fixed validation, zero-motion ablation with release decoder | 0.075531 |
| Fixed validation, permuted-motion ablation with release decoder | 0.053052 |

Scores use exact single-reference WER through the pinned UMI normalization adapter.
The output cap was selected using validation predictions before test opening. The
later beam-decoder choice used validation behavior after the original release had
already opened the test partition. No beam-decoder test score is reported as untouched
evidence.

The public selection ledger and motion-ablation record are aggregate-only projections
of bound private reports. They contain no per-sample source rows, references, tensor
identities, token IDs, hypotheses, or edit distances. Independent aggregate
reproduction requires legally obtained source data and the bound evaluators.

The selection ledger separately records the predecessor revision used for the
aggregate quality run and the source-rebound release revision. The latter preserves
the selected model, config, tokenizer, decoder, and materialized-motion evaluation
semantics while rebinding the checked-in runtime source closure. This is an explicit
evidence transfer, not a new quality evaluation.

Zero motion scored above real motion. The model reacts to motion because real motion
scored above the deterministic permutation, but this diagnostic does not establish
useful motion grounding. The real-motion score is below UMI's provisional 0.10 utility
floor. These results are outside the independent WER/CER validity study, public
calibration, ten-tempo agreement trial, and 30-day activation soak required by the UMI
specification.

## Intended use

This fixture supports:

- integration testing of the asynchronous UMI miner backend;
- a common replacement target for miners developing other models;
- measurement of raw-video extraction, model latency, resource use, response sealing,
  and post-reveal decryption.

It has no validated production, accessibility, interpreter, mobile, medical, legal,
financial, emergency, employment, or safety use.

## Limitations and excluded use

Outputs are often wrong, repetitive, or incomplete. The eight-word cap can truncate
meaning. Landmark extraction can fail under occlusion, unusual framing, motion blur,
or unsupported media. Skeletal features discard appearance cues that may matter to
translation. The motion ablation shows that the released state relies too heavily on
its learned language prior.

Do not use this model as an ASL interpreter or accessibility tool, or as input to a
decision affecting a person's health, legal rights, finances, safety, employment, or
access to services. Any displayed output must be identified as machine-generated and
unreliable.

## Licenses

The model weights and portable bundle are licensed under CC BY-SA 4.0. Training
lineage includes FLEURS-ASL under CC BY-SA 4.0 and FSboard under CC BY 4.0. Required
license texts, attributions, and modification notices are shipped as separate sealed
release artifacts. Runtime code is Apache-2.0.

The MediaPipe Holistic task binary is not redistributed. Operators download it from
the pinned Google storage URL and verify its SHA-256 digest.
