# Community baseline public diagnostic, 2026-09-15

The installed community baseline is not qualified for open-competition launch.
Three sentences were selected before inference. One returned text with a
single-reference word error rate of 1.0; two returned no complete response at
the installed 120-second limit. This small functional diagnostic is not an
estimate of overall ASL accuracy.

| Annotated clip length | Observed request time | Result |
| --- | --- | --- |
| 4.69 seconds | 60.67 seconds | Text returned; 13 word edits / 13 reference words |
| 9.42 seconds | 120.16 seconds | No complete response; `IncompleteReadError` |
| 14.08 seconds | 120.16 seconds | No complete response; `IncompleteReadError` |

The last request followed a worker-discarding failure, so its time includes
any replacement startup. It is not an isolated warm-worker measurement.
Missing outputs were not assigned fabricated transcripts or error scores.
The Studio was shared with other work; these timings do not establish a
controlled performance comparison.

## Selection and limits

The clips and English references came from version 2 of
[Google AI FLEURS-ASL](https://www.kaggle.com/datasets/googleai/fleurs-asl).
Its CC BY-SA 4.0 notices and license were retained with the diagnostic files.
No source videos, annotations or predictions are added to this repository.

Selection used `zs_split=test` and `devtest_` source IDs. In each duration bin
(0,5], (5,10] and (10,15] seconds, it selected the smallest SHA-256 of a fixed
seed plus the example ID. The bins contained 13, 229 and 338 candidates.
All three selected examples happen to use signer 1. Predictions and reference
difficulty were not inspected during selection, and failed cases were not
replaced. Source-annotated intervals were retained in full.

The pre-inference plan's SHA-256 is
`d825302b7a819bd6c16a5dd27c3425f63bd5026fc8357d02946d375a6c32732f`.
Each receipt binds the selected case, source and derived-video digests, runtime
identity, observed result and original reference. The tested local native runtime
revision was
`2cef63ca67096aeffac8bd2d26099a1840f55662af76554f39be4548fcd3cc33`.
It used MPS, four CPU threads, one model slot and the unchanged 120-second limit.

Only video bytes and request metadata entered the isolated model process.
The driver retained the authentic single reference and used UMI's text
normalization and edit distance. It did not duplicate labels to satisfy the
separate three-to-five-reference production scorer or change that scorer.

This public dataset is not a protected reward pool. Training exposure of the
supplied model remains unknown. These results create no evaluation signature,
model-rights approval, contributor attribution or reward authorization.

## Integration checks

Restoring the supplied video reader and landmark preprocessing for the same
frozen 4.69-second clip produced exactly the same incorrect text. All 113 frames
had detected landmarks. That comparison retained the model assets and generation
settings; it does not prove equivalence on every clip or exclude other
integration defects. Its longer diagnostic allowance did not change the serving
deadline.

A separate read-only comparison against the package's declared
[upstream translation source](https://huggingface.co/spaces/ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText/blob/69d3d77aa4a4fec89048f5417d44fa717e36f6f8/inference.py)
found identical syntax trees for all five translation classes after removing
docstrings. The SHuBERT configuration also matched; its model class differed only
in `compute_var`, where the supplied version selects the input device instead
of assuming CUDA. The inspected translation path does not call that helper.

All 482 checkpoint tensor names, shapes and dtypes matched a model skeleton
constructed from the supplied configuration. The skeleton used PyTorch's meta
device, with one 1,024-byte CPU mask constant. A diagnostic-only patch skipped
Fairseq's random-value initializer because it cannot copy meta tensors to CPU.
No checkpoint tensors were loaded into that skeleton and no inference ran.
The two serialized aliases of the decoder embedding were also byte-identical.
These checks cover structural compatibility, not actual loading behavior or
correct predictions. Serving code and model bytes were unchanged.

## Remaining qualification

Investigate translation quality and demonstrate completion across supported
clip lengths under the actual serving deadline before announcing this model as
a qualified baseline. Preserve the failed cases when testing a proposed fix.
Do not select replacement evaluation cases based on which ones it translates
well. Protected evaluation data and contribution-rights review remain separate
requirements.
