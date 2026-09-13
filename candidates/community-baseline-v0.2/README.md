# Candidate integration status

This directory adds a separately supplied inference candidate; it does not replace
S1 or activate a competition release. Weight and task binaries are intentionally
absent from Git. This is not yet integrated with the repository's S1 backend or
qualified for Linux deployment.

## Install the separately supplied model assets

Obtain `umi-community-baseline-v0.2.zip` from the maintainer handoff. Verify its
SHA-256 against `MODEL_ARTIFACT.json` before extracting it into a separate directory.
Copy the extracted `models/` directory into this directory, preserving its layout.
No private access URL or credential is required by the code or committed here.

Run `python3 verify_model_assets.py` from this directory to verify every model file
before following the standalone instructions below. The manifest contains hashes
of inference assets only, not training data. The existing repository S1 command
and release are unchanged. Automated installation cannot retrieve the private
archive; maintainers must supply it or publish an appropriately licensed artifact.

# UMI Community Baseline v0.2

Standalone SHuBERT/ByT5 ASL video-to-English inference. This package contains the
inference tensors, tokenizer, feature extraction and model implementation. It has
no registration, serving announcement, wallet, database or subnet configuration.

## Run

Use an Apple Silicon Mac with Python 3.10 and sufficient available memory
(36 GB host profile tested). From the extracted directory:

```bash
bash run.sh input.mp4
```

The first invocation installs pinned public Python dependencies into `.runtime`.
Installation requires internet access. Subsequent inference uses the included
models offline. Standard output is one JSON object containing English text and
elapsed seconds; diagnostic output goes to standard error. Pass `--device cpu`
to select CPU explicitly. CPU performance and Linux deployment are not qualified
by the supplied Apple Silicon smoke check.

The implementation decodes the supplied video and uses hand/face features and body
pose before the translation decoder. Decoding uses five beams and a 2048-token
maximum. It returns the decoder text directly, without an English rewrite layer.
Very long videos can exceed available memory; this is a serial research baseline,
not a production admission controller or a throughput guarantee.

The functional check demonstrates execution, not translation accuracy or an
improvement over another model. No benchmark scores, examples, source videos,
training state or evaluation annotations are distributed. No comparison claim is
made. A caller integrating this model with UMI must separately implement the
official model interface, cancellation and deployment requirements.

`MODEL_IDENTITY.json` identifies the release. `SHA256SUMS` and `FILE_LIST.json`
enumerate its files. Third-party terms and modifications are recorded in
`PROVENANCE.md` and `licenses/`. Read them before redistribution.
