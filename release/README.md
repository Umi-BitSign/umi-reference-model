# Release metadata and artifacts

The sealed `umi-s1-baseline-v0` set has eleven fixed artifacts:

| Label | File |
|---|---|
| `model` | `umi-s1-baseline-v0-portable.zip` |
| `selection-ledger` | `umi-s1-baseline-v0-selection-ledger.json` |
| `motion-ablation-evidence` | `umi-s1-baseline-v0-motion-ablation-evidence.json` |
| `rights-evidence` | `umi-s1-baseline-v0-rights-decision.json` |
| `release-e2e-evidence` | `umi-s1-baseline-v0-release-e2e-evidence.json` |
| `code-license` | `LICENSE` |
| `notice` | `NOTICE` |
| `model-license` | `CC-BY-SA-4.0.txt` |
| `fleurs-attribution` | `FLEURS-ATTRIBUTION.txt` |
| `fsboard-license` | `CC-BY-4.0.txt` |
| `fsboard-attribution` | `FSBOARD-ATTRIBUTION.txt` |

`release-manifest.json` binds every file to the base inference revision, source
commit, UMI commit, rights decision, and aggregate evidence content digest.
`SHA256SUMS` covers exactly the same eleven files. `runtime-files.txt` is the public
source staging allowlist.

The JSON evidence is aggregate-only. It contains no per-sample source rows,
references, tensor identities, token IDs, hypotheses, or edit distances. Row-level
evaluation evidence must not be published.

The model is a low-accuracy integration fixture. Zero motion outscored real motion in
the public ablation, so useful motion grounding has not been established. The E2E
record covers the functional release path on one Linux/AMD64 host and reports no
translation-quality result.

Release history is fixed as source commit A, E2E-evidence-only commit B, then
metadata-only commit C. The tag points to C. Generate and verify C with
`tools/release_artifacts.py` after completing `docs/RELEASE.md`; do not hand-edit the
manifest or checksum file.

No Docker image archive, private report, source data, MediaPipe task binary, secret,
or wallet material belongs in this directory.
