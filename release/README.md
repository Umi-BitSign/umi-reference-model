# Release metadata and artifacts

The sealed `umi-s1-public-finetune-v1` set has fifteen fixed artifacts:

| Label | File |
|---|---|
| `model` | `umi-s1-public-finetune-v1-portable.zip` |
| `intake-evidence` | `umi-s1-public-finetune-v1-evidence.json` |
| `intake-policy` | `umi-s1-public-finetune-v1-intake-policy.json` |
| `release-e2e-evidence` | `umi-s1-public-finetune-v1-release-e2e-evidence.json` |
| `code-license` | `LICENSE` |
| `notice` | `NOTICE` |
| `model-license` | `CC-BY-SA-4.0.txt` |
| `two-m-flores-license` | `2M-FLORES-ASL-LICENSE.txt` |
| `two-m-flores-attribution` | `2M-FLORES-ASL-ATTRIBUTION.txt` |
| `fleurs-license` | `FLEURS-ASL-LICENSE.txt` |
| `fleurs-attribution` | `FLEURS-ASL-ATTRIBUTION.txt` |
| `fsboard-license` | `FSBOARD-LICENSE.txt` |
| `fsboard-attribution` | `FSBOARD-ATTRIBUTION.txt` |
| `taskmaster-license` | `TASKMASTER-LICENSE.txt` |
| `taskmaster-attribution` | `TASKMASTER-ATTRIBUTION.txt` |

`release-manifest.json` binds the closed inventory to the source commit, exact UMI
commit, portable inference identity, intake policy, and public E2E record.
`SHA256SUMS` covers the same set. `runtime-files.txt` is the reviewed runtime-source
allowlist.

The JSON evidence is aggregate-only. It contains no source rows, references,
predictions, token IDs, per-example edit distances, raw video, or training
checkpoints.

This is a low-accuracy public bootstrap for miners. Its release E2E evidence covers
one Linux/AMD64 request-to-reveal path and reports no translation-quality result. It
does not establish production quality, accessibility value, interpreter equivalence,
or UMI weight-activation readiness.

Release history is fixed as source commit A, E2E-evidence-only commit B, then
metadata-only commit C. The tag points to C. Generate and verify C with
`tools/release_artifacts.py` after completing `docs/PUBLIC_S1_RELEASE.md`; do not
hand-edit the manifest or checksum file.

No Docker image archive, private report, source data, MediaPipe task binary, secret,
or wallet material belongs in this directory.
