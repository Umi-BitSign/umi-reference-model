# Community candidate qualification, 2026-09-13

One Linux ARM64 functional case has passed through UMI's real
`preserve_bundle` and `execute_offline_case` path. The input was a two-second,
256x256, 30 fps synthetic `testsrc2` clip encoded as H.264/yuv420p, with no audio
or person. Its output cannot measure sign-language accuracy.

The case completed in 231,638 ms with status `ok`, exit code 0 and bounded UTF-8
output. Cold initialization is included. The test had four CPUs, 12 GiB memory,
512 MiB combined temporary storage, 128 PIDs and a 600,000 ms outer deadline.
No network, wallet, reference text or writable model/input mount was available.

Identities:

- Source ZIP SHA-256: `f78979599486456e06e7886b126169f8d45e5e1e7d16d7a2b617648615eef3d4`.
- Preserved UMI bundle: `945446b1c531255bef3791d1c6c34c3d386037a5f8b9d760b9112ccffa3e5e1e`.
- Local ARM64 OCI manifest: `a1a06d7a7a0d47298c7bddc89c5d8343d0c0f9d13954b0012ac24c5263367d91`.
- Runtime v2 digest: `584d52fd6263d16b3c32cbd5d054d270fbb38e30f0ce336b28c1451698aa26d4`.
- Synthetic video SHA-256: `6cfabd972dd258519e65f2bfcc90d2f87710227421ad467fbfe58c773f6661b4`.
- Unsigned expired smoke-policy digest: `438877e8995d4661f9335e6939e7ca21d323c5a72c99e649022d024eec605d6e`.

The first runtime-v1 attempt failed during Fairseq import because its
multiprocessing lock required writable `/dev/shm`. Runtime v2 divides the existing
scratch budget between private shared memory and `/tmp`. Separate native tests
confirmed lock/shared-buffer access, the combined storage limit, unchanged v1
behavior and read-only model/input mounts.

This is one operator's local execution evidence. It grants no rights approval,
independent-evaluation quorum, contributor attribution or chain-write authority.
The image name is local and is not a published registry download.

Before a production baseline or promotion, complete protected ASL evaluation,
base-model/data provenance review, workload and concurrency measurements,
independently administered evaluation and approved signed policy inputs.
