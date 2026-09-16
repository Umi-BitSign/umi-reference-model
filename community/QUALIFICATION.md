# Community candidate qualification

## Current status, 2026-09-16

The native evaluator completed six previously exposed rehearsal clips using
preserved bundle
`ce459641c180c680aae32009052985165fa8bbffd165b0dca05d2dade0e619ed` and runtime
`888c6a3d2c95b3539dacf24f10dea52cfa1e909bae608922b6ac4a744b843a54`.
All six returned `ok` in 16.016, 19.922, 21.553, 29.052, 32.025 and 44.532
seconds, within the unchanged 120-second inference limit.

Replay of the signed paired result verified the evaluator's retained evidence.
The baseline scored 0.246914 for fingerspelling, 0.215873 for continuous signing,
and 0.223036 aggregate under that rehearsal policy. These are normalized
CER/WER-derived scores on six exposed examples, not accuracy percentages or
evidence of performance on unseen data.

The endpoint side returned six signed miner errors. Its retained resource
journal showed `video_fetch_failed` for every clip: the rehearsal video
delivery had expired before dispatch. Inference was not reached. The failed
responses and original signed round remain unchanged. Its zero endpoint score
prevents a settlement with a qualifying endpoint recipient.

A subsequent serving preflight used fresh capability URLs for the same clips.
The production downloader verified their bytes over HTTPS, and the running
miner's policy-bound Unix-socket translator returned nonempty bounded text for
all six. Inference took 32.542, 6.001, 14.079, 3.373, 9.686 and 12.139 seconds
in that test's order. This preflight did not sign evaluator evidence or submit
weights. A fresh signed round is still required to establish the complete
endpoint-to-settlement path. No unused private holdout cases were exercised.

The deployed service and sidecar source files match the reboot-recovery and
slot-rewarming changes on main. The serving preflight does not establish broad
translation accuracy, full-load capacity, or completion of the launch gates.
The earlier failures below remain part of the qualification record.

## Earlier status, 2026-09-15

This candidate remains unqualified for open-competition launch. The
[public sentence diagnostic](PUBLIC_DIAGNOSTIC.md) returned one incorrect
translation and two incomplete responses at the installed deadline. Static
source and checkpoint checks found no architecture/name/shape mismatch;
they do not establish correct inference or useful ASL accuracy.

The Linux results below record earlier execution checks. They do not override
these quality and capacity failures.

## Linux ARM64 execution checks, 2026-09-13

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

Two additional failure probes used the same preserved real model and runtime v2.
A three-second deadline returned `reason=deadline` after 3,015 ms and removed its
case container. Cancelling an evaluation after its case container had started
also removed that container. Neither probe executed a second copy or touched any
validator process. These checks cover the evaluator's deadline and cooperative
host cancellation paths; abrupt host power loss was not tested.

This is one operator's local execution evidence. It grants no rights approval,
independent-evaluation quorum, contributor attribution or chain-write authority.
The image name is local and is not a published registry download.

The original ZIP is retained in UMI's private `umi-model-candidates` R2 bucket
as eleven checksum-addressed chunks and a reconstruction manifest. Each chunk
was downloaded, and the reconstructed 2,934,700,086-byte ZIP matched the original
SHA-256 above. Public bucket access is disabled and no custom domain is attached.
This backup does not publish a baseline download or approve its rights.

Before a production baseline or promotion, complete protected ASL evaluation,
base-model/data provenance review, workload and concurrency measurements,
independently administered evaluation and approved signed policy inputs.
