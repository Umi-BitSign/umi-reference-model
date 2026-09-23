# Community model service

`service.py` is the process entry point for the isolated community-model sidecar.
It loads a reviewed local configuration, holds an exclusive lock before starting
any workers, and handles SIGTERM/SIGINT through the sidecar shutdown path. Run it
under the host's service manager. It does not contain a wallet or install the
miner HTTP process.

The model process can run locally before competition activation. Use the explicit
standby profile below until its actual scoring policy is ready. Do not generate
test keys or test policy hashes for a persistent deployment.

## Configuration

The input is canonical JSON, at most 256 KiB, in a regular owner-private file.
Use its SHA-256 from the reviewed local deployment inputs:

```sh
/absolute/path/to/supervisor-python -B -s /absolute/path/to/community/service.py \
  --config /absolute/private/model-service.json \
  --expected-config-sha256 REVIEWED_CONFIG_SHA256
```

The supervisor environment must contain the reviewed UMI `model_sidecar` helper.
It does not need Torch. Worker environments are explicit and do not inherit the
supervisor's environment.

The exact top-level fields are:

- `schema`: `umi-community-model-service/1`.
- `socket_path`: absolute path under an owner-private mode-0700 directory.
- `scoring_policy_sha256`: the actual policy identity used by the miner HTTP
  process, not a hash invented by this service.
- `validator_slot_count`: the actual policy's validator count.
- `workers`: one entry per enforced model slot, with equal slots per validator.

Each worker declares exactly `command`, `environment`, `cwd`, `model_revision`,
`startup_seconds`, `inference_seconds`, and `scratch_directory`, matching
`WarmModelProcess`. Paths and commands must come from the reviewed native
deployment. Include its sandbox and resource limits in `command`; this entry
point does not create them. Every worker needs a separate, empty mode-0700 scratch
directory. Worker runtime verification still happens before readiness.

The local configuration hash does not replace policy signatures, model rights
approval, measured capacity, or the worker's artifact verification. It prevents
the service manager from silently starting different local inputs.

## Linux CPU worker

On Linux x86_64, use `cpu_worker.py` behind this supervisor. It keeps one
`SHuBERTInferenceRuntime(device="cpu")` loaded per worker and speaks the same
bounded `worker_transport` protocol as the Metal worker. The supervisor stays
in the UMI Python 3.12 environment; the model runs in a separate Python 3.10
environment with the community baseline's dependencies. Linux miners do not
need to write their own worker entrypoint.

Use the [verified CPU bundle importer](README.md#verify-and-stage). This selects
the original CPU bundle, including its original preprocessing; it does not
silently substitute the later Mac adapters. Install its `model/requirements.txt`
in a dedicated Python 3.10 environment, using pip 24.0 for the supplied OmegaConf
metadata as in [the existing CPU image recipe](Dockerfile). Use the CPU builds of
the pinned Torch packages. Preserve the package licenses and provenance.

The deployment needs separate canonical absolute directories for worker code,
the Python 3.10 installation, its environment, the imported bundle, and scratch.
They must belong to the dedicated model-service user and must not overlap.
Use an operator-owned standalone Python installation: a system `/usr` tree is
not a worker-owned Python root. Copy these five files into the code directory:
`cpu_worker.py`, `cpu_runtime.py`, `native_runtime.py`, `bundle_verification.py`
and `worker_transport.py`. `native_runtime.py` supplies inventory helpers only;
this path imports no Metal worker or macOS overlay. Copy files rather than
hardlinking them, and keep group/other write permissions disabled.

Seal the installed runtime after dependency installation, before starting it.
Replace the paths and the imported bundle identity below with the verified
local values. The output manifest must be outside the inventoried directories:

```sh
/ABSOLUTE/ENVIRONMENT/bin/python -B -s /ABSOLUTE/CODE/cpu_runtime.py \
  --code /ABSOLUTE/CODE \
  --environment /ABSOLUTE/ENVIRONMENT \
  --python /ABSOLUTE/PYTHON310 \
  --bundle /ABSOLUTE/BUNDLE \
  --expected-bundle-sha256 IMPORTED_MODEL_BUNDLE_SHA256 \
  --output /ABSOLUTE/PRIVATE/cpu-runtime.json
```

The printed `model_revision` binds the bundle, code, Python installation,
installed packages and CPU execution settings. Use it in both the service
configuration and miner configuration. Keep the manifest owner-private and
read-only. A changed file, execution setting or bundle requires a new reviewed
manifest and revision. This is a local deployment identity, not a signed UMI
release or a claim of reproducible packages. The inventory has a 4 GiB ceiling
for code, environment and Python combined, excluding the separate model bundle.

The worker invocation is:

```sh
/ABSOLUTE/ENVIRONMENT/bin/python -B -s /ABSOLUTE/CODE/cpu_worker.py \
  --environment-root /ABSOLUTE/ENVIRONMENT \
  --python-root /ABSOLUTE/PYTHON310 \
  --bundle /ABSOLUTE/BUNDLE \
  --runtime-manifest /ABSOLUTE/PRIVATE/cpu-runtime.json \
  --model-revision PRINTED_MODEL_REVISION \
  --scratch /ABSOLUTE/SLOT_SCRATCH \
  --deny-read-marker /ABSOLUTE/OUTSIDE/harmless-marker
```

Run that invocation inside a reviewed Linux sandbox in each worker's `command`,
not directly on the host. For example, a Bubblewrap launcher should use
`--unshare-all --die-with-parent --new-session --cap-drop ALL`, expose only the
interpreter, required system libraries, worker code, bundle and manifest as
read-only, and mount only that slot's scratch as writable. Preserve the absolute
paths above inside the sandbox. Provide private `/proc`, `/dev` and `/dev/shm`;
Fairseq needs POSIX semaphores. Hide home directories, wallets, credentials and
host service sockets. Create the harmless marker outside those mounts before
launching; it must be unreadable inside. Do not bind the host root into the
sandbox. The worker checks read-only mounts, the hidden marker and blocked
outbound access before loading model code. These probes do not replace reviewing
the launcher and its mount list. See [Bubblewrap's sandbox documentation](https://github.com/containers/bubblewrap#sandboxing).

On Ubuntu 24.04, the operator must also have the distribution's Bubblewrap
AppArmor user-namespace profile installed and active. An error such as
`loopback: Failed RTM_NEWADDR: Operation not permitted` occurs before the worker
starts. Follow [Ubuntu's AppArmor guidance](https://discourse.ubuntu.com/t/understanding-apparmor-user-namespace-restriction/58007)
for the host; do not remove network isolation to work around it.

Enforce memory, CPU, task and scratch limits in the service manager/container as
well. Start with a measured slot budget; the existing cold CPU rehearsal used
four CPUs and 12 GiB, but it did not qualify warm serving capacity. Each worker
uses four CPU threads and one inference at a time. Give every configured slot
its own empty, mode-0700 scratch directory and enough aggregate resources.
`startup_seconds` and `inference_seconds` use the existing transport's `(0, 600]`
range; choose them from measurements within the miner's signed request budget.
A queued request's deadline includes its wait. More queued work cannot extend
that signed request budget.

Put the sandbox launcher plus the invocation above in `command`, an explicit
credential-free environment in `environment`, the code directory in `cwd`, and
the sealed identity in `model_revision`. The remaining fields are the same as
in [Configuration](#configuration). Use standby first for a real clip and a
restart check, then the actual policy and required validator slot count. Verify
capacity with representative clip lengths before registering this deployment
as ready. CPU workers propagate inference failures; they never invent text.

The tests cover the worker protocol, CPU selection, artifact drift, private
scratch cleanup, loading failure, deadline cancellation and replacement.
Linux CI also checks the sandbox probes using Bubblewrap on Python 3.10 and
3.12. Model loading is stubbed in these tests: they do not establish real-model
latency, translation quality or competition eligibility.

## Local baseline standby

For local baseline comparisons, use `umi-community-model-standby/1` with
`scoring_policy_sha256: null`, exactly one worker, and `validator_slot_count: 1`.
It uses the same verified model, private socket, isolation and inference deadline.
It imports no wallet and offers no HTTP endpoint. The UMI miner's policy-bound
capacity validator rejects this socket because no scoring policy is assigned.

This allows the model to stay loaded while launch inputs are prepared. Replace
the standby configuration through a normal stop/start with reviewed policy-bound
inputs when available. Standby is not mining, reward activation or proof that the
model meets the competition's full workload. Never report it as such.

## Supervision and recovery

After an inference failure or cancellation, a safely reaped worker reloads
under its existing startup deadline before its slot becomes available again.
Reloading runs separately from incoming requests: a queued request timing out
does not cancel the reload. Reloads are serialized to limit startup resource
bursts. Inference deadlines still include time spent waiting for a ready slot.
The capacity descriptor records configured capacity, not an instantaneous count
of ready workers or proof of measured throughput.

A reload verification failure closes the service; it does not return a broken
worker to the queue. Shutdown cancels reloads and drains worker cleanup before
exiting. This recovery does not change model weights or inference deadlines.

Run the service with the dedicated miner operator's identity and retain its
configuration outside model-writable paths. Configure a restart delay so an
invalid deployment cannot repeatedly load the model in a tight loop. Never run
two service definitions for the same socket.

An occupied lock rejects a second instance before model loading. The lock file
remains after normal shutdown and is reused; do not delete it while a service
may still hold it. Normal shutdown reaps the workers and removes the owned
socket and capacity descriptor, allowing the same configuration to restart.

To recover after a host reboot, add `--recover-after-reboot` to the service
command. Enable it during a normal stop/start with empty worker scratch and no
retained socket or capacity descriptor. It requires Linux's machine ID and boot
ID, or macOS's platform UUID and boot-session UUID. An unavailable identity stops
startup. Keep the configuration, socket directory and worker scratch on local
storage. Shared filesystems and copied VM identities are unsupported.

Before launching a worker, the controller durably records the host, boot,
configuration hash and scratch-directory identities in `SOCKET.reboot.json`.
Normal shutdown marks that journal clean only after worker cleanup succeeds.
After a reboot, an unfinished journal with the same host and configuration lets
the controller move the old socket, capacity descriptor and scratch into private
`.umi-reboot-*` sibling paths. It creates empty replacement scratch and starts
the configured workers. An interrupted quarantine resumes its saved plan.
Quarantined data is retained for review; it is never followed or recursively
deleted by recovery. It may include input clips, so retain its private permissions
and review disk use after an outage.

An unfinished service from the current boot still requires operator review,
even if no socket exists. This covers a controller killed while a worker was
starting. The controller does not infer worker death from a missing PID or socket.
Unknown files, changed identities and changed unfinished configurations also stop
startup. Never remove the journal or lock to bypass that check. Once recovery is
enabled, omitting the flag does not bypass its journal.

The HTTP miner's finality, assignment and nonce databases are separate retained
state. Neither this service nor model scratch cleanup may remove them. Starting
the model sidecar does not publish an axon, open enrollment, activate the 70/30
policy or establish that the miner is earning rewards.

## Test coverage

The September 16 [qualification record](QUALIFICATION.md) includes six real
requests through the installed policy-bound service, after production HTTPS
video retrieval. All completed within 120 seconds. This covers that six-clip
serving path, not signed settlement, general accuracy or full-load capacity.

Run these tests with Python 3.12 and the reviewed UMI source on `PYTHONPATH`:

```sh
PYTHONPATH=/ABSOLUTE/PATH/TO/umi/src python -m pytest \
  tests/test_community_service.py tests/test_community_sidecar.py \
  tests/test_community_worker_transport.py tests/test_community_reboot_recovery.py -W error
```

The service and socket tests require UMI's `model_sidecar` helper; they skip if
it is absent. Treat such skips as missing integration evidence. A model worker
command, sandbox and verified artifacts are separate deployment inputs; these
tests do not install or load them.

The service tests use inert workers and the actual UMI socket protocol. They
cover hash/schema/file checks, separate scratch, exclusive startup, duplicate
rejection, one request, graceful shutdown and restart with the same configuration.
The recovery tests simulate changed boot identities and interrupted filesystem
operations. They also check real OS identity reads and graceful service restarts.
They do not power-cycle a host or qualify native-model throughput or accuracy.

On Studio, the 32 service tests and 63 existing transport/socket tests passed
together with warnings treated as errors (95 total). The full reference suite
passed 378 tests with six optional integration skips. Both source hashes matched
the tested snapshot. Ruff 0.12.12 check and formatting passed.

The standby extension then passed all 102 service/socket/transport cases on
Studio, with no skips and warnings treated as errors. An installed native-model
standby answered a public two-second clip in 25.7 seconds, then 23.3 seconds after
a normal service stop/start. Those are local functional checks on one shared
machine. The public miner remained unavailable, and a supported longer clip
had already exceeded the ordinary 120-second limit. No full-workload or accuracy
qualification is claimed.

The reboot-recovery extension passed all 122 service, recovery, socket and worker
transport cases on Studio, with no skips and warnings treated as errors. The full
committed-source suite passed 297 tests, with four opt-in model/container/capacity
integrations skipped. This includes a real controller process exiting before
socket creation, followed by rejection of its same-boot restart. Reboot identities
in quarantine tests are simulated; an actual host power cycle is not covered.
