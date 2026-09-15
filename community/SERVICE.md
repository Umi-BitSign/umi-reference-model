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

Run the service with the dedicated miner operator's identity and retain its
configuration outside model-writable paths. Configure a restart delay so an
invalid deployment cannot repeatedly load the model in a tight loop. Never run
two service definitions for the same socket.

An occupied lock rejects a second instance before model loading. The lock file
remains after normal shutdown and is reused; do not delete it while a service
may still hold it. Normal shutdown reaps the workers and removes the owned
socket and capacity descriptor, allowing the same configuration to restart.

After SIGKILL or power loss, an existing socket, capacity descriptor or nonempty
scratch stops startup. The service does not guess whether an old worker is still
alive or delete those paths automatically. Verify process ownership and retained
state before recovery. Automatic crash recovery remains deployment work.

The HTTP miner's finality, assignment and nonce databases are separate retained
state. Neither this service nor model scratch cleanup may remove them. Starting
the model sidecar does not publish an axon, open enrollment, activate the 70/30
policy or establish that the miner is earning rewards.

## Test coverage

Run these tests with Python 3.12 and the reviewed UMI source on `PYTHONPATH`:

```sh
PYTHONPATH=/ABSOLUTE/PATH/TO/umi/src python -m pytest \
  tests/test_community_service.py tests/test_community_sidecar.py \
  tests/test_community_worker_transport.py -W error
```

The service and socket tests require UMI's `model_sidecar` helper; they skip if
it is absent. Treat such skips as missing integration evidence. A model worker
command, sandbox and verified artifacts are separate deployment inputs; these
tests do not install or load them.

The service tests use inert workers and the actual UMI socket protocol. They
cover hash/schema/file checks, separate scratch, exclusive startup, duplicate
rejection, one request, graceful shutdown and restart with the same configuration.
They do not qualify native-model throughput, accuracy, power-loss recovery or
the host's launchd installation.

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
