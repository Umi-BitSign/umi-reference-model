"""Filesystem transactions for a proposed per-batch word checkpoint policy.

Prepared for the separately authorized storage-recovery controller. The caller
supplies its original checkpoint writer and storage guard. Tensor validation and model restoration
remain the qualified writer/auditor's responsibility. Full epoch checkpoints,
source checkpoints and all receipts are retained; only retired working tensor
files can be removed by this store.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA = "umi-studio-word-working-checkpoint/1"
KEEP = 2
BYTE_BUDGET = 4 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            value.update(block)
    return value.hexdigest()


class WorkingCheckpoints:
    """Publish a complete model/Adam/RNG pair against an fsynced ledger prefix.

    Storage accounting always covers the whole run, including temporary files.
    A failed writer is never retried and its unpublished files remain evidence.
    The pointer can be replaced; tensor files and commit receipts cannot.
    """

    def __init__(self, output, *, workspace, plan_sha, storage_check, pair_reserve_bytes):
        self.workspace = Path(workspace).resolve()
        self.output = Path(output).resolve()
        require(
            self.output.is_relative_to(self.workspace / "checkpoints")
            and self.output != self.workspace / "checkpoints"
            and self.output.is_dir(),
            "working states require an existing owned checkpoint output",
        )
        require(
            isinstance(plan_sha, str)
            and len(plan_sha) == 64
            and all(character in "0123456789abcdef" for character in plan_sha),
            "missing frozen plan digest",
        )
        require(
            callable(storage_check) and type(pair_reserve_bytes) is int and pair_reserve_bytes > 0,
            "the original whole-run storage guard and pair reservation are required",
        )
        self.plan_sha = plan_sha
        self.storage_check = storage_check
        self.pair_reserve_bytes = pair_reserve_bytes
        self.directory = self.output / "working-checkpoints"
        self.pointer = self.output / "working-latest.json"
        self.journal = self.output / "working-checkpoints.jsonl"
        self._check_control_paths()

    def _check_control_paths(self):
        for path in (self.directory, self.pointer, self.journal, self.output / "steps.jsonl"):
            require(path.resolve() == path, "working control paths cannot follow aliases")

    def _reserve(self, prospective):
        self._check_control_paths()
        used = 0
        for path in self.output.rglob("*"):
            require(not path.is_symlink(), "working output cannot follow aliases")
            if path.is_file():
                used += path.stat().st_size
        require(
            used + prospective <= BYTE_BUDGET, "working states exceed the whole-run byte budget"
        )
        self.storage_check(prospective)

    def _path(self, relative):
        require(isinstance(relative, str), "invalid artifact path")
        path = self.workspace / relative
        require(
            not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and path.resolve() == path
            and path.is_relative_to(self.output),
            "working checkpoint cannot leave its owned output or follow an alias",
        )
        return path

    def _artifact(self, path):
        relative = str(path.relative_to(self.workspace))
        require(self._path(relative) == path and path.is_file(), "missing checkpoint artifact")
        return {"path": relative, "sha256": file_sha(path)}

    def _check_artifact(self, artifact):
        require(set(artifact) == {"path", "sha256"}, "invalid artifact binding")
        path = self._path(artifact["path"])
        require(path.is_file() and self._artifact(path) == artifact, "checkpoint bytes changed")
        return path

    def _events(self, *, allow_incomplete_tail=False):
        self._check_control_paths()
        if not self.journal.exists():
            return []
        data = self.journal.read_bytes()
        if not data.endswith(b"\n"):
            require(allow_incomplete_tail, "working journal has a partial append")
            data = data[: data.rfind(b"\n") + 1]
        return [json.loads(line) for line in data.splitlines()]

    def _append(self, event):
        self._check_control_paths()
        self._events()  # Never concatenate a new event onto an incomplete append.
        with self.journal.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _publish(self, state):
        self._check_control_paths()
        descriptor, temporary = tempfile.mkstemp(prefix=".working-pointer-", dir=self.output)
        path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(state, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path, self.pointer)
            self._sync_directory(self.output)
        finally:
            if path.exists():
                path.unlink()

    def _ledger(self):
        self._check_control_paths()
        raw = (self.output / "steps.jsonl").read_bytes()
        require(raw and raw.endswith(b"\n"), "checkpoint needs a complete nonempty ledger")
        rows = [json.loads(line) for line in raw.splitlines()]
        require(
            all(
                type(row.get("batch")) is int and row["batch"] == index
                for index, row in enumerate(rows)
            ),
            "checkpoint ledger batch indices are not a contiguous prefix",
        )
        return raw, rows

    @staticmethod
    def _sync_directory(path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _sync_artifacts(self, state):
        for artifact in [state[key] for key in ("visual", "optimizer", "commit")]:
            with self._check_artifact(artifact).open("rb") as stream:
                os.fsync(stream.fileno())
        with (self.output / "steps.jsonl").open("rb") as stream:
            os.fsync(stream.fileno())
        self._sync_directory(self._path(state["commit"]["path"]).parent)
        if self.directory.exists():
            self._sync_directory(self.directory)

    def _capture(self, commit_path, raw, rows, *, inherited):
        commit = json.loads(commit_path.read_text())
        require(
            commit.get("complete") is True and commit.get("committed_batch_RNG") is True,
            "writer did not commit the complete visual/Adam/RNG transaction",
        )
        for key in ("training_examples", "optimizer_updates"):
            require(
                type(commit.get(key)) is int
                and commit[key] > 0
                and rows[-1].get(key) == commit[key],
                "checkpoint counters differ from the accepted ledger",
            )
        artifacts = {}
        for key in ("visual", "optimizer"):
            artifact = commit[key]
            path = self._check_artifact(artifact)
            require(
                path.parent == commit_path.parent and path.suffix == ".safetensors",
                "tensor pair must share the commit directory",
            )
            artifacts[key] = artifact
        require(
            artifacts["visual"]["path"] != artifacts["optimizer"]["path"], "aliased tensor pair"
        )
        require(
            not os.path.samefile(
                self._path(artifacts["visual"]["path"]),
                self._path(artifacts["optimizer"]["path"]),
            ),
            "aliased tensor pair",
        )
        return {
            "schema": SCHEMA,
            "plan_sha256": self.plan_sha,
            "inherited_seed": inherited,
            "training_examples": commit["training_examples"],
            "optimizer_updates": commit["optimizer_updates"],
            "ledger_bytes": len(raw),
            "ledger_rows": len(rows),
            "ledger_sha256": digest(raw),
            **artifacts,
            "commit": self._artifact(commit_path),
        }

    def seed(self, commit_path):
        """Bind a caller-validated inherited checkpoint already copied into output."""
        require(
            not self.pointer.exists() and not self.journal.exists(), "working state already seeded"
        )
        self._reserve(16384)
        path = Path(commit_path).absolute()
        require(
            path.parent == self.output and path.suffix == ".json" and path.resolve() == path,
            "seed must be a root epoch commit",
        )
        raw, rows = self._ledger()
        state = self._capture(path, raw, rows, inherited=True)
        self._sync_artifacts(state)
        self._append({"kind": "prepared", "state": state})
        self._publish(state)
        self._append({"kind": "committed", "state": state})
        return state

    def latest(self):
        self._check_control_paths()
        state = json.loads(self.pointer.read_text())
        require(
            state.get("schema") == SCHEMA and state.get("plan_sha256") == self.plan_sha,
            "working pointer belongs to a different plan",
        )
        require(
            type(state.get("inherited_seed")) is bool
            and all(
                type(state.get(key)) is int and state[key] > 0
                for key in ("ledger_bytes", "ledger_rows")
            ),
            "invalid working pointer prefix fields",
        )
        require(
            {"kind": "prepared", "state": state} in self._events(allow_incomplete_tail=True),
            "working pointer has no durable preparation record",
        )
        prefix = (self.output / "steps.jsonl").read_bytes()[: state["ledger_bytes"]]
        require(
            len(prefix) == state["ledger_bytes"] and digest(prefix) == state["ledger_sha256"],
            "accepted checkpoint ledger prefix changed",
        )
        rows = [json.loads(line) for line in prefix.splitlines()]
        require(len(rows) == state["ledger_rows"], "checkpoint ledger row count changed")
        for key in ("visual", "optimizer", "commit"):
            self._check_artifact(state[key])
        actual = self._capture(
            self._path(state["commit"]["path"]), prefix, rows, inherited=state["inherited_seed"]
        )
        require(actual == state, "working pointer differs from its complete commit")
        return state

    def commit(self, *, examples, updates, writer):
        """Call writer once for the next accepted batch, then atomically publish."""
        self._events()  # A torn journal can be read for recovery but cannot advance.
        previous = self.latest()
        raw, rows = self._ledger()
        require(
            type(examples) is int
            and type(updates) is int
            and 0 < examples - previous["training_examples"] <= 4
            and updates - previous["optimizer_updates"] in (0, 1)
            and len(rows) == previous["ledger_rows"] + 1
            and rows[-1].get("training_examples") == examples
            and rows[-1].get("optimizer_updates") == updates,
            "working save must follow exactly one complete accepted batch",
        )
        self._reserve(self.pair_reserve_bytes + 16384)
        self.directory.mkdir(exist_ok=True)
        target = self.directory / f"batch-{len(rows):06d}"
        require(not target.exists(), "working writer is never retried or overwritten")
        target.mkdir()
        commit_path = Path(writer(target)).absolute()
        require(
            commit_path.parent == target
            and commit_path.suffix == ".json"
            and commit_path.resolve() == commit_path,
            "writer escaped its working directory",
        )
        require(
            (self.output / "steps.jsonl").read_bytes() == raw,
            "ledger changed during checkpoint write",
        )
        state = self._capture(commit_path, raw, rows, inherited=False)
        require(
            (state["training_examples"], state["optimizer_updates"]) == (examples, updates),
            "writer returned the wrong batch",
        )
        self._reserve(16384)
        self._sync_artifacts(state)
        self._append({"kind": "prepared", "state": state})
        self._publish(state)
        self._append({"kind": "committed", "state": state})
        self.retire()
        return state

    def retire(self):
        """Retain two working pairs and every receipt; retire tensor files only."""
        latest = self.latest()
        events = self._events()
        completed = [
            event["state"]
            for event in events
            if event["kind"] == "committed" and not event["state"]["inherited_seed"]
        ]
        require(
            len({state["ledger_rows"] for state in completed}) == len(completed),
            "duplicate working commits",
        )
        require(
            [state["ledger_rows"] for state in completed]
            == sorted(state["ledger_rows"] for state in completed),
            "working commit history is out of order",
        )
        if completed:
            require(completed[-1] == latest, "working history and pointer differ")
        protected = {
            state[key]["path"] for state in completed[-KEEP:] for key in ("visual", "optimizer")
        }
        for state in completed[:-KEEP]:
            for key in ("visual", "optimizer"):
                artifact = state[key]
                path = self._path(artifact["path"])
                require(
                    path.parent == self.directory / f"batch-{state['ledger_rows']:06d}"
                    and path.suffix == ".safetensors"
                    and artifact["path"] not in protected,
                    "retirement cannot reach epochs, seed or receipts",
                )
                intent = {"kind": "retire_intent", "artifact": artifact}
                done = {"kind": "retired", "artifact": artifact}
                if done in events:
                    require(not path.exists(), "retired tensor unexpectedly reappeared")
                    continue
                if path.exists():
                    self._check_artifact(artifact)
                    self._append(intent)
                    path.unlink()
                    self._sync_directory(path.parent)
                else:
                    require(intent in events, "working tensor missing without retirement intent")
                self._append(done)
