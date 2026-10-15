"""Local Jeff versus Terra medium on fixed diagnostic-only text pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from semantic_quality_judge import (
    INSTRUCTIONS,
    LEVELS,
    calibration_controls,
    from_jeff_probabilities,
    jeff_prompt,
    parse_terra_scores,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    (
        "mstrasser/jeff-base-gguf",
        "a5d0e595a0161864cad99efa9fe9e9a43a797893",
        "models/base-q4_k_m.gguf",
        672329312,
        "0120fd94d172c439fbfc733d5d2b15eecdd9d7a0a72cf184d5909fb02e926472",
    ),
    (
        "mstrasser/jeff-adapter-ground-gguf",
        "c47b9a5eff9fef6eb39f3a653df109964cfd24e0",
        "loras/ground.gguf",
        168603104,
        "ced9506c3ad5de4096ea244e3de0fde6c2c064d50432e180b081dddf22aea95b",
    ),
)
POLICY = {"maximum_api_usd": 0.10, "max_output_tokens": 4096, "cpu_threads": 2}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open("x") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def storage(prospective):
    # Reserve the next active training checkpoint as well as the physical floor.
    if shutil.disk_usage(ROOT).free < 80 * 1024**3 + 128 * 1024**2 + 215000000 + prospective:
        raise RuntimeError("Jeff pilot would breach disk floor or next training checkpoint reserve")


def request(url, value=None, *, key=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    body = None if value is None else json.dumps(value).encode()
    with urllib.request.urlopen(
        urllib.request.Request(url, data=body, headers=headers), timeout=timeout
    ) as response:
        return json.load(response)


def cases():
    plan = json.loads((ROOT / "runs/tiny-rgb-feature-winner-full-v1/plan.json").read_text())
    evaluation = {row["example_id"]: row for row in plan["evaluation"]}
    student = [
        json.loads(line)
        for line in (
            ROOT
            / "checkpoints/tiny-rgb-feature-winner-full-v1/cpu-evaluation-002048"
            / "evaluation-results.jsonl"
        )
        .read_text()
        .splitlines()
    ][:3]
    winner = {
        row["example_id"]: row
        for row in map(
            json.loads,
            (ROOT / "checkpoints/c5-winner-full-v2-early/evaluation-results.jsonl")
            .read_text()
            .splitlines(),
        )
        if row["model"] == "winner_matched_translator_control" and row["checkpoint_examples"] == 0
    }
    result = calibration_controls()
    for i, current in enumerate(student, 1):
        reference = evaluation[current["example_id"]]
        if reference["role"] != "open_development" or reference["training_allowed"] is not False:
            raise ValueError("requested examples must remain protected diagnostics")
        for variant, prediction in (("c5", winner[current["example_id"]]), ("student", current)):
            result.append(
                {
                    "id": f"example_{i}_{variant}",
                    "reference": reference["reference"],
                    "candidate": prediction["hypothesis"],
                    "example_id": current["example_id"],
                    "role": "owner_requested_open_development_diagnostic",
                    "training_allowed": False,
                }
            )
    return result


def terra_payload(rows):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores"],
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "score"],
                    "properties": {
                        "id": {"type": "string"},
                        "score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                        },
                    },
                },
            }
        },
    }
    # Opaque IDs avoid telling either judge which transcript came from which ASL model.
    public = [
        {"id": f"pair_{i:02d}", "reference": row["reference"], "candidate": row["candidate"]}
        for i, row in enumerate(rows)
    ]
    return {
        "model": "gpt-5.6-terra",
        "reasoning": {"effort": "medium"},
        "store": False,
        "max_output_tokens": POLICY["max_output_tokens"],
        "instructions": INSTRUCTIONS
        + "\nQuality levels:\n"
        + "\n".join(f"{i}: {description}" for i, description in enumerate(LEVELS))
        + "\nReturn one integer score for every pair, identified by its opaque ID.",
        "input": json.dumps(public, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "semantic_quality_scores",
                "strict": True,
                "schema": schema,
            }
        },
    }


def run(output):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("quality pilot requires a new runs/ output directory")
    storage(sum(item[3] for item in ASSETS) + 1024 * 1024)
    output.mkdir(parents=True)
    rows = cases()
    payload = terra_payload(rows)
    # UTF-8 bytes plus framing conservatively bound input tokens. Include a 25% input
    # cache-write surcharge. Hidden reasoning is included in max_output_tokens.
    upper = (len(json.dumps(payload).encode()) + 1024) * 2.5 / 1e6
    upper += POLICY["max_output_tokens"] * 12 / 1e6
    if upper > POLICY["maximum_api_usd"]:
        raise ValueError("Terra request exceeds the declared worst-case API budget")
    write(
        output / "plan.json",
        {
            "schema": "umi-studio-jeff-terra-semantic-quality-pilot/1",
            "cases": rows,
            "rubric": LEVELS,
            "instructions": INSTRUCTIONS,
            "policy": POLICY,
            "api_request_cost_upper_bound_usd": upper,
            "assets": ASSETS,
            "source_sha256": {
                str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                "tools/semantic_quality_judge.py": sha(ROOT / "tools/semantic_quality_judge.py"),
            },
            "asr_optimizer_updates": 0,
            "training_allowed": False,
            "wer_gate_changed": False,
            "miner_modified": False,
        },
    )
    assets = output / "assets"
    assets.mkdir()
    for repo, revision, name, size, digest in ASSETS:
        destination = assets / Path(name).name
        storage(size)
        with (
            urllib.request.urlopen(
                f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=120
            ) as source,
            destination.open("xb") as target,
        ):
            total = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                total += len(block)
                if total > size:
                    raise ValueError("Jeff download exceeds pinned size")
                target.write(block)
        if total != size or sha(destination) != digest:
            raise ValueError("Jeff asset fails pinned size/hash")
        print(json.dumps({"asset_verified": name, "bytes": size}), flush=True)
    repo, revision, *_ = ASSETS[1]
    config = request(f"https://huggingface.co/{repo}/resolve/{revision}/models/ground.jeff.json")
    if config["prompt_layout"] != "live-last" or config["token_ids"][:11] != list(range(32, 43)):
        raise ValueError("Jeff prompt/code contract changed")
    write(output / "ground.jeff.json", config)
    binary = Path(shutil.which("llama-server") or "").resolve()
    write(
        output / "runtime.json", {"binary": str(binary), "sha256": sha(binary), "gpu_used": False}
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        str(binary),
        "-m",
        str(assets / "base-q4_k_m.gguf"),
        "--lora",
        str(assets / "ground.gguf"),
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "--no-op-offload",
        "--no-kv-offload",
        "--fit",
        "off",
        "--threads",
        "2",
        "--threads-batch",
        "2",
        "--parallel",
        "1",
        "--ctx-size",
        "2048",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    base = f"http://127.0.0.1:{port}"
    with (output / "owned-jeff-server.log").open("x") as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write(output / "owned-server.json", {"pid": worker.pid, "command": command})
        try:
            deadline = time.monotonic() + 60
            while True:
                if worker.poll() is not None:
                    raise RuntimeError("owned Jeff server terminated during startup")
                try:
                    if request(base + "/health", timeout=1).get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("owned Jeff server startup failed; no restart authorized")
                time.sleep(0.5)
            jeff = []
            for row in rows:
                started = time.monotonic()
                prompt = jeff_prompt(row["reference"], row["candidate"], config["codes"][:11])
                tokens = request(
                    base + "/tokenize",
                    {
                        "content": prompt,
                        "add_special": False,
                        "parse_special": True,
                    },
                )["tokens"]
                if len(tokens) > 2047:
                    raise ValueError("Jeff prompt would exceed context; never truncate")
                response = request(
                    base + "/completion",
                    {
                        "prompt": tokens,
                        "n_predict": 1,
                        "cache_prompt": False,
                        "lora": [{"id": 0, "scale": 1.0}],
                        "seed": 317,
                        "samplers": ["temperature"],
                        "temperature": 1.0,
                        "logit_bias": [[token, 1000] for token in config["token_ids"][:11]],
                        "n_probs": 11,
                        "post_sampling_probs": True,
                    },
                )
                write(output / f"jeff-{row['id']}.json", response)
                probabilities = response["completion_probabilities"][0]["top_probs"]
                by_id = {p["id"]: p["prob"] for p in probabilities}
                result = from_jeff_probabilities(
                    [by_id[token] for token in config["token_ids"][:11]],
                    config["temperature_by_format"]["q4_k_m"],
                )
                jeff.append({"id": row["id"], **result, "seconds": time.monotonic() - started})
                print(json.dumps(jeff[-1]), flush=True)
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=10)
            write(
                output / "owned-server-terminal.json",
                {"pid": worker.pid, "exit_code": worker.returncode},
            )
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Terra comparison needs the configured OpenAI API key")
    write(output / "terra-request.json", payload)
    # This marker prevents accidental replay after an ambiguous network failure.
    write(output / "terra-request-started.json", {"maximum_cost_usd": upper, "retries": 0})
    started = time.monotonic()
    response = request("https://api.openai.com/v1/responses", payload, key=key)
    elapsed = time.monotonic() - started
    write(output / "terra-response.json", response)
    scores = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(rows))])
    result = {
        "complete": True,
        "comparison": [
            {**row, "jeff": observed, "terra_quality": scores[f"pair_{i:02d}"]}
            for i, (row, observed) in enumerate(zip(rows, jeff, strict=True))
        ],
        "terra_model": response["model"],
        "reasoning_effort": "medium",
        "terra_seconds": elapsed,
        "terra_usage": response["usage"],
        "asr_optimizer_updates": 0,
        "training_allowed": False,
        "scores_are_subjective_semantic_quality_not_wer": True,
    }
    write(output / "decision.json", result)
    print(json.dumps({"complete": True, "decision": str(output / "decision.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)
