"""Terra medium quality judgments and differentiable minimum-risk training.

The candidate-set objective follows Shen et al., ACL 2016, equations 12-14.
Terra supplies fixed semantic errors, never gradients or reference-only weights.
"""

from __future__ import annotations

import math
import os
import time
import urllib.error
import urllib.parse
from pathlib import Path

from probe_jeff_terra_quality import request, sha, terra_payload, write
from semantic_quality_judge import parse_terra_scores


def responses_endpoint(environment):
    configured = [
        environment[key].rstrip("/")
        for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE")
        if environment.get(key)
    ]
    if len(set(configured)) > 1:
        raise ValueError("configured OpenAI endpoints disagree")
    base = configured[0] if configured else "https://api.openai.com/v1"
    parsed = urllib.parse.urlsplit(base)
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/v1")
        or not (
            parsed.scheme == "https"
            or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1"))
        )
    ):
        raise ValueError("OpenAI endpoint must be HTTPS or explicit loopback, without credentials")
    return base + "/responses"


def safe_response(response):
    """Retain scoring evidence without encrypted reasoning or account identifiers."""
    return {
        key: value
        for key, value in {
            **{
                key: response.get(key)
                for key in (
                    "id",
                    "model",
                    "status",
                    "created_at",
                    "completed_at",
                    "usage",
                    "reasoning",
                    "incomplete_details",
                    "error",
                )
            },
            "output": [
                item for item in response.get("output", []) if item.get("type") == "message"
            ],
        }.items()
        if value is not None
    }


def score_with_terra(rows, output: Path):
    """One audited request, no automatic retries or endpoint/model fallbacks."""
    if not rows or len(rows) > 64 or output.exists():
        raise ValueError("Terra scoring requires 1-64 pairs and a fresh receipt directory")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("configured OpenAI key is absent")
    endpoint = responses_endpoint(os.environ)
    payload = terra_payload(rows)
    output.mkdir(parents=True)
    write(output / "request.json", payload)
    write(
        output / "started.json",
        {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "pairs": len(rows),
            "request_sha256": sha(output / "request.json"),
            "automatic_retries": 0,
            "configured_endpoint_used": bool(
                os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
            ),
        },
    )
    started = time.monotonic()
    try:
        response = request(endpoint, payload, key=key, timeout=120)
    except urllib.error.HTTPError as error:
        message = error.read(8192).decode("utf-8", errors="replace").replace(key, "[redacted]")
        write(
            output / "failure.json",
            {
                "http_status": error.code,
                "message": message,
                "ambiguous_completion": False,
                "automatic_retry_allowed": False,
            },
        )
        raise RuntimeError(
            f"Terra request rejected with HTTP {error.code}; receipt preserved"
        ) from None
    except (OSError, ValueError) as error:
        write(
            output / "failure.json",
            {
                "error_type": type(error).__name__,
                "ambiguous_completion": True,
                "automatic_retry_allowed": False,
            },
        )
        raise RuntimeError(
            "Terra transport/response failed; inspect receipt before any replay"
        ) from None
    write(output / "response.json", safe_response(response))
    if (
        response.get("model") != "gpt-5.6-terra"
        or response.get("reasoning", {}).get("effort") != "medium"
    ):
        raise RuntimeError("Terra response model or reasoning effort differs from request")
    scores = parse_terra_scores(response, [f"pair_{index:02d}" for index in range(len(rows))])
    result = {
        "scores": [scores[f"pair_{index:02d}"] for index in range(len(rows))],
        "seconds": time.monotonic() - started,
        "usage": response.get("usage"),
        "model": response["model"],
        "reasoning_effort": "medium",
        "response_sha256": sha(output / "response.json"),
    }
    write(output / "scores.json", result)
    return result


def sequence_log_probability(logits, labels):
    """Unnormalized sequence log probability, including EOS, ignoring padding only."""
    import torch

    if logits.shape[:2] != labels.shape or labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("minimum-risk likelihood requires one complete candidate")
    valid = labels != -100
    if not valid.any() or not torch.isfinite(logits).all():
        raise ValueError("empty or nonfinite candidate likelihood")
    indices = labels.masked_fill(~valid, 0)
    if (indices < 0).any() or (indices >= logits.shape[-1]).any():
        raise ValueError("candidate token outside decoder vocabulary")
    selected = logits.log_softmax(-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return selected[valid].sum()


def minimum_risk(log_probabilities, quality, *, alpha=0.005):
    """Expected error over a deduplicated candidate set; higher quality lowers risk."""
    import torch

    if (
        log_probabilities.ndim != 1
        or not len(log_probabilities)
        or len(quality) != len(log_probabilities)
        or not math.isfinite(alpha)
        or alpha <= 0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10
            for value in quality
        )
        or not torch.isfinite(log_probabilities).all()
    ):
        raise ValueError("invalid candidate log probabilities, Terra quality or smoothing")
    errors = log_probabilities.new_tensor([10 - value for value in quality])
    probabilities = (alpha * log_probabilities).softmax(0)
    loss = (probabilities * errors).sum()
    coefficients = alpha * probabilities * (errors - loss)
    return {"loss": loss, "probabilities": probabilities, "coefficients": coefficients}
