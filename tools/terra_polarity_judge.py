"""Separate append-only polarity clarification; original scores and controls stay intact."""

from __future__ import annotations

import hashlib
import json
import time
from types import FunctionType

import c5_word_training as training
from probe_jeff_terra_quality import terra_payload as original_payload
from semantic_quality_judge import calibration_controls, parse_terra_scores
from terra_quality_training import score_with_terra

CLARIFICATION = (
    " A candidate that reverses whether the reference's central event or state holds "
    "contradicts the central message. Score such a central contradiction at most 3, "
    "even when it repeats the same entities, action words, quantities or time. "
    "Judge the meaning, not the presence of a negation word: matching negated meanings "
    "and faithful paraphrases remain eligible for 10."
)


def bind(function, **replacements):
    result = FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def payload(rows):
    result = original_payload(rows)
    result["instructions"] += CLARIFICATION
    return result


def protocol():
    return {
        "schema": "umi-studio-Terra-polarity-clarification/1",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "clarification_append_only": CLARIFICATION,
        "full_instructions_sha256": hashlib.sha256(
            payload([])["instructions"].encode()
        ).hexdigest(),
        "original_controls": calibration_controls(),
        "score_postprocessing": "none; every actual original control and human anchor must pass",
        "historical_scores_or_failed_lineages_reclassified": False,
        "existing_training_continuation_authorized": False,
        "promotion_authorized": False,
    }


score_with_polarity = bind(score_with_terra, terra_payload=payload)
verified_training_rewards = bind(training.verified_rewards, terra_payload=payload)
score_training_batch = bind(training.score_batch, verified_rewards=verified_training_rewards)


class Pacer:
    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep, self.previous_start = clock, sleep, None

    def score(self, rows, output):
        if not 1 <= len(rows) <= 16:
            raise ValueError("polarity scorer requires1-16 pairs")
        if self.previous_start is not None:
            while (remaining := 60 - (self.clock() - self.previous_start)) > 0:
                self.sleep(min(30, remaining))
        self.previous_start = self.clock()
        return score_with_polarity(rows, output)


def authenticated_scores(rows, directory):
    def read(name):
        return json.loads((directory / name).read_text())

    if not 1 <= len(rows) <= 16:
        raise ValueError("polarity authentication requires1-16 pairs")
    request, started, response, saved = [
        read(name) for name in ("request.json", "started.json", "response.json", "scores.json")
    ]
    if (
        request != payload(rows)
        or started["request_sha256"]
        != hashlib.sha256((directory / "request.json").read_bytes()).hexdigest()
        or started["pairs"] != len(rows)
        or started["automatic_retries"] != 0
        or any(record.get("model") != "gpt-5.6-terra" for record in (started, response, saved))
        or any(record.get("reasoning_effort") != "medium" for record in (started, saved))
        or response.get("reasoning", {}).get("effort") != "medium"
        or saved["response_sha256"]
        != hashlib.sha256((directory / "response.json").read_bytes()).hexdigest()
    ):
        raise ValueError("polarity scores lost their exact prompt/model/request/response bindings")
    parsed = parse_terra_scores(response, [f"pair_{i:02d}" for i in range(len(rows))])
    actual = [parsed[f"pair_{i:02d}"] for i in range(len(rows))]
    if actual != saved["scores"]:
        raise ValueError("polarity scores differ from the actual raw response")
    return actual
