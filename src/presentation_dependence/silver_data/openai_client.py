"""Batched-self-consistency scoring against an OpenAI-compatible chat endpoint.

The closed-model genBSC baseline scores a candidate set the same way the
open-weight scorers do: one prompt holds several candidates, the model emits
one integer grade per candidate against the shared rubric, and the pool is
scored under several shuffled presentations whose grades are averaged. Only the
backend differs, so this client reuses the prompt builder and rubric in
:mod:`presentation_dependence.rerankers.grade_rubrics` rather than defining its own, which
is what keeps the row comparable to the trained scorers beside it.

Works against anything speaking the OpenAI chat-completions API: the hosted
service, a self-hosted gateway, or a local vLLM server. Point `OPENAI_BASE_URL`
at the endpoint and `OPENAI_API_KEY` at its credential. Only the standard
library is used, so no provider SDK enters the dependency set.

This is optional, provider-neutral infrastructure. Running it sends configured
corpus text to the operator-selected endpoint. Operators are responsible for
ensuring that their corpus use and provider agreement permit that processing.
The source repository includes no credentials or generated provider output.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from presentation_dependence.rerankers.grade_rubrics import build_pathc_grade_user_body, resolve_grade_rubric
from presentation_dependence.silver_data.client import SilverRequest, SilverResponse, iter_request_groups

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# `[3] Grade: 2`, tolerating stray whitespace and a trailing period.
_GRADE_LINE = re.compile(r"\[\s*(\d+)\s*\]\s*Grade\s*:\s*([0-9]+)", re.IGNORECASE)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ClosedModelError(RuntimeError):
    """Raised when the endpoint cannot be reached or its reply cannot be used."""


class OpenAICompatibleClient:
    """Score candidate sets through an OpenAI-compatible chat endpoint.

    Args:
        model_id: Model name the endpoint expects, e.g. ``gpt-5.4``.
        subset_size: Candidates per prompt. The canonical batched setting is 20.
        runs: Shuffled presentations to average over. ``1`` disables averaging
            here, which is what evaluation configs want when the PSI harness owns
            presentation diversity instead.
        run_seeds: One seed per run, so a presentation is reproducible. Defaults
            to ``range(runs)``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        subset_size: int = 20,
        runs: int = 10,
        run_seeds: list[int] | None = None,
        score_min: float = 0.0,
        score_max: float = 3.0,
        max_tokens: int = 1536,
        temperature: float = 0.0,
        instruction: str | None = None,
        grade_rubric_id: str | None = None,
        max_doc_chars: int | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 5,
        impute_missing_doc_scores: bool = False,
    ) -> None:
        self.model_id = model_id
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.subset_size = max(1, int(subset_size))
        self.runs = max(1, int(runs))
        self.run_seeds = list(run_seeds) if run_seeds else list(range(self.runs))
        if len(self.run_seeds) != self.runs:
            raise ValueError(f"run_seeds has {len(self.run_seeds)} entries but runs={self.runs}")
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.instruction = instruction
        self.grade_rubric_id = grade_rubric_id
        self.max_doc_chars = int(max_doc_chars) if max_doc_chars else 1200
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.impute_missing_doc_scores = bool(impute_missing_doc_scores)
        self._call_index = 0

    # ---------------------------------------------------------------- transport

    def _chat(self, system_prompt: str, user_body: str) -> dict[str, Any]:
        """POST one chat completion, retrying transient failures with backoff."""
        if not self._api_key:
            raise ClosedModelError("No API key. Set OPENAI_API_KEY, or pass api_key= when constructing the client.")
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_body},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in _RETRYABLE_STATUS:
                    raise ClosedModelError(f"{exc.code} from {self.base_url}: {exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            time.sleep(min(2.0**attempt, 30.0))
        raise ClosedModelError(f"{self.max_retries} attempts failed against {self.base_url}: {last}")

    # ------------------------------------------------------------------ scoring

    def _score_one_prompt(self, query: str, documents: list[str], raw_dir: Path) -> tuple[dict[int, float], dict]:
        """Grade one presentation of up to `subset_size` candidates.

        Returns grades keyed by 1-based slot, which is how the answer skeleton
        labels them, plus the raw reply for provenance.
        """
        rubric = resolve_grade_rubric(self.grade_rubric_id)
        user_body = build_pathc_grade_user_body(
            self.instruction,
            query,
            documents,
            max_doc_chars=self.max_doc_chars,
            grade_rubric_id=self.grade_rubric_id,
        )
        raw = self._chat(rubric.system_prompt, user_body)
        try:
            content = str(raw["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ClosedModelError(f"Unexpected chat-completion shape: {raw!r}") from exc

        grades: dict[int, float] = {}
        for slot, value in _GRADE_LINE.findall(content):
            index = int(slot)
            if 1 <= index <= len(documents):
                grades[index] = min(max(float(value), self.score_min), self.score_max)

        self._call_index += 1
        raw_path = raw_dir / f"call_{self._call_index:06d}.json"
        raw_path.write_text(json.dumps({"request": user_body, "response": raw}, indent=2), encoding="utf-8")
        return grades, raw

    def _score_group(self, group: list[SilverRequest], raw_dir: Path) -> dict[str, list[float]]:
        """Grade one query's candidates under every configured presentation."""
        query = group[0].query
        documents = [request.document for request in group]
        per_doc: dict[str, list[float]] = {request.doc_id: [] for request in group}

        for seed in self.run_seeds:
            order = list(range(len(group)))
            if self.runs > 1:
                random.Random(seed).shuffle(order)
            for start in range(0, len(order), self.subset_size):
                chunk = order[start : start + self.subset_size]
                grades, _ = self._score_one_prompt(query, [documents[i] for i in chunk], raw_dir)
                for slot, position in enumerate(chunk, start=1):
                    grade = grades.get(slot)
                    if grade is None:
                        if not self.impute_missing_doc_scores:
                            continue
                        grade = self.score_min
                    per_doc[group[position].doc_id].append(grade)
        return per_doc

    # -------------------------------------------------------------------- client

    def submit_query_batches(
        self, requests: list[SilverRequest], raw_dir: Path
    ) -> Iterator[tuple[SilverRequest, SilverResponse]]:
        """Score one query group per call set, yielding a response per request."""
        if not requests:
            return
        raw_dir.mkdir(parents=True, exist_ok=True)
        for group in iter_request_groups(requests):
            try:
                per_doc = self._score_group(group, raw_dir)
            except ClosedModelError as exc:
                for request in group:
                    yield (
                        request,
                        SilverResponse(
                            call_id=request.call_id,
                            score_raw=None,
                            raw_response={},
                            success=False,
                            error=str(exc),
                        ),
                    )
                continue
            for request in group:
                grades = per_doc.get(request.doc_id) or []
                if not grades:
                    yield (
                        request,
                        SilverResponse(
                            call_id=request.call_id,
                            score_raw=None,
                            raw_response={},
                            success=False,
                            error="model returned no grade for this candidate",
                        ),
                    )
                    continue
                mean = sum(grades) / len(grades)
                yield (
                    request,
                    SilverResponse(
                        call_id=request.call_id,
                        score_raw=f"{mean:.6f}",
                        raw_response={"grades": grades, "model": self.model_id},
                        score_raw_vector=list(grades),
                    ),
                )

    def submit_batch(self, requests: list[SilverRequest], raw_dir: Path) -> list[SilverResponse]:
        """Score requests and return responses in request order."""
        by_call_id = {request.call_id: response for request, response in self.submit_query_batches(requests, raw_dir)}
        return [by_call_id[request.call_id] for request in requests]

    def shutdown(self) -> None:
        """No persistent process to stop; present so callers can close uniformly."""
