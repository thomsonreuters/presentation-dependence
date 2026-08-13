"""Request, response, and client contract for closed-model scoring.

One scoring call carries a query, one candidate document, and the prompt that
frames them. A client turns a list of those into responses; how it reaches the
model is its own business, which is what lets the generation pipeline stay
model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol


@dataclass(frozen=True)
class SilverRequest:
    call_id: str
    query_id: str
    doc_id: str
    query: str
    document: str
    prompt: str
    prompt_template_id: str
    teacher_model_id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SilverResponse:
    call_id: str
    score_raw: str | None
    raw_response: dict[str, Any]
    # One grade per presentation when the client scores several shuffled views;
    # `score_raw` is their mean.
    score_raw_vector: list[float] | None = None
    cost_usd: float | None = None
    batch_id: str | None = None
    success: bool = True
    error: str | None = None


class SilverClient(Protocol):
    def submit_batch(self, requests: list[SilverRequest], raw_dir: Path) -> list[SilverResponse]:
        """Score requests and persist raw provider output under `raw_dir`."""


def iter_request_groups(requests: list[SilverRequest]) -> Iterator[list[SilverRequest]]:
    """Split a request list into consecutive runs sharing a query, template and model.

    Consecutive rather than sorted: callers build requests in candidate order,
    and a batched scorer has to preserve that order to reproduce a presentation.
    """
    current_key: tuple[str, str, str] | None = None
    current: list[SilverRequest] = []
    for request in requests:
        key = (request.query_id, request.prompt_template_id, request.teacher_model_id)
        if current_key is not None and key != current_key:
            yield current
            current = []
        current_key = key
        current.append(request)
    if current:
        yield current
