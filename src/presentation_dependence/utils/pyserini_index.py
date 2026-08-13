"""Guards for using Pyserini: importing it, and opening a prebuilt index.

``LuceneSearcher.from_prebuilt_index`` and ``FaissSearcher.from_prebuilt_index``
return ``None`` for an unknown index name rather than raising. The failure then
surfaces much later as ``AttributeError: 'NoneType' object has no attribute
'doc'`` deep inside a loader or a data build, which reads like a corrupt index
rather than a typo in ``data.index``.

Wrapping every construction site turns that into an immediate, named error and
narrows the type for static analysis at the same time.

:func:`pyserini_import` covers a separate hazard in the same dependency: the
import itself fails without an OpenAI key. See its docstring.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, TypeVar

SearcherT = TypeVar("SearcherT")

# Deliberately not key-shaped, so a secret scanner does not flag it and nobody
# mistakes it for a revoked credential.
_OPENAI_PLACEHOLDER = "unused-no-openai-calls-in-this-process"


@contextmanager
def pyserini_import() -> Iterator[None]:
    """Make ``pyserini.search.lucene`` and ``pyserini.encode`` importable.

    ``pyserini.encode`` constructs an ``openai.OpenAI`` client at module scope,
    and that constructor raises when no key is present. Importing anything
    under ``pyserini.search.lucene`` pulls it in, so on a machine that has
    never set ``OPENAI_API_KEY`` a BM25 lookup dies with a credentials error
    naming a service it does not use. Nothing here calls OpenAI; the encoders
    this repository uses are ONNX and Hugging Face.

    The placeholder is removed again on the way out. A key must stay absent for
    the rest of the process, because a closed-model reranker can run in this
    same process over a Pyserini first stage: leaving a fake key set would turn
    "you forgot to export your key" into a 401 from the provider, which the
    eval launchers treat as retryable and would spin on.
    """
    if os.environ.get("OPENAI_API_KEY"):
        yield
        return
    os.environ["OPENAI_API_KEY"] = _OPENAI_PLACEHOLDER
    try:
        yield
    finally:
        if os.environ.get("OPENAI_API_KEY") == _OPENAI_PLACEHOLDER:
            del os.environ["OPENAI_API_KEY"]


def require_prebuilt_index(searcher: SearcherT | None, index_name: str | None) -> SearcherT:
    """Return ``searcher``, or raise naming the index that could not be opened."""
    if searcher is not None:
        return searcher
    raise RuntimeError(
        f"Pyserini could not open prebuilt index {index_name!r}. "
        "`from_prebuilt_index` returns None for an unknown name, so this is "
        "usually a typo or a renamed index rather than a corrupt download. "
        "List the valid names with "
        "`uv run python -m pyserini.prebuilt_index_info`."
    )
