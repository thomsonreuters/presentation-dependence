"""Transformers v5 compatibility shim for vLLM 0.10.x.

vLLM 0.10.x calls ``tokenizer.all_special_tokens_extended``, a property
that was removed in transformers 5.0 (see vllm-project/vllm#29686, fix
landed in vLLM 0.12.0). This shim is intentionally retained for the
Qwen3-4B-Instruct-2507 and Qwen3-Reranker-4B silver configs, which use
a PyTorch 2.8 base with vLLM 0.10.2. Current reproduction and
the other silver configs use the vLLM 0.19 image and do not require this
patch.

Restores ``all_special_tokens_extended`` as a property on
``transformers.PreTrainedTokenizerBase`` when running on tf 5+. The
implementation reproduces the tf 4.x semantics: return a deduped list
of all special tokens (named + extra) preserving the original token
type (``str`` or ``AddedToken``), but reads from tf 5's storage:
``_special_tokens_map`` (named tokens) + ``_extra_special_tokens``
(model-specific extras). ``special_tokens_map_extended`` itself was
also removed in tf 5, so we cannot port the tf 4.x impl verbatim.

The shim is invoked at *module import time* of
:mod:`presentation_dependence.self_distill.engines.vllm`, so any code path that
imports the vLLM engine gets the patch automatically, including vLLM's
internal tokenizer code.

Migration trigger: after the two retained Qwen teacher configs move off
vLLM 0.10.2, delete this module, the import in ``engines/vllm.py``, and
the entrypoint ``.pth`` installer. ``test_tf5_vllm_compat.py`` pins the
legacy behavior until then.

Apache-2.0 source and modification notice:

* Behavioral source is Transformers 4.57.6
  (``753d61104116eefc8ffc977327b441ee0c8d599f``),
  ``tokenization_utils_base.py``. Copyright 2020 The HuggingFace Inc. team.
* Compatibility target is Transformers 5.4.0
  (``276f1402020831d949c2e1a80574a5603995de23``).
* The local implementation replaces the removed
  ``special_tokens_map_extended`` traversal with separate reads from
  Transformers 5's private named-token and model-extra stores while preserving
  ``AddedToken`` objects for old vLLM.

Licence text is at ``third_party_licenses/Apache-2.0.txt``; see
``THIRD_PARTY_NOTICES.md``. The upstream repository has no root ``NOTICE``.
"""

from __future__ import annotations


def _all_special_tokens_extended_shim(self):
    """Tf 5 reimplementation of tf 4.x's ``all_special_tokens_extended``.

    Returns a deduplicated list of all special tokens (named + extras)
    preserving the original token type (``str`` or ``AddedToken``). Built
    on tf 5's private ``_special_tokens_map`` and ``_extra_special_tokens``
    storage, which replaced tf 4.x's
    ``special_tokens_map_extended`` dict.

    Mirrors the structure of tf 5's own ``all_special_tokens`` property
    (which returns ``list[str]``) but preserves AddedToken objects so
    vLLM's tokenizer code that iterates the list can still introspect
    ``.lstrip``/``.rstrip``/``.normalized`` flags.
    """
    seen: set[str] = set()
    out: list = []
    for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
        value = self._special_tokens_map.get(attr)
        if value is None:
            continue
        token_str = str(value)
        if token_str in seen:
            continue
        seen.add(token_str)
        out.append(value)
    for token in self._extra_special_tokens:
        token_str = str(token)
        if token_str in seen:
            continue
        seen.add(token_str)
        out.append(token)
    return out


def ensure_tf5_vllm_compat() -> bool:
    """Backfill ``all_special_tokens_extended`` on tf 5+ tokenizers.

    Returns ``True`` if the shim was applied (tf 5 detected), ``False`` if
    transformers already exposes the attribute (tf 4.x or future tf with a
    re-added attribute) or if transformers isn't importable.

    Idempotent: safe to call any number of times. Does NOT overwrite an
    existing implementation; only adds the property if missing.
    """
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return False

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return False

    PreTrainedTokenizerBase.all_special_tokens_extended = property(_all_special_tokens_extended_shim)
    return True
