"""PINE HF/eager reranker for Qwen-style expected-grade scoring.

SPDX-License-Identifier: MIT
Portions Copyright (c) 2025 Ziqi Wang, from https://github.com/wzq016/PINE.
Licence text at ``third_party_licenses/PINE-MIT.txt``.

Implementation notes:
  - Locally adapted from the official MIT-licensed PINE repository
    (https://github.com/wzq016/PINE), especially
    ``pine/models/qwen2/modeling_qwen2.py`` and ``tokenization_qwen2.py``.
  - The official implementation is HF/eager-only, single-input, and has no vLLM
    support. This wrapper keeps the same constraints instead of relabelling a
    faster approximation as PINE.
  - This file implements only the path this repo needs for the external
    baseline: Qwen-family decoder LMs prompted as fixed expected-grade rerankers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Sequence

from presentation_dependence.rerankers._rank_result import scores_to_rank_result
from presentation_dependence.rerankers._torch_utils import resolve_device as _resolve_device
from presentation_dependence.rerankers._torch_utils import resolve_dtype as _resolve_dtype
from presentation_dependence.rerankers.base import Passage, Query, RankResult, Reranker
from presentation_dependence.rerankers.grade_rubrics import (
    DEFAULT_GRADE_RUBRIC_ID as _DEFAULT_GRADE_RUBRIC_ID,
)
from presentation_dependence.rerankers.grade_rubrics import (
    DEFAULT_INSTRUCTION as _DEFAULT_INSTRUCTION,
)
from presentation_dependence.rerankers.grade_rubrics import build_doc_blocks as _build_doc_blocks
from presentation_dependence.rerankers.grade_rubrics import build_pathc_grade_user_body as _build_pathc_grade_user_body
from presentation_dependence.rerankers.grade_rubrics import resolve_grade_rubric as _resolve_grade_rubric
from presentation_dependence.self_distill.readout import grade_probabilities as _grade_probabilities
from presentation_dependence.self_distill.readout import resolve_grade_token_ids as _resolve_grade_token_ids
from presentation_dependence.utils.setup_logging import setup_logging

_PINE_DUMMY_GRADE = "0"
_DOCS_MARKER = "__PRESENTATION_DEPENDENCE_PINE_DOCS__"
_PINE_VLLM_UNSUPPORTED = (
    "PineReranker cannot run faithfully on stock vLLM. PINE changes the Qwen "
    "attention graph itself (document-level bidirectional attention plus "
    "position-stacked RoPE); the repo's vLLM path only supports stock model "
    "forwards with logit readout/prefix caching and cannot pass PINE segment "
    "metadata into a custom attention backend. Use inference_engine='hf', or "
    "set reranker.pine_vllm_mode='hf_fallback' to make a vLLM-oriented config "
    "explicitly run the faithful HF/eager PINE path."
)


@dataclass(frozen=True)
class PinePathCPrompt:
    """Segmented expected-grade prompt consumed by PINE tokenization."""

    prefix_text: str
    doc_texts: list[str]
    suffix_intro_text: str
    slot_markers: list[str]
    dummy_grade: str = _PINE_DUMMY_GRADE

    @property
    def full_text(self) -> str:
        suffix = self.suffix_intro_text
        for marker in self.slot_markers:
            suffix += f"{marker} Grade: {self.dummy_grade}\n"
        return self.prefix_text + "".join(self.doc_texts) + suffix


@dataclass(frozen=True)
class PineEncodedPrompt:
    input_ids: list[int]
    attention_mask: list[int]
    position_ids: list[int]
    all_position_ids: list[int]
    doc_range: list[list[int]]
    doc_mask: list[list[int]]
    doc_select: list[int]
    self_region_mask: list[list[int]]
    slot_read_positions: list[int]


def _encode_segment(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(x) for x in encoded["input_ids"]]


def build_pine_pathc_prompt(
    tokenizer: Any,
    instruction: str | None,
    query: str,
    passages: list[Passage],
    *,
    max_doc_chars: int,
    chat_template_kwargs: dict[str, Any] | None = None,
    grade_rubric_id: str | None = None,
    dummy_grade: str = _PINE_DUMMY_GRADE,
) -> PinePathCPrompt:
    """Build ``prefix, [docs], suffix`` for PINE.

    A marker is threaded through the tokenizer chat template so the document
    region keeps the same system/user/assistant scaffolding as the normal
    expected-grade prompt, while still giving PINE explicit document segments.
    """
    if not passages:
        raise ValueError("PINE prompt requires at least one passage")
    if not dummy_grade:
        raise ValueError("dummy_grade must not be empty")
    rubric = _resolve_grade_rubric(grade_rubric_id or _DEFAULT_GRADE_RUBRIC_ID)
    doc_blocks = _build_doc_blocks(
        [str(passage["text"]) for passage in passages],
        max_doc_chars=max_doc_chars,
    )
    doc_section = "\n".join(doc_blocks)
    user_body = _build_pathc_grade_user_body(
        instruction,
        query,
        [str(passage["text"]) for passage in passages],
        max_doc_chars=max_doc_chars,
        grade_rubric_id=grade_rubric_id or _DEFAULT_GRADE_RUBRIC_ID,
    )
    docs_anchor = "Documents:\n"
    before_docs, after_docs_anchor = user_body.split(docs_anchor, 1)
    rendered_doc_section, after_docs = after_docs_anchor.split("\n\n", 1)
    if rendered_doc_section != doc_section:
        raise ValueError("PINE document segmentation drifted from expected-grade prompt rendering")
    segmented_user_body = before_docs + docs_anchor + _DOCS_MARKER + "\n\n" + after_docs
    chat_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": rubric.system_prompt},
            {"role": "user", "content": segmented_user_body},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    if _DOCS_MARKER not in chat_text:
        raise ValueError("chat template did not preserve PINE document marker")
    prefix_text, post_docs_text = chat_text.split(_DOCS_MARKER, 1)
    return PinePathCPrompt(
        prefix_text=prefix_text,
        doc_texts=[block + ("\n" if i < len(doc_blocks) - 1 else "") for i, block in enumerate(doc_blocks)],
        suffix_intro_text=post_docs_text + "Grades:\n",
        slot_markers=[f"[{i}]" for i in range(1, len(passages) + 1)],
        dummy_grade=str(dummy_grade),
    )


def encode_pine_pathc_prompt(
    tokenizer: Any,
    prompt: PinePathCPrompt,
    *,
    max_length: int,
) -> PineEncodedPrompt:
    """Tokenize ``prefix + [docs] + suffix`` with official PINE metadata.

    Adapted from official ``ps_call``. Reject overflow instead of truncating,
    because truncating document segments would corrupt PINE's doc-range and
    position-stack metadata.
    """
    prefix_ids = _encode_segment(tokenizer, prompt.prefix_text)
    doc_ids = [_encode_segment(tokenizer, text) for text in prompt.doc_texts]
    if any(len(ids) == 0 for ids in doc_ids):
        raise ValueError("PINE document segment tokenized to zero tokens")

    input_ids = list(prefix_ids)
    all_position_ids = list(range(len(prefix_ids)))
    if not all_position_ids:
        raise ValueError("PINE prefix tokenized to zero tokens")

    total_doc_len = sum(len(ids) for ids in doc_ids)
    doc_end_position = all_position_ids[-1] + total_doc_len + 1
    doc_start_abs = len(input_ids)
    max_doc_len = max(len(ids) for ids in doc_ids)
    doc_range: list[list[int]] = []
    doc_mask: list[list[int]] = []
    doc_select: list[int] = []
    self_region_mask = [[0 for _ in range(total_doc_len)] for _ in range(total_doc_len)]

    local_cursor = 0
    abs_cursor = doc_start_abs
    for doc_idx, ids in enumerate(doc_ids):
        input_ids.extend(ids)
        all_position_ids.extend(range(doc_end_position - len(ids), doc_end_position))
        padded_range = [0] * max_doc_len
        padded_range[: len(ids)] = list(range(abs_cursor, abs_cursor + len(ids)))
        padded_mask = [0] * max_doc_len
        padded_mask[: len(ids)] = [1] * len(ids)
        doc_range.append(padded_range)
        doc_mask.append(padded_mask)
        doc_select.extend([doc_idx] * len(ids))
        for r in range(local_cursor, local_cursor + len(ids)):
            for c in range(local_cursor, local_cursor + len(ids)):
                self_region_mask[r][c] = 1
        local_cursor += len(ids)
        abs_cursor += len(ids)

    suffix_position_start = all_position_ids[-1] + 1
    suffix_ids: list[int] = []
    intro_ids = _encode_segment(tokenizer, prompt.suffix_intro_text)
    suffix_ids.extend(intro_ids)
    slot_read_positions: list[int] = []
    for marker in prompt.slot_markers:
        slot_prefix_ids = _encode_segment(tokenizer, f"{marker} Grade: ")
        suffix_ids.extend(slot_prefix_ids)
        slot_read_positions.append(len(input_ids) + len(suffix_ids) - 1)
        suffix_ids.extend(_encode_segment(tokenizer, prompt.dummy_grade + "\n"))
    input_ids.extend(suffix_ids)
    all_position_ids.extend(range(suffix_position_start, suffix_position_start + len(suffix_ids)))

    if len(input_ids) > max_length:
        raise ValueError(
            f"PINE prompt length {len(input_ids)} exceeds max_length={max_length}. "
            "Reduce reranker.max_doc_chars or docs_per_score_forward; PINE does not use tail truncation."
        )
    return PineEncodedPrompt(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        position_ids=list(range(len(input_ids))),
        all_position_ids=all_position_ids,
        doc_range=doc_range,
        doc_mask=doc_mask,
        doc_select=doc_select,
        self_region_mask=self_region_mask,
        slot_read_positions=slot_read_positions,
    )


def _gather_cos_sin(cos: Any, sin: Any, positions: Any) -> tuple[Any, Any]:
    # PINE's official implementation is single-input only. Indexing the batch-0
    # RoPE table keeps this adaptation narrow and easy to audit.
    return cos[0][positions], sin[0][positions]


def _apply_rope_with_positions(x: Any, cos: Any, sin: Any) -> Any:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = x.new_empty(x.shape)
    rotated[..., : x.shape[-1] // 2] = -x2
    rotated[..., x.shape[-1] // 2 :] = x1
    return (x * cos) + (rotated * sin)


def _repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _pine_qwen_attention_forward(  # noqa: C901
    self: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any],
    attention_mask: Any | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, Any | None]:
    """Qwen2/Qwen3 eager attention with PINE document-level position stacking."""
    doc_range = kwargs.get("pine_doc_range")
    if doc_range is None:
        original = self._slm_pine_original_forward
        return original(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    if past_key_values is not None:
        raise ValueError("PINE attention path does not support KV cache / generation")

    import torch

    doc_mask = kwargs["pine_doc_mask"]
    doc_select = kwargs["pine_doc_select"]
    all_position_ids = kwargs["pine_all_position_ids"]
    self_region_mask = kwargs["pine_self_region_mask"]

    bsz, q_len, _ = hidden_states.size()
    if bsz != 1:
        raise ValueError("PINE attention path supports batch size 1 only")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos_table, sin_table = position_embeddings
    num_docs, _max_doc_len = doc_range.shape
    doc_st = doc_range[:, 0]
    doc_ed = torch.max(doc_range, dim=-1).values + 1
    doc_len = doc_mask.sum(dim=-1)
    if torch.any(doc_len <= 0):
        raise ValueError("PINE document length must be positive for every segment")
    doc_region_start = int(doc_st[0].item())
    doc_region_end = int(doc_ed[-1].item())
    doc_range_region = (doc_range - doc_range[0, 0]) * doc_mask
    n_heads = int(query_states.shape[1])
    scale = float(getattr(self, "scaling", self.head_dim**-0.5))

    if num_docs == 1:
        position_shift = torch.zeros((bsz, n_heads, 1, 1), dtype=torch.long, device=query_states.device)
    elif num_docs == 2:
        position_shift = torch.zeros((bsz, n_heads, num_docs, num_docs), dtype=torch.long, device=query_states.device)
        position_shift[:, :, 0, 1] = doc_len[0]
        position_shift[:, :, 1, 0] = doc_len[1]
    else:
        key_states_no_rope = _repeat_kv(key_states, self.num_key_value_groups)
        query_states_region = query_states[:, :, doc_region_start:doc_region_end, :]
        key_states_region = key_states_no_rope[:, :, doc_region_start:doc_region_end, :]
        attn_weights = torch.matmul(query_states_region, key_states_region.transpose(2, 3)) * scale
        region_mask = self_region_mask.to(attn_weights.dtype)
        region_mask = region_mask.masked_fill(self_region_mask == 1, torch.finfo(region_mask.dtype).min)
        attn_weights = attn_weights + region_mask[None, None, :, :]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights_per_doc = attn_weights[:, :, :, doc_range_region]
        attn_weights_per_doc *= doc_mask[None, None, None, :, :]
        attn_weights_per_doc = attn_weights_per_doc.sum(dim=-1)
        attn_weights_per_doc = attn_weights_per_doc / doc_len[None, None, None, :]
        attn_weights_doc2doc = attn_weights_per_doc[:, :, doc_range_region, :]
        attn_weights_doc2doc *= doc_mask[None, None, :, :, None]
        attn_weights_doc2doc = attn_weights_doc2doc.sum(dim=-2)
        diag_mask = torch.eye(num_docs, dtype=torch.bool, device=attn_weights_doc2doc.device)
        attn_weights_doc2doc = attn_weights_doc2doc.masked_fill(
            diag_mask[None, None, :, :],
            torch.finfo(attn_weights_doc2doc.dtype).max,
        )
        sort_idx = torch.argsort(attn_weights_doc2doc, dim=-1, stable=True, descending=True)
        sort_inv = torch.argsort(sort_idx, dim=-1)
        position_shift = doc_len[sort_idx]
        position_shift = torch.cumsum(position_shift, dim=-1) - position_shift
        position_shift = torch.gather(position_shift, dim=-1, index=sort_inv)

    attn_outputs_list: list[Any] = []
    if doc_region_start > 0:
        prefix_q = query_states[:, :, :doc_region_start, :]
        prefix_k = _repeat_kv(key_states[:, :, :doc_region_start, :], self.num_key_value_groups)
        prefix_v = _repeat_kv(value_states[:, :, :doc_region_start, :], self.num_key_value_groups)
        prefix_pos = all_position_ids[:, :doc_region_start]
        prefix_cos, prefix_sin = _gather_cos_sin(cos_table, sin_table, prefix_pos)
        prefix_cos = prefix_cos.unsqueeze(1)
        prefix_sin = prefix_sin.unsqueeze(1)
        prefix_q = _apply_rope_with_positions(prefix_q, prefix_cos, prefix_sin)
        prefix_k = _apply_rope_with_positions(prefix_k, prefix_cos, prefix_sin)
        prefix_attn = torch.matmul(prefix_q, prefix_k.transpose(2, 3)) * scale
        if attention_mask is not None:
            prefix_attn = prefix_attn + attention_mask[:, :, :doc_region_start, :doc_region_start]
        prefix_attn = torch.nn.functional.softmax(prefix_attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_outputs_list.append(torch.matmul(prefix_attn, prefix_v))

    for doc_id in range(num_docs):
        doc_position_shift = position_shift[:, :, doc_id, :].clone()
        doc_position_shift = doc_position_shift[:, :, doc_select]
        doc_position = all_position_ids[:, None, :].expand(bsz, n_heads, q_len).clone()
        doc_position[:, :, doc_region_start:doc_region_end] -= doc_position_shift
        doc_position = doc_position[:, :, :doc_region_end]
        doc_cos, doc_sin = _gather_cos_sin(cos_table, sin_table, doc_position[0])
        doc_cos = doc_cos.unsqueeze(0)
        doc_sin = doc_sin.unsqueeze(0)
        st = int(doc_st[doc_id].item())
        ed = int(doc_ed[doc_id].item())
        doc_q = query_states[:, :, st:ed, :]
        doc_q = _apply_rope_with_positions(doc_q, doc_cos[:, :, st:ed, :], doc_sin[:, :, st:ed, :])
        doc_k = _repeat_kv(key_states[:, :, :doc_region_end, :], self.num_key_value_groups)
        doc_k = _apply_rope_with_positions(doc_k, doc_cos, doc_sin)
        doc_v = _repeat_kv(value_states[:, :, :doc_region_end, :], self.num_key_value_groups)
        doc_attn = torch.matmul(doc_q, doc_k.transpose(2, 3)) * scale
        doc_mask_causal = doc_position[:, :, st:ed].unsqueeze(-1) < doc_position.unsqueeze(-2)
        doc_attn = doc_attn.masked_fill(doc_mask_causal, torch.finfo(doc_attn.dtype).min)
        doc_attn = torch.nn.functional.softmax(doc_attn, dim=-1, dtype=torch.float32).to(doc_q.dtype)
        attn_outputs_list.append(torch.matmul(doc_attn, doc_v))

    suffix_q_len = q_len - doc_region_end
    if suffix_q_len > 0:
        suffix_query_states = query_states[:, :, doc_region_end:, :]
        key_states_no_rope = _repeat_kv(key_states, self.num_key_value_groups)
        key_states_region = key_states_no_rope[:, :, doc_region_start:doc_region_end, :]
        attn_weights = torch.matmul(suffix_query_states, key_states_region.transpose(2, 3)) * scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_per_doc = attn_weights[:, :, :, doc_range_region]
        attn_weights_per_doc *= doc_mask[None, None, None, :, :]
        attn_weights_per_doc = attn_weights_per_doc.sum(dim=-1)
        attn_weights_per_doc = attn_weights_per_doc / doc_len[None, None, None, :]
        sort_idx = torch.argsort(attn_weights_per_doc, dim=-1, stable=True, descending=True)
        sort_inv = torch.argsort(sort_idx, dim=-1)
        suffix_shift = doc_len[sort_idx]
        suffix_shift = torch.cumsum(suffix_shift, dim=-1) - suffix_shift
        suffix_shift = torch.gather(suffix_shift, dim=-1, index=sort_inv)
        suffix_shift = suffix_shift[:, :, :, doc_select]
        dynamic_position = all_position_ids[:, None, None, :].expand(bsz, n_heads, suffix_q_len, q_len).clone()
        dynamic_position[:, :, :, doc_region_start:doc_region_end] -= suffix_shift
        dynamic_cos, dynamic_sin = _gather_cos_sin(cos_table, sin_table, dynamic_position[0])
        dynamic_cos = dynamic_cos.unsqueeze(0)
        dynamic_sin = dynamic_sin.unsqueeze(0)
        query_positions = all_position_ids[:, doc_region_end:]
        q_cos, q_sin = _gather_cos_sin(cos_table, sin_table, query_positions)
        q_cos = q_cos.unsqueeze(1)
        q_sin = q_sin.unsqueeze(1)
        dynamic_q = _apply_rope_with_positions(suffix_query_states, q_cos, q_sin).unsqueeze(-2)
        dynamic_k = _repeat_kv(key_states, self.num_key_value_groups).unsqueeze(2)
        dynamic_k = _apply_rope_with_positions(dynamic_k, dynamic_cos, dynamic_sin)
        dynamic_v = _repeat_kv(value_states, self.num_key_value_groups)
        dynamic_attn = torch.matmul(dynamic_q, dynamic_k.transpose(3, 4)).squeeze(-2) * scale
        if attention_mask is not None:
            dynamic_attn = dynamic_attn + attention_mask[:, :, doc_region_end:, :]
        dynamic_attn = torch.nn.functional.softmax(dynamic_attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_outputs_list.append(torch.matmul(dynamic_attn, dynamic_v))

    attn_output = torch.cat(attn_outputs_list, dim=2)
    if attn_output.size() != (bsz, n_heads, q_len, self.head_dim):
        raise ValueError(f"PINE attention output shape mismatch: got {tuple(attn_output.size())}")
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def patch_qwen_model_for_pine(model: Any) -> None:
    """Patch Qwen2/Qwen3 attention modules to consume PINE segment kwargs."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("PINE currently supports Qwen-style causal LM models with model.layers")
    patched = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if not hasattr(attn, "_slm_pine_original_forward"):
            attn._slm_pine_original_forward = attn.forward
            attn.forward = MethodType(_pine_qwen_attention_forward, attn)
        patched += 1
    if patched == 0:
        raise ValueError("PINE could not find any Qwen attention layers to patch")


class PineHFPathCEngine:
    """HF/eager single-input scorer for segmented PINE expected-grade prompts."""

    def __init__(self, *, model: Any, tokenizer: Any, max_length: int, enable_pine_attention: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enable_pine_attention = bool(enable_pine_attention)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def score_prompt_probabilities(
        self,
        prompt: PinePathCPrompt,
        grade_token_ids: Sequence[int],
    ) -> list[list[float]]:
        import torch

        encoded = encode_pine_pathc_prompt(self.tokenizer, prompt, max_length=self.max_length)
        device = self.model.device
        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long, device=device)
        # Stock Qwen builds the causal mask and RoPE table from normal sequence
        # positions. PINE's stacked positions are carried separately and consumed
        # by the patched attention forward.
        position_ids = torch.tensor([encoded.position_ids], dtype=torch.long, device=device)
        pine_kwargs: dict[str, Any] = {"use_cache": False}
        if self.enable_pine_attention:
            pine_kwargs.update(
                {
                    "pine_doc_range": torch.tensor(encoded.doc_range, dtype=torch.long, device=device),
                    "pine_doc_mask": torch.tensor(encoded.doc_mask, dtype=torch.long, device=device),
                    "pine_doc_select": torch.tensor(encoded.doc_select, dtype=torch.long, device=device),
                    "pine_all_position_ids": torch.tensor([encoded.all_position_ids], dtype=torch.long, device=device),
                    "pine_self_region_mask": torch.tensor(encoded.self_region_mask, dtype=torch.bool, device=device),
                }
            )
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **pine_kwargs,
            )
        logits = outputs.logits[0]
        pos_idx = torch.tensor(encoded.slot_read_positions, dtype=torch.long, device=logits.device)
        selected = logits.index_select(0, pos_idx)
        probs = _grade_probabilities(selected, grade_token_ids)
        return [[float(v) for v in row] for row in probs.detach().cpu().float().tolist()]


def resolve_pine_execution_engine(reranker_config: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve PINE execution engine without silently faking vLLM support.

    Returns ``(engine_kind, note)``. The only faithful in-repo execution engine
    is HF/eager. ``inference_engine: vllm`` is accepted only when the config
    explicitly opts into ``pine_vllm_mode: hf_fallback``.
    """
    engine_kind = str(reranker_config.get("inference_engine", "hf")).strip().lower()
    if engine_kind == "hf":
        return "hf", None
    if engine_kind != "vllm":
        raise ValueError("PineReranker inference_engine must be one of 'hf' or 'vllm'")

    mode = str(reranker_config.get("pine_vllm_mode", "reject")).strip().lower()
    if mode in {"reject", "strict"}:
        raise ValueError(_PINE_VLLM_UNSUPPORTED)
    if mode in {"hf_fallback", "fallback_hf"}:
        return (
            "hf",
            "Requested inference_engine='vllm' for PineReranker, but stock vLLM cannot "
            "run PINE's custom attention. Running the faithful HF/eager PINE path "
            "because reranker.pine_vllm_mode='hf_fallback' was set.",
        )
    raise ValueError("reranker.pine_vllm_mode must be one of 'reject' or 'hf_fallback'")


class PineReranker(Reranker):
    """Wang et al.'s PINE applied to a Qwen-family expected-grade scorer."""

    paradigm = "batched_pointwise"
    supports_query_batching = False

    def __init__(self, config: dict):  # noqa: C901
        super().__init__(config)
        self.logger = setup_logging(self.__class__.__name__, config)
        rc = self.reranker_config
        self.inference_engine_kind, fallback_note = resolve_pine_execution_engine(rc)
        if fallback_note:
            self.logger.warning(fallback_note)
        if str(rc.get("attn_implementation", "eager")).strip().lower() != "eager":
            raise ValueError("PineReranker requires reranker.attn_implementation='eager'")

        self.model_name = str(rc.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"))
        self.device = _resolve_device(rc.get("device", "auto"))
        self.dtype = _resolve_dtype(rc.get("dtype", "auto"), self.device)
        self.revision = rc.get("revision")
        self.instruction = rc.get("instruction", _DEFAULT_INSTRUCTION)
        self.grade_rubric_id = str(rc.get("grade_rubric_id", _DEFAULT_GRADE_RUBRIC_ID)).strip()
        _resolve_grade_rubric(self.grade_rubric_id)
        self.max_length = int(rc.get("max_length", 8192))
        self.docs_per_score_forward = int(rc.get("docs_per_score_forward", 20))
        if self.docs_per_score_forward <= 0:
            raise ValueError("reranker.docs_per_score_forward must be positive")
        self.batch_size = 1
        self.max_doc_chars = int(rc.get("max_doc_chars", 1200))
        self.apply_attention_patch = bool(rc.get("apply_attention_patch", True))
        self.grade_skeleton_dummy = str(rc.get("grade_skeleton_dummy", _PINE_DUMMY_GRADE))
        if not self.grade_skeleton_dummy:
            raise ValueError("reranker.grade_skeleton_dummy must not be empty")
        prefix_mode = str(rc.get("grade_prefix_mode", "cumulative")).strip().lower()
        if prefix_mode not in {"cumulative", "pine"}:
            raise ValueError("PineReranker supports only cumulative grade skeletons")
        chat_template_kwargs = rc.get("chat_template_kwargs") or {}
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("reranker.chat_template_kwargs must be a mapping when provided")
        self.chat_template_kwargs = dict(chat_template_kwargs)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_kwargs: dict[str, Any] = {"padding_side": "left"}
        if self.revision:
            tok_kwargs["revision"] = self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.grade_token_ids = _resolve_grade_token_ids(self.tokenizer)

        model_kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "attn_implementation": "eager",
        }
        if self.revision:
            model_kwargs["revision"] = self.revision
        if "trust_remote_code" in rc:
            model_kwargs["trust_remote_code"] = bool(rc["trust_remote_code"])
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.apply_attention_patch:
            patch_qwen_model_for_pine(self.model)
        else:
            self.logger.warning(
                "PineReranker running with apply_attention_patch=false; this is an ablation that keeps "
                "the PINE prompt/readout but disables PINE's custom attention mechanism."
            )
        self.model.to(self.device)
        self.model.eval()
        self.engine = PineHFPathCEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            enable_pine_attention=self.apply_attention_patch,
        )

    @staticmethod
    def _scores_from_grade_probabilities(vectors: list[list[float]]) -> list[float]:
        return [sum(float(i) * float(p) for i, p in enumerate(vector)) / 3.0 for vector in vectors]

    def _score_chunk_grade_probabilities(self, query: Query, chunk: list[Passage]) -> list[list[float]]:
        prompt = build_pine_pathc_prompt(
            self.tokenizer,
            self.instruction,
            query["text"],
            chunk,
            max_doc_chars=self.max_doc_chars,
            chat_template_kwargs=self.chat_template_kwargs,
            grade_rubric_id=self.grade_rubric_id,
            dummy_grade=self.grade_skeleton_dummy,
        )
        return self.engine.score_prompt_probabilities(prompt, self.grade_token_ids)

    def rank(self, query: Query, passages: list[Passage]) -> RankResult:
        if not passages:
            return {
                "top_k_psgs": [],
                "scores_init_order": [],
                "grade_probabilities_init_order": [],
                "prompting_runtimes": [0.0],
                "paradigm": self.paradigm,
            }
        t0 = time.perf_counter()
        scores: list[float] = []
        vectors: list[list[float]] = []
        for start in range(0, len(passages), self.docs_per_score_forward):
            chunk = passages[start : start + self.docs_per_score_forward]
            chunk_vectors = self._score_chunk_grade_probabilities(query, chunk)
            vectors.extend(chunk_vectors)
            scores.extend(self._scores_from_grade_probabilities(chunk_vectors))
        elapsed = time.perf_counter() - t0
        result = scores_to_rank_result(scores, passages, elapsed, self.paradigm, model_name=self.__class__.__name__)
        result["grade_probabilities_init_order"] = vectors
        return result
