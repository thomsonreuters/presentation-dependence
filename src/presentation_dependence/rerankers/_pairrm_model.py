"""Vendored ``DebertaV2PairRM`` head for the HF-native PairRM anchor.

PairRM (``llm-blender/PairRM-hf``) is a ``microsoft/deberta-v3-large`` (0.4B)
backbone with a custom pairwise head that reads the ``<|source|>`` /
``<|candidate1|>`` / ``<|candidate2|>`` prefix-token encodings and scores
candidate 1 vs candidate 2 side-by-side (bidirectional cross-attention over the
pair). The official checkpoint requires this exact model class.

Rather than depend on the stale ``llm-blender`` package (which pins an older
``transformers`` that conflicts with the repo tf5 pin), we vendor the class here
per the model card's explicit guidance ("you can also copy the simple definition
of ``DebertaV2PairRM`` ... as your local file, instead of importing it from the
``llm-blender`` package"). It subclasses only **core, stable** transformers
deberta_v2 building blocks, so it loads on modern transformers.

SPDX-License-Identifier: Apache-2.0

Source and modification notice (adapted inference-only implementation):
  https://github.com/yuchenlin/LLM-Blender/blob/5d38ca9528cdeb23e89d40500ced511d08bb5996/llm_blender/pair_ranker/pairrm.py
  (Jiang, Ren & Lin, "LLM-Blender"; Apache-2.0). The training
  ``compute_loss`` branch is dropped, configuration access is defensive, and
  the forward path validates required inputs. The code repository is
  Apache-2.0 while the ``llm-blender/PairRM-hf`` checkpoint repository is
  tagged MIT; this file derives from the former and redistributes no weights.
  See ``THIRD_PARTY_NOTICES.md``; licence text at
  ``third_party_licenses/Apache-2.0.txt``. The reviewed upstream source file
  carries no copyright or attribution header and the repository has no
  ``NOTICE`` file.

Kept in its own module so importing the reranker registry does not eagerly pull
transformers' deberta_v2 modeling code. The reranker imports this lazily in its
``__init__`` (same lazy heavy-import pattern as the other wrappers).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)


class DebertaV2PairRM(DebertaV2PreTrainedModel):
    """PairRM head over a DeBERTa-v2 backbone (eval-only, vendored)."""

    def __init__(self, config):
        super().__init__(config)

        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out

        self.pretrained_model = DebertaV2Model(config)
        self.hidden_size = config.hidden_size

        # ``sep_token_id`` / ``cand_prefix_id`` are unused in the (eval-only)
        # forward and are absent from the PairRM-hf config, so read them
        # defensively; the three prefix ids the forward keys on are required.
        self.sep_token_id = getattr(config, "sep_token_id", None)
        self.source_prefix_id = config.source_prefix_id
        self.cand_prefix_id = getattr(config, "cand_prefix_id", None)
        self.cand1_prefix_id = config.cand1_prefix_id
        self.cand2_prefix_id = config.cand2_prefix_id

        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.hidden_size, 1 * self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, self.n_tasks),
        )
        self.sigmoid = nn.Sigmoid()

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """Score candidate1 vs candidate2: ``logits[i] > 0`` ⇒ cand1 better."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Optional in the signature to match the HF forward contract, but this
        # head reads the prefix tokens and column mask directly, so both are
        # required. ``tokenize_pair`` always supplies them.
        if input_ids is None or attention_mask is None:
            raise ValueError(
                "DebertaV2PairRM.forward requires input_ids and attention_mask; build them with tokenize_pair()."
            )

        # Each row must carry the three prefix tokens (built by ``tokenize_pair``).
        assert all(self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])), "source id missing"
        assert all(self.cand1_prefix_id in input_ids[i] for i in range(input_ids.shape[0])), "cand1 id missing"
        assert all(self.cand2_prefix_id in input_ids[i] for i in range(input_ids.shape[0])), "cand2 id missing"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand1_idxs = torch.where(input_ids == self.cand1_prefix_id)
        cand1_encs = encs[cand1_idxs[0], cand1_idxs[1], :]
        cand2_idxs = torch.where(input_ids == self.cand2_prefix_id)
        cand2_encs = encs[cand2_idxs[0], cand2_idxs[1], :]

        source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
        source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
        left_pred_scores = self.head_layer(source_cand1_encs)
        right_pred_scores = self.head_layer(source_cand2_encs)

        preds = (left_pred_scores - right_pred_scores).mean(dim=-1)
        return SequenceClassifierOutput(
            loss=None,
            logits=preds,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
