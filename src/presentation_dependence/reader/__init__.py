"""Evaluate stored QA scorer rankings with a frozen downstream reader.

The reader consumes each scorer ranking's top-k passages as input and reports
answer EM/F1 and answer stability across scorer-input permutations. Reader
outputs are evaluation metrics only; they are not training targets.

Protocol invariants:

- Use greedy decoding with a pinned prompt.
- Preserve scorer output order as reader input order.
- Report per-query answer quality and across-permutation stability separately.
"""

from __future__ import annotations
