"""Reranker class registry.

One entry per file under `presentation_dependence.rerankers.*`. Add a base model by:

1. Dropping a new file under this directory subclassing `Reranker`.
2. Adding one line here.
3. Referencing the class name from a YAML under `configs/experiments/`.
4. Updating `presentation_dependence.eval.tau_psi_geometry` if τ-PSI@B needs a
   dedicated geometry branch or a batched-pointwise inclusion/exclusion.
5. Extending `configs/experiments/_schema.md` when the model introduces new
   reusable config keys.

Lazy-import pattern: we import eagerly at module load so a typo/error in the
registry surfaces on first `import presentation_dependence`, not only when the
offending experiment is run.
"""

from __future__ import annotations

from presentation_dependence.rerankers.base import Reranker
from presentation_dependence.rerankers.capcal import CapCalReranker
from presentation_dependence.rerankers.closed_model_genbsc import ClosedModelGenBscReranker
from presentation_dependence.rerankers.gemma4 import Gemma4GradeReranker
from presentation_dependence.rerankers.granite_41 import Granite41GradeReranker
from presentation_dependence.rerankers.identity import IdentityReranker
from presentation_dependence.rerankers.jina_listwise import JinaListwiseReranker
from presentation_dependence.rerankers.mxbai_pointwise import MxbaiPointwise

# PairRM (response-ranking external anchor). Registered eagerly: the module imports cleanly
# without ``llm_blender`` (the heavy dep is imported lazily inside ``__init__``,
# like MxbaiPointwise), so only *instantiating* it requires the [rerankers] extra.
from presentation_dependence.rerankers.pairrm import PairRMReranker
from presentation_dependence.rerankers.pine import PineReranker
from presentation_dependence.rerankers.qwen3 import Qwen3Reranker
from presentation_dependence.rerankers.qwen3_instruct_grade import Qwen3InstructGradeReranker
from presentation_dependence.rerankers.rankzephyr import RankZephyrReranker
from presentation_dependence.rerankers.skywork_bt import SkyworkBTReranker

RERANKER_CLASSES: dict[str, type[Reranker]] = {
    "CapCalReranker": CapCalReranker,
    "ClosedModelGenBscReranker": ClosedModelGenBscReranker,
    "Gemma4GradeReranker": Gemma4GradeReranker,
    "Granite41GradeReranker": Granite41GradeReranker,
    "IdentityReranker": IdentityReranker,
    "JinaListwiseReranker": JinaListwiseReranker,
    "MxbaiPointwise": MxbaiPointwise,
    "PairRMReranker": PairRMReranker,
    "PineReranker": PineReranker,
    "Qwen3InstructGradeReranker": Qwen3InstructGradeReranker,
    "Qwen3Reranker": Qwen3Reranker,
    "RankZephyrReranker": RankZephyrReranker,
    "SkyworkBTReranker": SkyworkBTReranker,
}


def get_reranker_class(name: str) -> type[Reranker]:
    try:
        return RERANKER_CLASSES[name]
    except KeyError:
        known = ", ".join(sorted(RERANKER_CLASSES))
        raise ValueError(
            f"Unknown reranker class {name!r}. Known classes: {known}. "
            f"Add a new class under src/presentation_dependence/rerankers/ and register it here."
        ) from None
