"""Input-symmetry render-variant operators for batched-pointwise expected-grade scoring.

Swaps prompt-rendering nuisances (document ID, separator, rubric phrasing) while
holding document order fixed. Index 0 of every variant list is the canonical /
training-seen rendering (numeric IDs, newline separator, relevance_v1).
"""

from __future__ import annotations

import random
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from presentation_dependence.rerankers.grade_rubrics import DEFAULT_GRADE_RUBRIC_ID

if TYPE_CHECKING:
    from presentation_dependence.rerankers.scale_variants import ScaleScheme

RenderStrategy = Literal["id_relabel", "separator", "rubric_paraphrase", "scale"]
IdScheme = Literal["numeric", "alpha", "doc_n", "roman", "rand3"]
SeparatorStyle = Literal["newline", "dash", "hash", "xml", "passage_n"]

#: Label-vocabulary schemes sampled into the training distribution at Stage 2.
#: ``numeric`` is the historical training-seen rendering; ``alpha`` / ``doc_n``
#: are additional *seen* schemes. ``roman`` / ``rand3`` are deliberately held
#: out so the analyzer's "unseen scheme" label-PSI measures generalization
#: rather than memorization.
TRAIN_ID_SCHEME_POOL: tuple[IdScheme, ...] = ("numeric", "alpha", "doc_n")
HELDOUT_ID_SCHEME_POOL: tuple[IdScheme, ...] = ("roman", "rand3")

RENDER_STRATEGIES: tuple[RenderStrategy, ...] = (
    "id_relabel",
    "separator",
    "rubric_paraphrase",
    "scale",
)

_ID_SCHEMES: tuple[IdScheme, ...] = ("numeric", "alpha", "doc_n", "roman", "rand3")
_SEPARATOR_STYLES: tuple[SeparatorStyle, ...] = ("newline", "dash", "hash", "xml", "passage_n")
_RUBRIC_PARAPHRASE_IDS: tuple[str, ...] = (
    DEFAULT_GRADE_RUBRIC_ID,
    "relevance_v1_p1",
    "relevance_v1_p2",
    "relevance_v1_p3",
    "relevance_v1_p4",
)


@dataclass(frozen=True)
class RenderVariant:
    """One meaning-preserving rendering of a B-document expected-grade prompt."""

    label: str
    id_scheme: IdScheme
    separator: SeparatorStyle
    rubric_id: str
    #: ``slot_labels[i]`` is the document-marker string for input slot ``i+1``
    #: (same marker on the doc line and the grade line).
    slot_labels: tuple[str, ...]
    is_canonical: bool
    seen: bool
    strategy: RenderStrategy
    variant_index: int
    rand3_seed: int | None = None
    scale_scheme: ScaleScheme | None = None

    @property
    def id_slot_permutation(self) -> tuple[int, ...]:
        """1-based label indices assigned to each slot (decorrelation readout)."""
        return tuple(_label_index_from_marker(marker, self.id_scheme, self.rand3_seed) for marker in self.slot_labels)


def _alpha_label(n: int) -> str:
    if n < 1:
        raise ValueError("slot index must be >= 1")
    out = ""
    x = n
    while x > 0:
        x, rem = divmod(x - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _roman_label(n: int) -> str:
    pairs = [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]
    if n < 1:
        raise ValueError("slot index must be >= 1")
    out: list[str] = []
    remaining = n
    for value, numeral in pairs:
        while remaining >= value:
            out.append(numeral)
            remaining -= value
    return "".join(out)


def _rand3_token(seed: int, label_index: int) -> str:
    rng = random.Random(seed * 10_000 + label_index)
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(3))


def format_label_index(label_index: int, id_scheme: IdScheme, *, rand3_seed: int | None = None) -> str:
    """Format a 1-based label index under ``id_scheme`` (without brackets)."""
    if label_index < 1:
        raise ValueError("label_index must be >= 1")
    if id_scheme == "numeric":
        return str(label_index)
    if id_scheme == "alpha":
        return _alpha_label(label_index)
    if id_scheme == "doc_n":
        return f"Doc-{label_index}"
    if id_scheme == "roman":
        return _roman_label(label_index)
    if id_scheme == "rand3":
        if rand3_seed is None:
            raise ValueError("rand3_seed required for id_scheme=rand3")
        return _rand3_token(rand3_seed, label_index)
    raise ValueError(f"Unknown id_scheme: {id_scheme!r}")


def _label_index_from_marker(marker: str, id_scheme: IdScheme, rand3_seed: int | None) -> int:  # noqa: C901
    if id_scheme == "numeric":
        return int(marker)
    if id_scheme == "alpha":
        value = 0
        for ch in marker:
            value = value * 26 + (ord(ch) - ord("A") + 1)
        return value
    if id_scheme == "doc_n":
        return int(marker.removeprefix("Doc-"))
    if id_scheme == "roman":
        pairs = [
            ("m", 1000),
            ("cm", 900),
            ("d", 500),
            ("cd", 400),
            ("c", 100),
            ("xc", 90),
            ("l", 50),
            ("xl", 40),
            ("x", 10),
            ("ix", 9),
            ("v", 5),
            ("iv", 4),
            ("i", 1),
        ]
        total = 0
        rest = marker
        while rest:
            matched = False
            for numeral, value in pairs:
                if rest.startswith(numeral):
                    total += value
                    rest = rest[len(numeral) :]
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Could not parse roman marker {marker!r}")
        return total
    if id_scheme == "rand3":
        if rand3_seed is None:
            raise ValueError("rand3_seed required for rand3 scheme")
        for label_index in range(1, 10_000):
            if _rand3_token(rand3_seed, label_index) == marker:
                return label_index
        raise ValueError(f"Could not reverse rand3 marker {marker!r}")
    raise ValueError(f"Unknown id_scheme: {id_scheme!r}")


def bracketed_label(label_index: int, id_scheme: IdScheme, *, rand3_seed: int | None = None) -> str:
    """Bracketed marker used on doc + grade lines (e.g. ``[A]``, ``[Doc-2]``)."""
    if id_scheme == "passage_n":
        raise ValueError("passage_n uses Passage N: prefix, not bracketed labels")
    inner = format_label_index(label_index, id_scheme, rand3_seed=rand3_seed)
    if id_scheme == "doc_n":
        return f"[{inner}]"
    return f"[{inner}]"


def decorrelated_permutation(n_slots: int, seed: int) -> tuple[int, ...]:
    """Return a deterministic permutation of ``1..n_slots`` (slot-decorrelated IDs)."""
    if n_slots <= 0:
        return ()
    order = list(range(1, n_slots + 1))
    random.Random(seed).shuffle(order)
    if order == list(range(1, n_slots + 1)) and n_slots > 1:
        order[0], order[1] = order[1], order[0]
    return tuple(order)


def slot_labels_for_chunk(
    n_slots: int,
    *,
    id_scheme: IdScheme,
    id_slot_permutation: tuple[int, ...] | None,
    rand3_seed: int | None = None,
) -> tuple[str, ...]:
    """Markers for each slot; permutation assigns label identity independently of slot."""
    if n_slots <= 0:
        return ()
    if id_slot_permutation is None:
        perm = tuple(range(1, n_slots + 1))
    else:
        if len(id_slot_permutation) != n_slots:
            raise ValueError(f"id_slot_permutation length {len(id_slot_permutation)} != n_slots {n_slots}")
        perm = id_slot_permutation
    return tuple(bracketed_label(label_index, id_scheme, rand3_seed=rand3_seed) for label_index in perm)


def sample_label_view(
    n_slots: int,
    *,
    seed: int,
    pool: Sequence[IdScheme] = TRAIN_ID_SCHEME_POOL,
    decorrelate: bool = True,
) -> tuple[tuple[str, ...], IdScheme]:
    """Sample one ``id_scheme`` from ``pool`` and build its per-slot markers.

    Used by the Stage-2 view builder to inject label-vocabulary variation into
    the consistency penalty: each shuffled view independently samples a marker
    scheme so the MSE that already forces order-invariance now also forces
    label-invariance.

    ``decorrelate=True`` assigns label identity independently of slot position
    (via :func:`decorrelated_permutation`) so the model cannot shortcut
    ``label == slot index``. Fully deterministic in ``seed`` so a training run
    is reproducible across DDP ranks and resumed checkpoints.

    Returns the ``(slot_labels, id_scheme)`` pair; ``slot_labels`` is empty when
    ``n_slots <= 0``.
    """
    pool_t = tuple(pool)
    if not pool_t:
        raise ValueError("scheme pool must be non-empty")
    if n_slots <= 0:
        scheme = random.Random(seed).choice(pool_t)
        return (), scheme
    rng = random.Random(seed)
    scheme = rng.choice(pool_t)
    rand3_seed = seed if scheme == "rand3" else None
    perm = decorrelated_permutation(n_slots, seed) if decorrelate else None
    labels = slot_labels_for_chunk(
        n_slots,
        id_scheme=scheme,
        id_slot_permutation=perm,
        rand3_seed=rand3_seed,
    )
    return labels, scheme


def canonical_render_variant(*, n_slots: int = 20) -> RenderVariant:
    """Training-seen rendering: numeric IDs, newline separator, relevance_v1."""
    labels = slot_labels_for_chunk(n_slots, id_scheme="numeric", id_slot_permutation=None)
    return RenderVariant(
        label="canonical",
        id_scheme="numeric",
        separator="newline",
        rubric_id=DEFAULT_GRADE_RUBRIC_ID,
        slot_labels=labels,
        is_canonical=True,
        seen=True,
        strategy="id_relabel",
        variant_index=0,
        rand3_seed=None,
    )


def generate_render_variants(
    strategy: RenderStrategy,
    K: int,
    seeds: list[int],
    *,
    n_slots: int = 20,
) -> list[RenderVariant]:
    """Deterministic variant list; index 0 is always canonical on the varied axis."""
    if K <= 0:
        raise ValueError("K must be positive")
    if len(seeds) != K:
        raise ValueError(f"seeds length ({len(seeds)}) must equal K ({K})")

    if strategy == "id_relabel":
        return _id_relabel_variants(K, seeds, n_slots=n_slots)
    if strategy == "separator":
        return _separator_variants(K, seeds, n_slots=n_slots)
    if strategy == "rubric_paraphrase":
        return _rubric_paraphrase_variants(K, seeds, n_slots=n_slots)
    if strategy == "scale":
        from presentation_dependence.rerankers.scale_variants import generate_scale_variants

        return generate_scale_variants(K, seeds, n_slots=n_slots)
    raise ValueError(f"Unknown render strategy: {strategy!r}")


def _id_relabel_variants(K: int, seeds: list[int], *, n_slots: int) -> list[RenderVariant]:
    schemes = _ID_SCHEMES[:K]
    out: list[RenderVariant] = []
    for idx, (scheme, seed) in enumerate(zip(schemes, seeds, strict=False)):
        rand3_seed = seed if scheme == "rand3" else None
        perm = None if idx == 0 else decorrelated_permutation(n_slots, seed)
        labels = slot_labels_for_chunk(
            n_slots,
            id_scheme=scheme,
            id_slot_permutation=perm,
            rand3_seed=rand3_seed,
        )
        out.append(
            RenderVariant(
                label=f"id_{scheme}" if idx else "canonical",
                id_scheme=scheme,
                separator="newline",
                rubric_id=DEFAULT_GRADE_RUBRIC_ID,
                slot_labels=labels,
                is_canonical=idx == 0,
                seen=idx == 0,
                strategy="id_relabel",
                variant_index=idx,
                rand3_seed=rand3_seed,
            )
        )
    return out


def _separator_variants(K: int, seeds: list[int], *, n_slots: int) -> list[RenderVariant]:
    separators = _SEPARATOR_STYLES[:K]
    labels = slot_labels_for_chunk(n_slots, id_scheme="numeric", id_slot_permutation=None)
    out: list[RenderVariant] = []
    for idx, (separator, _seed) in enumerate(zip(separators, seeds, strict=False)):
        out.append(
            RenderVariant(
                label=f"sep_{separator}" if idx else "canonical",
                id_scheme="numeric",
                separator=separator,
                rubric_id=DEFAULT_GRADE_RUBRIC_ID,
                slot_labels=labels,
                is_canonical=idx == 0,
                seen=idx == 0,
                strategy="separator",
                variant_index=idx,
            )
        )
    return out


def _rubric_paraphrase_variants(K: int, seeds: list[int], *, n_slots: int) -> list[RenderVariant]:
    rubric_ids = _RUBRIC_PARAPHRASE_IDS[:K]
    labels = slot_labels_for_chunk(n_slots, id_scheme="numeric", id_slot_permutation=None)
    out: list[RenderVariant] = []
    for idx, (rubric_id, _seed) in enumerate(zip(rubric_ids, seeds, strict=False)):
        out.append(
            RenderVariant(
                label=f"rubric_{rubric_id}" if idx else "canonical",
                id_scheme="numeric",
                separator="newline",
                rubric_id=rubric_id,
                slot_labels=labels,
                is_canonical=idx == 0,
                seen=idx == 0,
                strategy="rubric_paraphrase",
                variant_index=idx,
            )
        )
    return out
