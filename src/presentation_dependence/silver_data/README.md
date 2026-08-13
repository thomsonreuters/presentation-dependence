# `silver_data/`

Shared silver-label generation support. Both teacher kinds
use `configs/silver/` and the `silver_data` config classifier, but they use
different executors:

| Kind | Discriminator (`teacher.protocol`) | Executor | Output root |
|---|---|---|---|
| Open-weight | `k_shot_bsc` | `self_distill.teacher.KShotBSCTeacher` | `runs/self-distill/<id>/<timestamp>/silver/` |
| Hosted / closed | `closed_model_generated_bsc` | `SilverGenerator` (`silver_data`) | `runs/silver/<id>/<timestamp>/` |

`scripts/run_silver_generation.py` loads a config via `load_silver_config`, then
dispatches on kind. Open-weight teachers score with expected-grade logits over K
shuffled presentations. Hosted teachers score pointwise B=1 through the
hosted endpoint and average generated integer grades.

Hosted generation plus the shared config and transform helpers live here. The
open-weight teacher itself lives under `self_distill/`.

## Shared

- [`config.py`](config.py): resolve, classify, and validate silver configs
  (`SilverConfig`, `SilverConfigKind` = open-weight / hosted). Discriminator is
  `teacher.protocol`; see `configs/silver/_schema.md`.
- [`transforms.py`](transforms.py): deterministic silver-label derivation and
  cohort operations (`derive_silver_shard`, `materialize_qid_delta`,
  `split_silver_cohort`), used by reproduction for QA/response cohorts.

## Hosted generation

- [`generate.py`](generate.py): orchestrator for
  `closed_model_generated_bsc`. Builds pointwise B=1 requests from the
  configured dataloader, hands them to a `SilverClient`, parses teacher output
  with the configured prompt template, and writes resumable JSONL plus a
  manifest. Exposes `SilverGenerator`, `BudgetExceeded`, `RunDirLocked`.
- [`client.py`](client.py): the request, response, and client contract. One
  scoring call carries a query, one candidate, and the prompt framing them.
- [`openai_client.py`](openai_client.py): batched-self-consistency scoring
  against an OpenAI-compatible chat endpoint. Builds the shared grade prompt,
  scores each pool under the configured presentations, parses the answer
  skeleton, and averages. Standard library only; point `OPENAI_BASE_URL` at a
  gateway or a local server, and `OPENAI_API_KEY` at its credential.

The [`prompts/`](prompts/README.md) subpackage holds the pointwise prompt
templates and their parsers.

## Hosted client

The OpenAI-compatible client is optional, provider-neutral infrastructure.
Running it sends configured corpus text to the operator-selected endpoint.
Operators are responsible for ensuring that their corpus use and provider
agreement permit that processing. The source repository includes no credentials
or generated provider output.

Resume only with byte-equivalent config and inputs. Before consuming generated
BSC labels, require each document's raw score vector to have exactly the
configured number of presentations.

## Public API

[`__init__.py`](__init__.py) re-exports the config and transform helpers eagerly;
the generation stack (`SilverGenerator` etc.) is loaded lazily via
`__getattr__` so `import silver_data` does not pull in the eval/reranker
dependency graph.
