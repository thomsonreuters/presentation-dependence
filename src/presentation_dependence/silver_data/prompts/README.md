# `silver_data/prompts/`

Pointwise (B=1) prompt templates and parsers for silver-label generation: one
query/document pair per teacher judgement. Templates define the audit prompt and
the parser that turns teacher output into a normalized label.

## Files

- [`templates.py`](templates.py): the template registry and parsers.
  `PromptTemplate` (id, text, parser) and `PromptParseResult` (parsed and
  normalized score), with lookup via `get_prompt_template`.


[`__init__.py`](__init__.py) re-exports `PromptTemplate`, `PromptParseResult`,
and `get_prompt_template`.
