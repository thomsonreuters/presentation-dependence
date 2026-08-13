"""Redact secrets from run logs, entrypoint logs, and sweep manifests.

A secret reaches this code by being passed as a dotted-key override or an
environment value, which then gets echoed to stdout or written into a sweep
manifest on disk. This module only stops that echo; it does not protect the
value anywhere it is legitimately stored.

Detection is by key name rather than by value, so it catches a credential
whatever it looks like, and it errs toward redacting: any leaf ending in
``_token``, ``_api_key``, ``_secret`` or ``_password`` is treated as sensitive.
"""

from __future__ import annotations


def _leaf_name(key: str) -> str:
    return key.strip().split(".")[-1].lower()


def _is_sensitive_leaf(leaf: str) -> bool:
    if leaf in {
        "hf_token",
        "huggingface_hub_token",
        "hub_token",
        "token",  # bare ``TOKEN`` env vars are almost always credentials
        "password",
        "authorization",
    }:
        return True
    if leaf.endswith("_token"):
        return True
    if leaf.endswith("_api_key") or leaf.endswith("_secret") or leaf.endswith("_password"):
        return True
    if "aws_secret" in leaf or "secret_access" in leaf:
        return True
    return False


def _override_key_is_sensitive(key: str) -> bool:
    return _is_sensitive_leaf(_leaf_name(key))


def redact_override_spec(spec: str) -> str:
    """Redact ``KEY=VALUE`` override strings for logging."""
    if "=" not in spec:
        return spec
    key, _val = spec.split("=", 1)
    if _override_key_is_sensitive(key):
        return f"{key.strip()}=[REDACTED]"
    return spec


def redact_override_list(overrides: list[str]) -> list[str]:
    return [redact_override_spec(s) for s in overrides]


def redact_environment_for_log(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in env.items():
        kl = str(k).lower()
        if _is_sensitive_leaf(kl):
            out[str(k)] = "[REDACTED]"
        else:
            out[str(k)] = str(v)
    return out


def redact_launch_argv_for_log(argv: list[str]) -> list[str]:
    """Redact ``--override VALUE`` pairs in a subprocess argv sequence."""
    out = list(argv)
    i = 0
    while i + 1 < len(out):
        if out[i] == "--override":
            out[i + 1] = redact_override_spec(out[i + 1])
            i += 2
        else:
            i += 1
    return out


def manifest_override_lists_for_disk(overrides: list[str]) -> list[str]:
    """Overrides as stored in sweep ``manifest.json`` (secrets redacted)."""
    return redact_override_list(overrides)
