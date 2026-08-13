#!/usr/bin/env bash
# Bootstrap the uv environment without mutating git state.

set -euo pipefail

FULL=false
INSTALL_KERNEL=true

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--full] [--no-kernel]

  default      Install the development/check environment.
  --full       Add training, reranker, analysis, and dense extras.
  --no-kernel  Skip Jupyter kernel registration.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=true ;;
    --no-kernel) INSTALL_KERNEL=false ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/
and rerun this script.
EOF
  exit 2
fi

if [[ -n "${VIRTUAL_ENV:-}" ]] && declare -F deactivate >/dev/null 2>&1; then
  echo "Deactivating virtualenv ${VIRTUAL_ENV}"
  deactivate
fi

sync_args=(--extra dev)
if [[ "${FULL}" == true ]]; then
  sync_args+=(
    --extra training
    --extra rerankers
    --extra analysis
    --extra dense
  )
fi

echo "Syncing frozen Python 3.11 environment: uv sync --frozen ${sync_args[*]}"
uv sync --frozen "${sync_args[@]}"

if [[ "${INSTALL_KERNEL}" == true ]]; then
  uv run poe install-experiment-kernel
fi

echo
echo "Environment ready."
if [[ "${FULL}" == true ]]; then
  cat <<'EOF'
Next steps:
  1. Copy .env.example to .env and fill in the credentials you need.
  2. Preview and materialize public automated data:
       uv run python scripts/setup_reproduction_data.py plan
       uv run python scripts/setup_reproduction_data.py run
  3. Run code/declaration checks:
       uv run python scripts/study.py structural audit
       uv run python scripts/study.py representative validate

After manual/gated inputs are provisioned, validate the full population and
source lock:
  uv run python scripts/setup_reproduction_data.py validate
  uv run python scripts/source_lock.py verify
EOF
else
  echo "For training, rerankers, analysis, and dense extras, rerun: ./setup.sh --full"
fi
