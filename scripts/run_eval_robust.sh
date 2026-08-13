#!/usr/bin/env bash
# Resume-safe laptop launcher for eval configs (`run_experiment` + `run_eval`),
# for paid closed-model API rerankers where resume-on-crash matters.
#
# Safeguards:
#   * `caffeinate -dimsu`   prevent display/idle/disk/system sleep (AC power).
#                           macOS only; skipped where it is unavailable
#   * `nohup` + `disown`    detach from the terminal so closing it (or the lid,
#                           on power) does not SIGHUP the run
#   * pidfile               refuse to double-launch the same config
#   * bounded retry loop    re-run on transient failure with exponential backoff
#   * resume in one dir      every attempt reuses a single `runs/<id>/<ts>/` dir
#                           via `run_experiment.py --run-dir`, so completed
#                           per-query results are skipped and only missing qids
#                           are re-ranked
#   * auth-aware stop       expired or denied credentials are not retried; the
#                           launcher exits for credential renewal and a resumed
#                           relaunch
#
# Usage:
#   export OPENAI_API_KEY=...                  # provider auth (closed-model rerankers)
#   scripts/run_eval_robust.sh example-passage-mxbai-large-v2
#
#   # Re-launch after a failure / re-login: same command. It auto-resumes the
#   # most recent dir for this config (or pass an explicit one):
#   scripts/run_eval_robust.sh example-passage-mxbai-large-v2 --resume
#   RESUME_RUN_DIR=runs/<id>/<ts> scripts/run_eval_robust.sh <id>
#
# Notes:
#   * On macOS `caffeinate -dimsu` keeps the box awake while plugged in OR with
#     the lid open. On battery + lid closed, Apple Silicon sleeps regardless.
#     Keep the laptop plugged in for the duration.
#   * PSI configs (a `robustness:` block) are refused here. Use
#     `scripts/run_eval_robust_psi.sh` instead: it resumes base scoring into one
#     run dir and resumes PSI per query/permutation so retries do not re-pay.
#
# Logs:
#   * Inner python log:   runs/_eval_launcher_logs/<config>_<ts>.python.log
#   * Worker script:      runs/_eval_launcher_logs/<config>_<ts>.worker.sh
#   * Pidfile (launcher): runs/_eval_launcher_logs/<config>.pid

set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "usage: $0 CONFIG_ID [--resume]" >&2
  exit 2
fi

CONFIG="$1"
RESUME_FLAG="${2:-}"
if [ -n "$RESUME_FLAG" ] && [ "$RESUME_FLAG" != "--resume" ]; then
  echo "usage: $0 CONFIG_ID [--resume]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/runs/_eval_launcher_logs"
mkdir -p "$LOG_DIR"

if [[ "$CONFIG" == *.yaml || -f "$CONFIG" ]]; then
  CONFIG_PATH="$CONFIG"
else
  CONFIG_PATH="$REPO_ROOT/configs/experiments/${CONFIG}.yaml"
fi
if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: experiment config not found: $CONFIG_PATH" >&2
  exit 2
fi

# Inspect the config: id, reranker class, and whether it carries a
# robustness (PSI) block (which this launcher refuses).
CFG_INFO="$(uv run python - "$CONFIG_PATH" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
rr = cfg.get("reranker") or {}
exp_id = str(cfg.get("id") or "")
cls = str(rr.get("class") or "")
has_psi = "1" if cfg.get("robustness") else "0"
if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", exp_id):
    raise SystemExit(
        "ERROR: config id must start with a letter, digit, underscore, or hyphen "
        "and contain only letters, digits, dot, underscore, or hyphen"
    )
if "\n" in cls or "\r" in cls:
    raise SystemExit("ERROR: reranker class must not contain newlines")
print(exp_id)
print(cls)
print(has_psi)
PY
)"
EXP_ID="${CFG_INFO%%$'\n'*}"
REST="${CFG_INFO#*$'\n'}"
RERANKER_CLASS="${REST%%$'\n'*}"
HAS_PSI="${REST#*$'\n'}"

if [ -z "$EXP_ID" ]; then
  echo "ERROR: config $CONFIG_PATH has no 'id'." >&2
  exit 2
fi
if [ "$HAS_PSI" = "1" ]; then
  echo "ERROR: '$EXP_ID' has a 'robustness:' (PSI) block." >&2
  echo "Use the PSI launcher (base scoring + PSI, resume-safe):" >&2
  echo "  scripts/run_eval_robust_psi.sh $CONFIG" >&2
  exit 2
fi

# File names come only from the validated experiment id, never from the raw
# config argument. This keeps paths stable when CONFIG is an absolute or
# nested YAML path and prevents shell metacharacters from reaching scripts.
TS="$(date +%Y%m%d_%H%M%S)"
PYTHON_LOG="$LOG_DIR/${EXP_ID}_${TS}.python.log"
PIDFILE="$LOG_DIR/${EXP_ID}.pid"

# Closed-model API rerankers need provider auth. Validate it before entering
# the retry loop, so a missing key fails in a second rather than six retries.
if [ "$RERANKER_CLASS" = "ClosedModelGenBscReranker" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set for the closed-model reranker." >&2
    echo "Export OPENAI_API_KEY, and OPENAI_BASE_URL for a non-default endpoint." >&2
    exit 3
  fi
fi

# Refuse to double-launch the launcher itself.
if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: another robust eval launcher for $CONFIG is already running (pid $OLD_PID)." >&2
    echo "Tail its log under $LOG_DIR or stop it with: kill -TERM $OLD_PID" >&2
    exit 4
  fi
  rm -f "$PIDFILE"
fi

# Resolve the resume run dir. Priority:
#   1. RESUME_RUN_DIR env (explicit).
#   2. --resume flag -> most recent existing runs/<id>/<ts>/ for this id.
#   3. otherwise mint a fresh runs/<id>/<ts>/ (stable across this launch's
#      retries so resume works within the launch).
RUN_BASE="$REPO_ROOT/runs/$EXP_ID"
RESUME_DIR=""
if [ -n "${RESUME_RUN_DIR:-}" ]; then
  RESUME_DIR="$RESUME_RUN_DIR"
elif [ "$RESUME_FLAG" = "--resume" ] && [ -d "$RUN_BASE" ]; then
  # newest subdir by mtime
  RESUME_DIR="$(ls -dt "$RUN_BASE"/*/ 2>/dev/null | head -n1 || true)"
fi
if [ -n "$RESUME_DIR" ]; then
  RUN_DIR="${RESUME_DIR%/}"
  echo "[launcher] RESUMING into existing run dir: $RUN_DIR"
else
  RUN_DIR="$RUN_BASE/$TS"
  echo "[launcher] fresh run dir: $RUN_DIR"
fi

MAX_ATTEMPTS="${EVAL_LAUNCHER_MAX_ATTEMPTS:-6}"

# Build the inner worker as a separate file (avoids bash quoting pain when
# forwarding args through nohup/caffeinate).
INNER="$LOG_DIR/${EXP_ID}_${TS}.worker.sh"
cat >"$INNER" <<'EOF'
#!/usr/bin/env bash
# Inner worker: resume-into-RUN_DIR run_experiment, then run_eval. Retries
# transient failures with backoff; stops immediately on fatal patterns.
set -uo pipefail
CFG_PATH="$1"
RUN_DIR="$2"
EXP_ID="$3"
ATTEMPT=0
MAX="${EVAL_LAUNCHER_MAX_ATTEMPTS:-6}"
BACKOFF=20

# Non-transient: re-running will not help until credentials or config are fixed.
# Credential expiry/denial stops retries so a dead key is not spun against a
# paid endpoint; re-login and re-launch with --resume. The per-query
# "missing/failed score" RuntimeError stays transient so resume retries only
# the failed qids.
FATAL_RE='401 from |403 from |No API key|Unknown reranker class|reranker.model_id is required|score_max .* must be > score_min|Config not found|Could not resolve run dir|Prompt template .* is not supported'

while : ; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "[worker] === run_experiment attempt ${ATTEMPT}/${MAX} at $(date '+%Y-%m-%dT%H:%M:%S%z') (run_dir=${RUN_DIR}) ==="
  uv run python scripts/run_experiment.py -e "$CFG_PATH" --run-dir "$RUN_DIR"
  rc=$?
  echo "[worker] run_experiment exited rc=${rc}"
  if [ "${rc}" = "0" ]; then
    echo "[worker] reranking complete; evaluating ${RUN_DIR}"
    uv run python scripts/run_eval.py -e "$RUN_DIR"
    erc=$?
    echo "[worker] run_eval exited rc=${erc}"
    exit "${erc}"
  fi
  if [ -n "${PYTHON_LOG:-}" ] && [ -f "${PYTHON_LOG}" ] && grep -E "$FATAL_RE" "${PYTHON_LOG}" >/dev/null 2>&1; then
    echo "[worker] fatal/non-transient error detected; not retrying." >&2
    echo "[worker] if this was a credential expiry: re-login, then re-launch with --resume (no re-pay)." >&2
    exit "${rc}"
  fi
  if [ "${ATTEMPT}" -ge "${MAX}" ]; then
    echo "[worker] giving up after ${ATTEMPT} attempts." >&2
    exit "${rc}"
  fi
  echo "[worker] sleeping ${BACKOFF}s before retry (will resume completed qids) ..."
  sleep "${BACKOFF}"
  if [ "${BACKOFF}" -lt 600 ]; then
    BACKOFF=$((BACKOFF * 2))
  fi
done
EOF
chmod +x "$INNER"

echo "[launcher] CONFIG=$CONFIG (id=$EXP_ID class=$RERANKER_CLASS)"
echo "[launcher] run_dir=$RUN_DIR"
echo "[launcher] python_log=$PYTHON_LOG"
echo "[launcher] worker_script=$INNER"
echo "[launcher] pidfile=$PIDFILE"
echo "[launcher] max_attempts=$MAX_ATTEMPTS"

# Use a static positional-argument wrapper. Config-derived values are never
# interpolated into shell source or a `bash -c` command string.
LAUNCH_WRAPPER="$LOG_DIR/${EXP_ID}_${TS}.launch.sh"
cat >"$LAUNCH_WRAPPER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PIDFILE="$1"
PYTHON_LOG="$2"
MAX_ATTEMPTS="$3"
INNER="$4"
CONFIG_PATH="$5"
RUN_DIR="$6"
EXP_ID="$7"

exec >>"$PYTHON_LOG" 2>&1
echo "$$" >"$PIDFILE"
echo "[launcher] pid $$ guarding eval worker on $(date)"
export PYTHON_LOG
export EVAL_LAUNCHER_MAX_ATTEMPTS="$MAX_ATTEMPTS"
exec "$INNER" "$CONFIG_PATH" "$RUN_DIR" "$EXP_ID"
EOF
chmod +x "$LAUNCH_WRAPPER"

# Detach: nohup ignores SIGHUP on terminal close; caffeinate keeps the box
# awake where it exists; redirection fully decouples stdio.
LAUNCH_COMMAND=(
  bash "$LAUNCH_WRAPPER"
  "$PIDFILE" "$PYTHON_LOG" "$MAX_ATTEMPTS" "$INNER" "$CONFIG_PATH" "$RUN_DIR" "$EXP_ID"
)
if command -v caffeinate >/dev/null 2>&1; then
  LAUNCH_COMMAND=(caffeinate -dimsu "${LAUNCH_COMMAND[@]}")
else
  echo "[launcher] caffeinate not found (non-macOS); nohup alone keeps the run detached."
fi
nohup "${LAUNCH_COMMAND[@]}" </dev/null >>"$PYTHON_LOG" 2>&1 &
disown || true

sleep 1
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  echo "[launcher] launched. caffeinate+worker pid=$PID"
  echo "[launcher] watch:  tail -F $PYTHON_LOG"
  echo "[launcher] stop:   kill -TERM $PID"
  echo "[launcher] safe to close this shell now."
else
  echo "[launcher] WARNING: pidfile not written; check $PYTHON_LOG" >&2
fi
