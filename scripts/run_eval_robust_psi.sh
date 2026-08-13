#!/usr/bin/env bash
# Resume-safe laptop launcher for PSI eval configs (identity-order base + PSI).
#
# Three commands share one run directory:
#   1) run_experiment.py: identity-order scoring (BM25 input order, 1 rerank/query)
#   2) run_eval.py: metrics.json at run root
#   3) run_psi.py: writes psi/psi_metrics.json and sc_metrics.json
#
# Base scoring uses the same resume-into-one-dir pattern as run_eval_robust.sh so
# credential expiry / laptop sleep never re-pays for completed base queries.
# PSI resumes per query and per permutation: completed artifacts under
# psi/per_query_results/ are skipped, and transient partial failures are retried.
#
# Usage:
#   export OPENAI_API_KEY=...
#   scripts/run_eval_robust_psi.sh example-passage-closed-model-genbsc-b20-psi
#   scripts/run_eval_robust_psi.sh example-passage-closed-model-genbsc-b20-psi --resume
#
# Detaching uses screen where available, then setsid, then nohup, and wraps the
# job in caffeinate on macOS to stop the machine sleeping mid-run. None of the
# three is required: the run works on a plain Linux box without any of them.
#
# Logs: runs/_psi_launcher_logs/<config>_<ts>.log

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

LOG_DIR="$REPO_ROOT/runs/_psi_launcher_logs"
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
if [ "$HAS_PSI" != "1" ]; then
  echo "ERROR: '$EXP_ID' has no 'robustness:' block: use scripts/run_eval_robust.sh instead." >&2
  exit 2
fi

# File names come only from the validated experiment id, never from the raw
# config argument. This keeps paths stable for absolute or nested YAML paths.
TS="$(date +%Y%m%d_%H%M%S)"
PYTHON_LOG="$LOG_DIR/${EXP_ID}_${TS}.log"
PIDFILE="$LOG_DIR/${EXP_ID}.pid"

if [ "$RERANKER_CLASS" = "ClosedModelGenBscReranker" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set for the closed-model reranker." >&2
    echo "Export OPENAI_API_KEY, and OPENAI_BASE_URL for a non-default endpoint." >&2
    exit 3
  fi
fi

if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: another PSI launcher for $CONFIG is already running (pid $OLD_PID)." >&2
    echo "Tail its log under $LOG_DIR or stop it with: kill -TERM $OLD_PID" >&2
    exit 4
  fi
  rm -f "$PIDFILE"
fi

RUN_BASE="$REPO_ROOT/runs/$EXP_ID"
RESUME_DIR=""
if [ -n "${RESUME_RUN_DIR:-}" ]; then
  RESUME_DIR="$RESUME_RUN_DIR"
elif [ "$RESUME_FLAG" = "--resume" ] && [ -d "$RUN_BASE" ]; then
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

INNER="$LOG_DIR/${EXP_ID}_${TS}.worker.sh"
cat >"$INNER" <<'EOF'
#!/usr/bin/env bash
# Stage 1: identity-order base (resume-able). Stage 2: PSI into same run_dir.
set -uo pipefail
CFG_PATH="$1"
RUN_DIR="$2"
EXP_ID="$3"
ATTEMPT=0
MAX="${EVAL_LAUNCHER_MAX_ATTEMPTS:-6}"
BACKOFF=20

FATAL_RE='401 from |403 from |No API key|Unknown reranker class|reranker.model_id is required|score_max .* must be > score_min|Config not found|Could not resolve run dir|Prompt template .* is not supported'

# --- Stage 1: base scoring + eval (skip if metrics.json already present) ---
if [ -f "$RUN_DIR/metrics.json" ]; then
  echo "[worker] stage 1 already complete (metrics.json exists); skipping base scoring"
else
  while : ; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[worker] === stage 1 run_experiment attempt ${ATTEMPT}/${MAX} at $(date '+%Y-%m-%dT%H:%M:%S%z') (run_dir=${RUN_DIR}) ==="
    uv run python scripts/run_experiment.py -e "$CFG_PATH" --run-dir "$RUN_DIR"
    rc=$?
    echo "[worker] run_experiment exited rc=${rc}"
    if [ "${rc}" = "0" ]; then
      echo "[worker] reranking complete; evaluating ${RUN_DIR}"
      uv run python scripts/run_eval.py -e "$RUN_DIR"
      erc=$?
      echo "[worker] run_eval exited rc=${erc}"
      if [ "${erc}" != "0" ]; then
        exit "${erc}"
      fi
      break
    fi
    if [ -n "${PYTHON_LOG:-}" ] && [ -f "${PYTHON_LOG}" ] && grep -E "$FATAL_RE" "${PYTHON_LOG}" >/dev/null 2>&1; then
      echo "[worker] fatal/non-transient error in stage 1; not retrying." >&2
      exit "${rc}"
    fi
    if [ "${ATTEMPT}" -ge "${MAX}" ]; then
      echo "[worker] stage 1 giving up after ${ATTEMPT} attempts." >&2
      exit "${rc}"
    fi
    echo "[worker] sleeping ${BACKOFF}s before stage 1 retry ..."
    sleep "${BACKOFF}"
    if [ "${BACKOFF}" -lt 600 ]; then
      BACKOFF=$((BACKOFF * 2))
    fi
  done
fi

# --- Stage 2: PSI (resume per qid/perm; retry transient partial failures) ---
if [ -f "$RUN_DIR/psi/psi_metrics.json" ]; then
  echo "[worker] stage 2 already complete (psi/psi_metrics.json exists); done"
  exit 0
fi

ATTEMPT=0
BACKOFF=20
while : ; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "[worker] === stage 2 run_psi attempt ${ATTEMPT}/${MAX} at $(date '+%Y-%m-%dT%H:%M:%S%z') (run_dir=${RUN_DIR}) ==="
  uv run python scripts/run_psi.py -e "$CFG_PATH" --run-dir "$RUN_DIR"
  prc=$?
  echo "[worker] run_psi exited rc=${prc}"
  if [ "${prc}" = "0" ]; then
    exit 0
  fi
  if [ -f "$RUN_DIR/psi/psi_metrics.json" ]; then
    echo "[worker] psi_metrics.json appeared despite rc=${prc}; treating as success"
    exit 0
  fi
  if [ -n "${PYTHON_LOG:-}" ] && [ -f "${PYTHON_LOG}" ] && grep -E "$FATAL_RE" "${PYTHON_LOG}" >/dev/null 2>&1; then
    echo "[worker] fatal/non-transient error in stage 2; not retrying." >&2
    exit "${prc}"
  fi
  if [ "${ATTEMPT}" -ge "${MAX}" ]; then
    echo "[worker] stage 2 giving up after ${ATTEMPT} attempts." >&2
    exit "${prc}"
  fi
  echo "[worker] sleeping ${BACKOFF}s before stage 2 retry (resume will skip completed qids/perms) ..."
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
echo "[launcher] max_attempts(stage1)=$MAX_ATTEMPTS"

# Prefer `screen -dmS` over bare nohup+disown+& so the process survives
# controlling-terminal and parent-shell teardown. Nohup only blocks SIGHUP
# delivery to this process, and disown only removes it from the invoking shell's
# job table; neither creates a new process group or session.
# `screen -dmS` calls setsid() under the hood, giving the job its own session
# fully detached from the controlling terminal.
#
# The pid-writing/env-export shim is its own script file rather than a nested
# `bash -c "... 'inner' ..."` string: multi-level nested quoting through screen
# produced a broken process tree (caffeinate as sibling/child instead of the
# wrapping parent). A real file sidesteps quoting and is easy to verify with
# `cat`.
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
echo "[launcher] pid $$ guarding PSI eval worker on $(date)"
export PYTHON_LOG
export EVAL_LAUNCHER_MAX_ATTEMPTS="$MAX_ATTEMPTS"
exec "$INNER" "$CONFIG_PATH" "$RUN_DIR" "$EXP_ID"
EOF
chmod +x "$LAUNCH_WRAPPER"

LAUNCH_COMMAND=(
  bash "$LAUNCH_WRAPPER"
  "$PIDFILE" "$PYTHON_LOG" "$MAX_ATTEMPTS" "$INNER" "$CONFIG_PATH" "$RUN_DIR" "$EXP_ID"
)
if command -v caffeinate >/dev/null 2>&1; then
  LAUNCH_COMMAND=(caffeinate -dimsu "${LAUNCH_COMMAND[@]}")
fi

# screen is preferred for the setsid() semantics described above. Where it is
# absent, setsid gives the same new-session guarantee directly, and nohup is the
# last resort: it blocks SIGHUP but leaves the job in this session.
SCREEN_NAME="psi-$(printf '%s' "$EXP_ID" | tr -c 'A-Za-z0-9_' '-')-$$"
if command -v screen >/dev/null 2>&1; then
  screen -dmS "$SCREEN_NAME" "${LAUNCH_COMMAND[@]}"
  echo "[launcher] screen session: $SCREEN_NAME (screen -r $SCREEN_NAME to attach; screen -S $SCREEN_NAME -X quit to kill)"
elif command -v setsid >/dev/null 2>&1; then
  setsid "${LAUNCH_COMMAND[@]}" </dev/null >/dev/null 2>&1 &
  echo "[launcher] screen not found; detached with setsid."
else
  nohup "${LAUNCH_COMMAND[@]}" </dev/null >/dev/null 2>&1 &
  disown || true
  echo "[launcher] no screen or setsid; detached with nohup, which survives SIGHUP but not session teardown."
fi

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
