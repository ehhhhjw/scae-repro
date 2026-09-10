#!/usr/bin/env bash
# Common preamble for the stage scripts: locate the repository root, pin down environment noise, check preconditions.
# Do not execute this file directly; pull it in with `source`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Environment noise pinning (U_env pinning / common random numbers) -- the premise of the paper's Assumption 3, "replayable environment".
# Equivalent to PINNED_ENV in pilot/harness/appserver.py; the two must stay consistent.
# shellcheck source=../pilot/config/env_pin.sh
source "$REPO_ROOT/pilot/config/env_pin.sh"

# The experiment settings from the paper's Appendix C.2, overridable via environment variables
MODEL="${SCAE_MODEL:-gpt-5.5-0424-global}"
PROVIDER="${SCAE_PROVIDER:-idealab}"
WORKERS="${SCAE_WORKERS:-6}"
TURN_TIMEOUT="${SCAE_TURN_TIMEOUT:-900}"
RUNS="${SCAE_RUNS:-$REPO_ROOT/pilot/runs}"
CODEX_BIN="${SCAE_CODEX_BIN:-codex}"

mkdir -p "$RUNS"

log()  { printf '\n\033[1m########## %s ##########\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

require_api_key() {
  [ -n "${IDEALAB_API_KEY:-}" ] || die "IDEALAB_API_KEY is not set. Required for the data collection stages; for pure offline analysis run scripts/04_analysis_offline.sh"
}

require_codex() {
  command -v "$CODEX_BIN" >/dev/null 2>&1 || die "codex CLI not found ($CODEX_BIN). The paper uses the codex 0.145.0 app-server."
}

require_dataset() {
  python3 - <<'PY' || die "Dataset not ready. Run python3 setup/prepare_dataset.py first; see step 1 of REPRODUCE.md."
import sys
from pilot.probes import task as T
if not T.CSV_PATH.exists():
    print(f"cannot find {T.CSV_PATH}"); sys.exit(1)
insts = T.load_instances()
n = sum(1 for i in insts if (T.PROJECTS_ROOT / i.instance_id / ".git").exists())
print(f"  dataset: {len(insts)} instances, {n} with a repository snapshot ({T.PROJECTS_ROOT})")
if n == 0:
    print("  no repository snapshots at all; the localization experiment cannot run"); sys.exit(1)
PY
}

banner() {
  info "repo root     $REPO_ROOT"
  info "model         $MODEL ($PROVIDER)"
  info "artifact dir  $RUNS"
  info "parallelism   $WORKERS   per-turn timeout ${TURN_TIMEOUT}s"
}
