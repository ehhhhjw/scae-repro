# Environment noise pinning (U_env pinning / common random numbers)
#
# Purpose: make Assumption A3 (replayable environment) hold empirically. Any output that varies
# with time / locale / terminal width becomes uncontrolled exogenous noise and breaks the
# comparability of prefix replays.
#
# Usage: source this file, or let PINNED_ENV in harness/appserver.py inject it equivalently.

# Sorting and text comparison must be locale-independent
export LC_ALL=C
export LANG=C

# Timestamps and relative times must be stable
export TZ=UTC

# Disable all colors / control characters
export NO_COLOR=1
export CLICOLOR=0
export CLICOLOR_FORCE=0
export TERM=dumb

# Terminal width changes the wrapping / truncation behavior of many tools
export COLUMNS=200
export LINES=50

# Hash randomization on the Python side changes dict/set iteration order
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# git must not read system/global config and must not prompt interactively
export GIT_CONFIG_NOSYSTEM=1
export GIT_TERMINAL_PROMPT=0
export GIT_PAGER=cat
export PAGER=cat

# ripgrep must not read the user config (otherwise --smart-case etc. change the hit set)
export RIPGREP_CONFIG_PATH=/dev/null

# Disable anonymous telemetry and update checks in various CLIs (they add network latency and nondeterministic output)
export DO_NOT_TRACK=1
export CI=1
