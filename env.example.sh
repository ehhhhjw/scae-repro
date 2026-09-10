#!/usr/bin/env bash
# Copy to env.sh, adjust as needed, then `source env.sh`.
# env.sh is already in .gitignore (do not commit your API key).

# ---------------------------------------------------------------- required (data collection)
# Model API key. The paper uses the idealab endpoint; switching providers also requires changing
# SCAE_PROVIDER below and aligning the config.toml base_url in pilot/harness/runner.py.
export IDEALAB_API_KEY="replace with your key"

# ---------------------------------------------------------------- dataset location
# When unset, defaults to <repo>/dataset/instance_summary.csv and <repo>/dataset/projects
export SCAE_CSV="/path/to/instance_summary.csv"
export SCAE_PROJECTS="/path/to/swebench_verified_projects/projects"

# ---------------------------------------------------------------- experiment settings
# Defaults match the paper's Appendix C.2; changing them deviates from the paper's definitions
export SCAE_MODEL="gpt-5.5-0424-global"   # model of the agent under test
export SCAE_PROVIDER="idealab"
export SCAE_WORKERS=6                     # number of parallel shards (bounded by API rate limits and disk IO)
export SCAE_TURN_TIMEOUT=900              # per-turn timeout (seconds)
export SCAE_CODEX_BIN="codex"             # codex CLI path (the paper uses 0.145.0)

# Artifact directory. The reproduction package carries no recorded data; if you already have a
# copy, point here to skip the collection stages.
# export SCAE_RUNS="/path/to/existing/runs"

# ---------------------------------------------------------------- attribution sample sizes
export SCAE_M_JUDGE=3      # replays for the full_attr arm (RQ5 judge comparison)
export SCAE_M_POWER=8      # replays for the v4prime arm (RQ4 step-level power, the paper's main-text definition)
export SCAE_N_POWER=38     # episodes in the v4prime arm (38 x 5 = 190 valuation points)

# ---------------------------------------------------------------- judge cross-model
export SCAE_JUDGE_GENERAL="qwen3.6-plus"
export SCAE_JUDGE_CODER="qwen3-coder-plus"
export SCAE_JUDGE_LIMIT=25            # number of episodes
export SCAE_JUDGE_MAX_CALLS=10        # max calls per window (rate limit)
export SCAE_JUDGE_WINDOW_MIN=60       # rate-limit window (minutes)

# ---------------------------------------------------------------- debugging
# Small-sample smoke: run only the first N instances, to check the pipeline without burning budget
# export SCAE_LIMIT=10
# export SCAE_YES=1                   # run_all.sh skips the interactive confirmation
# export SCAE_SMOKE_INSTANCE="astropy__astropy-12907"
