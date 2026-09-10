#!/usr/bin/env bash
# 各阶段脚本的公共前导：定位仓库根、钉死环境噪声、校验前置条件。
# 不要直接执行本文件，用 `source` 引入。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 环境噪声钉死（U_env pinning / 共同随机数）——论文假设 3「可重放环境」的前提。
# 与 pilot/harness/appserver.py 的 PINNED_ENV 等价，必须两边一致。
# shellcheck source=../pilot/config/env_pin.sh
source "$REPO_ROOT/pilot/config/env_pin.sh"

# 论文 Appendix C.2 的实验设置，可用环境变量覆盖
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
  [ -n "${IDEALAB_API_KEY:-}" ] || die "未设置 IDEALAB_API_KEY。数据采集阶段必需；纯离线分析请跑 scripts/04_analysis_offline.sh"
}

require_codex() {
  command -v "$CODEX_BIN" >/dev/null 2>&1 || die "找不到 codex CLI（$CODEX_BIN）。论文使用 codex 0.145.0 app-server。"
}

require_dataset() {
  python3 - <<'PY' || die "数据集未就绪。先跑 python3 setup/prepare_dataset.py，详见 REPRODUCE.md 第 1 步。"
import sys
from pilot.probes import task as T
if not T.CSV_PATH.exists():
    print(f"找不到 {T.CSV_PATH}"); sys.exit(1)
insts = T.load_instances()
n = sum(1 for i in insts if (T.PROJECTS_ROOT / i.instance_id / ".git").exists())
print(f"  数据集：{len(insts)} 实例，{n} 个有仓库快照（{T.PROJECTS_ROOT}）")
if n == 0:
    print("  没有任何仓库快照，定位实验无法进行"); sys.exit(1)
PY
}

banner() {
  info "仓库根     $REPO_ROOT"
  info "模型       $MODEL ($PROVIDER)"
  info "产物目录   $RUNS"
  info "并行度     $WORKERS   单轮超时 ${TURN_TIMEOUT}s"
}
