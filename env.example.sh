#!/usr/bin/env bash
# 复制为 env.sh 后按需修改，然后 `source env.sh`。
# env.sh 已在 .gitignore 中（别把 API key 提交上去）。

# ---------------------------------------------------------------- 必需（数据采集）
# 模型 API key。论文用 idealab 端点；换供应商需同时改下面的 SCAE_PROVIDER
# 并对齐 pilot/harness/runner.py 里 config.toml 的 base_url。
export IDEALAB_API_KEY="替换成你的 key"

# ---------------------------------------------------------------- 数据集位置
# 不设置时默认读 <repo>/dataset/instance_summary.csv 与 <repo>/dataset/projects
export SCAE_CSV="/path/to/instance_summary.csv"
export SCAE_PROJECTS="/path/to/swebench_verified_projects/projects"

# ---------------------------------------------------------------- 实验设置
# 默认值与论文 Appendix C.2 一致，改动会偏离论文口径
export SCAE_MODEL="gpt-5.5-0424-global"   # 被测 agent 的模型
export SCAE_PROVIDER="idealab"
export SCAE_WORKERS=6                     # 并行分片数（受 API 限流与磁盘 IO 约束）
export SCAE_TURN_TIMEOUT=900              # 单轮超时（秒）
export SCAE_CODEX_BIN="codex"             # codex CLI 路径（论文用 0.145.0）

# 产物目录。复现包不携带录制数据；若你已有一份，指过去即可跳过采集阶段。
# export SCAE_RUNS="/path/to/existing/runs"

# ---------------------------------------------------------------- 归因样本规模
export SCAE_M_JUDGE=3      # full_attr 档的重放次数（RQ5 judge 对比）
export SCAE_M_POWER=8      # v4prime 档的重放次数（RQ4 步级功效，论文正文口径）
export SCAE_N_POWER=38     # v4prime 档的 episode 数（38 × 5 = 190 个估值点）

# ---------------------------------------------------------------- judge 跨模型
export SCAE_JUDGE_GENERAL="qwen3.6-plus"
export SCAE_JUDGE_CODER="qwen3-coder-plus"
export SCAE_JUDGE_LIMIT=25            # episode 数
export SCAE_JUDGE_MAX_CALLS=10        # 每个窗口的最大调用数（限流）
export SCAE_JUDGE_WINDOW_MIN=60       # 限流窗口（分钟）

# ---------------------------------------------------------------- 调试
# 小样本冒烟：只跑前 N 个实例，用来验链路而不烧预算
# export SCAE_LIMIT=10
# export SCAE_YES=1                   # run_all.sh 跳过交互确认
# export SCAE_SMOKE_INSTANCE="astropy__astropy-12907"
