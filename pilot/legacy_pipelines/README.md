# 历史编排脚本（已被 ../../scripts/ 取代）

这两个脚本是原始实验实际使用的编排入口，保留在此仅作**溯源参考**。
它们引用了若干不随复现包分发的数据产物（`runs/sample40.txt`、`runs/upolicy_ids.txt`），
直接运行会失败。请用 `scripts/` 下的分阶段脚本。

| 历史脚本 | 对应的新入口 | 差异 |
|---|---|---|
| `run_pilot.sh` | `scripts/00_smoke.sh` + `scripts/01_locate.sh` | 原脚本跑 40 实例分层样本（需 `sample40.txt`）；新脚本跑全量，小样本用 `SCAE_LIMIT` |
| `run_remaining.sh` [1] | `scripts/01` + `scripts/05_upolicy.sh` + `scripts/02_attribution.sh` + `scripts/04` | 分片逻辑抽成 `setup/stage_attr_shards.py`；U_policy 抽成 `scripts/05` |

[1] `run_remaining.sh` 的第 3 步用 `multi[i::W]` 轮转分片并 symlink rollouts；
`setup/stage_attr_shards.py` 改用 `shard_by_repo`（与定位阶段一致）。
两者跑的实例集合相同（多文件筛选由 `run_attribution --multifile-only` 完成），
只影响负载均衡，不影响结果。
