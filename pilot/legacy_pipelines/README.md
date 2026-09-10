# Legacy orchestration scripts (superseded by ../../scripts/)

These two scripts were the orchestration entry points actually used in the original experiments; they are kept
here purely for **provenance reference**. They reference several data artifacts that are not shipped with the
reproduction package (`runs/sample40.txt`, `runs/upolicy_ids.txt`), so running them directly will fail. Use the
per-stage scripts under `scripts/` instead.

| Legacy script | Corresponding new entry point | Difference |
|---|---|---|
| `run_pilot.sh` | `scripts/00_smoke.sh` + `scripts/01_locate.sh` | The old script ran a 40-instance stratified sample (needs `sample40.txt`); the new script runs the full set, with `SCAE_LIMIT` for a small sample |
| `run_remaining.sh` [1] | `scripts/01` + `scripts/05_upolicy.sh` + `scripts/02_attribution.sh` + `scripts/04` | The sharding logic was extracted into `setup/stage_attr_shards.py`; U_policy was extracted into `scripts/05` |

[1] Step 3 of `run_remaining.sh` shards round-robin with `multi[i::W]` and symlinks rollouts;
`setup/stage_attr_shards.py` switches to `shard_by_repo` (consistent with the localization stage).
Both run the same instance set (multi-file filtering is done by `run_attribution --multifile-only`),
so this only affects load balancing, not the results.
