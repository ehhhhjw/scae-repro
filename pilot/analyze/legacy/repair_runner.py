"""E1 repair experiment (RepairSuccess@1) -- a falsifiable test of attribution utility. Four arms + precheck version.

Design lesson (v1 failed)
-------------------------
v1's oracle action always read sorted(gold)[0], but in 11/19 cases the file actually missed was a **different**
gold file -- wrong object fixed, all three arms zero, the experiment uninformative. The very first result exposed it; archived.
Lesson: any intervention experiment that needs API calls must first assert, at zero cost, that "the intervention points at the right object".

Four-arm design (neither control is dispensable)
------------------------------------------------
  none          no intervention, continue from tau_{1:t-1}              baseline
  top1_missed   inject "read the **missed** file" at the top-1 step     main effect
  rand_missed   inject the same action at a **random other step**       position control
  top1_irrelev  inject "read an **irrelevant** file" at the top-1 step  content control

Position control only → cannot rule out "injecting anything works"
Content control only → cannot rule out "injecting anywhere works"

Irrelevant files are defined on the AST dependency graph (>=3 hops from every gold file, or unreachable).

Acceptance criteria (preregistered in EXPERIMENT_DESIGN_v2.md §2.4)
-------------------------------------------------------------------
  main criterion     lower bound of the paired bootstrap CI on succ(top1_missed) - succ(none) > 0
  position-specific  succ(top1_missed) > succ(rand_missed)
  content-specific   succ(top1_missed) > succ(top1_irrelev)
  failure verdict    none and top1_missed indistinguishable -> attribution did not localize the causal step, report faithfully
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.graph.ast_index import build_index                            # noqa: E402
from pilot.harness import action_exec as AE                              # noqa: E402
from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer, thread_rollout_path       # noqa: E402
from pilot.harness.runner import write_codex_config                      # noqa: E402
from pilot.probes import task as T                                       # noqa: E402


# --------------------------------------------------------------------------
# Zero-cost precheck: drop every unusable case before running, so no API is wasted
# --------------------------------------------------------------------------
def precheck(rec: dict, inst, rollout_file: Path, idx) -> dict:
    """Returns {ok, reason, target_missed, irrelev_cmd, ...}."""
    out = {"instance_id": rec["instance_id"], "ok": False}
    missed = rec.get("missed_files") or []
    if not missed:
        out["reason"] = "no missed_files (successful cases do not enter E1)"
        return out
    t1 = rec.get("causal_top1_step")
    if not t1:
        out["reason"] = "no top1_step"
        return out
    if not rollout_file.exists():
        out["reason"] = "rollout missing"
        return out
    try:
        items = R.load(rollout_file)
        steps = R.steps(items)
    except (OSError, ValueError) as e:
        out["reason"] = f"rollout parse failed: {e}"
        return out
    if t1 > len(steps):
        out["reason"] = f"top1_step={t1} > n_steps={len(steps)}"
        return out

    target = missed[0]
    if not (inst.repo / target).is_file():
        out["reason"] = f"missed file not present in the workspace: {target}"
        return out

    # assert the oracle command really returns something non-empty
    T.reset_repo(inst)
    _, _, ocmd = AE.oracle_action(inst, target=target)
    o_out = AE.exec_readonly_action(inst.repo, ocmd)
    if len(o_out.strip()) < 20:
        out["reason"] = f"oracle output too short ({len(o_out)})"
        return out

    # irrelevant file (content control)
    irr = AE.irrelevant_action(inst, idx)
    if irr is None:
        out["reason"] = "no irrelevant file >=3 hops from gold on the AST graph"
        return out
    _, _, icmd = irr
    i_out = AE.exec_readonly_action(inst.repo, icmd)
    if len(i_out.strip()) < 20:
        out["reason"] = f"irrelevant file output too short ({len(i_out)})"
        return out

    out.update({"ok": True, "reason": "", "target_missed": target,
                "top1_step": t1, "n_steps": len(steps),
                "oracle_cmd": ocmd, "irrelev_cmd": icmd,
                "oracle_out_len": len(o_out), "irrelev_out_len": len(i_out)})
    return out


def continue_from(srv, fork_file: Path, inst, model, provider,
                  m: int, turn_timeout: float) -> list[float]:
    """Fork from the given history and run m continuations; returns the recall of each."""
    out = []
    for _ in range(m):
        T.reset_repo(inst)
        try:
            fk = srv.thread_fork(
                path=fork_file, cwd=inst.repo, model=model, model_provider=provider,
                sandbox="read-only", approval_policy="never",
                base_instructions=T.BASE_INSTRUCTIONS)
            srv.turn_run(fk["thread"]["id"], T.CONTINUE_PROMPT, timeout=turn_timeout)
            pth = thread_rollout_path(fk)
            ans = T.answer_from_rollout(R.load(pth)) if pth and Path(pth).exists() else None
            out.append(T.outcome(ans, inst.gold)["recall"])
        except Exception:  # noqa: BLE001
            continue
    return out


def run_case(srv, pc: dict, inst, rollout_file: Path, workdir: Path,
             model, provider, m: int, turn_timeout: float,
             rng: random.Random) -> dict:
    items = R.load(rollout_file)
    steps = R.steps(items)
    t1 = pc["top1_step"]
    target = pc["target_missed"]
    workdir.mkdir(parents=True, exist_ok=True)

    tool, o_args, o_cmd = AE.oracle_action(inst, target=target)
    T.reset_repo(inst)
    o_out = AE.exec_readonly_action(inst.repo, o_cmd)
    i_cmd = pc["irrelev_cmd"]
    i_args = {"cmd": i_cmd, "yield_time_ms": 10000, "max_output_tokens": 20000}
    T.reset_repo(inst)
    i_out = AE.exec_readonly_action(inst.repo, i_cmd)

    others = [t for t in range(1, len(steps) + 1) if t != t1]
    t_rand = rng.choice(others) if others else t1

    res = {k: pc[k] for k in ("instance_id", "target_missed", "top1_step", "n_steps")}
    res.update({"project": inst.project, "random_step": t_rand,
                "oracle_cmd": o_cmd[:120], "irrelev_cmd": i_cmd[:120]})

    arms = {
        "none": ("truncate", t1, None, None),
        "top1_missed": ("subst", t1, o_args, o_out),
        "rand_missed": ("subst", t_rand, o_args, o_out),
        "top1_irrelev": ("subst", t1, i_args, i_out),
    }
    for name, (kind, tt, args_d, out_txt) in arms.items():
        try:
            if kind == "truncate":
                hist = R.truncate_before_step(items, tt)
            else:
                hist = R.substitute_action(items, tt, tool, args_d, out_txt)
            R.assert_no_dangling_calls(hist)
            f = R.dump(hist, workdir / f"arm_{name}.jsonl")
            vals = continue_from(srv, f, inst, model, provider, m, turn_timeout)
            res[f"recall_{name}"] = vals
            if vals:
                res[f"mean_{name}"] = round(sum(vals) / len(vals), 4)
                res[f"succ_{name}"] = round(
                    sum(1 for x in vals if x >= 0.999) / len(vals), 4)
        except Exception as exc:  # noqa: BLE001
            res[f"err_{name}"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--out-dir", default="pilot/runs/repair2")
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--turn-timeout", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--precheck-only", action="store_true",
                    help="run only the zero-cost precheck, no API calls")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    insts = {i.instance_id: i for i in T.load_instances()}
    recs = [json.loads(l) for l in Path(args.attr).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if not r.get("error") and r.get("Y") == 0]

    # ---------------- Phase 1: zero-cost precheck ----------------
    print(f"=== precheck ({len(recs)} failed cases, zero API) ===")
    checks, usable = [], []
    for r in recs:
        inst = insts.get(r["instance_id"])
        if inst is None:
            continue
        rf = outdir.parent / "repair" / "rollouts" / f"{r['instance_id']}__k0.jsonl"
        if not rf.exists():
            cands = list(Path("pilot/runs").glob(
                f"**/rollouts/{r['instance_id']}__k0.jsonl"))
            rf = cands[0] if cands else rf
        try:
            idx = build_index(inst.repo)
        except (OSError, RecursionError) as e:
            checks.append({"instance_id": r["instance_id"], "ok": False,
                           "reason": f"AST index failed: {e}"})
            continue
        pc = precheck(r, inst, rf, idx)
        checks.append(pc)
        if pc["ok"]:
            usable.append((r, pc, rf))
        print(f"  {'OK ' if pc['ok'] else 'SKIP'} {r['instance_id'][:30]:<30} "
              f"{pc.get('reason','') or ('missed='+pc['target_missed'].split('/')[-1])}")
    (outdir / "precheck.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nusable {len(usable)}/{len(recs)}; precheck results → {outdir/'precheck.json'}")
    if args.precheck_only or not usable:
        return 0

    # ---------------- Phase 2: four-arm intervention ----------------
    codex_home = outdir / "codex_home"
    write_codex_config(codex_home, args.model)
    rng = random.Random(args.seed)
    srv = AppServer(codex_bin=args.codex_bin, codex_home=codex_home,
                    log_path=outdir / "appserver.log", experimental_api=True)
    out_path = outdir / "repair.jsonl"
    t0 = time.time()
    n = 0
    try:
        with open(out_path, "a", encoding="utf-8") as fh:
            for r, pc, rf in usable:
                inst = insts[r["instance_id"]]
                try:
                    res = run_case(srv, pc, inst, rf, outdir / "work" / r["instance_id"],
                                   args.model, args.model_provider, args.m,
                                   args.turn_timeout, rng)
                except Exception as exc:  # noqa: BLE001
                    res = {"instance_id": r["instance_id"],
                           "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                           "trace": traceback.format_exc()[-400:]}
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                print(f"[{n}/{len(usable)}] {r['instance_id'][:28]:<28} "
                      f"none={res.get('succ_none')} top1={res.get('succ_top1_missed')} "
                      f"rand={res.get('succ_rand_missed')} "
                      f"irrel={res.get('succ_top1_irrelev')}", flush=True)
    finally:
        srv.close()
    print(f"\ndone {n} records in {(time.time()-t0)/60:.1f}min → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
