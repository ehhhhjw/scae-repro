"""full-trace LLM-as-judge attribution -- the baseline attacked in C3 (reproducing mainstream practice).

The judge sees the **full trace (actions + observations) + the final outcome** and is asked to name the
top-3 root-cause steps. This is exactly the object we want to show has collider bias: it conditions on
the subsequent steps and the outcome (descendants of $A_t$), and theory predicts its attribution will
systematically skew towards later steps (§3.5).

A prefix-only judge (given only the prefix, without successors or outcome) is added as an isolating control:
if the full-trace judge's step position is significantly later than the prefix-only judge's, collider bias has been singled out.

The judge goes through the same idealab chat/responses endpoint, done in a one-shot turn (no tools needed).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer                            # noqa: E402
from pilot.probes import task as T                                       # noqa: E402

JUDGE_INSTRUCTIONS = """\
You are analyzing the execution trace of a code-localization agent. The agent was asked to \
identify which file(s) must be modified to resolve an issue. You are given the ordered list \
of the agent's steps (each step is a tool call and its observed output){outcome_clause}.

Your job: identify which single step was MOST responsible for the final localization outcome \
— the step that most determined whether the agent found the correct file(s). Also give your \
top-3 in order.

Answer ONLY with a JSON object on the last line, no prose after it:
{{"top1_step": <int>, "top3_steps": [<int>, <int>, <int>], "reason": "<one sentence>"}}

Step indices are 1-based, matching the numbering in the trace below.
"""

OUTCOME_CLAUSE_FULL = ", and the final outcome (whether localization succeeded)"
OUTCOME_CLAUSE_PREFIX = ""

_JSON_RE = re.compile(r"\{[^{}]*\"top1_step\"[^{}]*\}", re.DOTALL)


def render_trace(steps: list, max_out_chars: int = 400, upto: int | None = None) -> str:
    """Render the step sequence as judge-readable text. upto=render only the first N steps (for prefix-only)."""
    lines = []
    seq = steps[:upto] if upto else steps
    for s in seq:
        out = (s.output_text or "").replace("\n", " ")
        if len(out) > max_out_chars:
            out = out[:max_out_chars] + "…"
        args = s.args_raw.replace("\n", " ")
        if len(args) > 200:
            args = args[:200] + "…"
        lines.append(f"Step {s.idx}: tool={s.tool} args={args}\n  -> output: {out}")
    return "\n".join(lines)


def _parse_judge(text: str, n_steps: int) -> dict | None:
    m = _JSON_RE.findall(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m[-1])
    except json.JSONDecodeError:
        return None
    t1 = obj.get("top1_step")
    if not isinstance(t1, int) or not (1 <= t1 <= n_steps):
        return None
    top3 = [x for x in obj.get("top3_steps", []) if isinstance(x, int) and 1 <= x <= n_steps]
    return {"top1_step": t1, "top3_steps": top3 or [t1], "reason": obj.get("reason", "")}


def judge_attribution(
    srv: AppServer,
    base_items: list[dict],
    inst: T.Instance,
    outcome_Y: int,
    model: str | None,
    provider: str | None,
    mode: str = "full",          # "full" | "prefix"
    prefix_upto: int | None = None,
    turn_timeout: float = 300.0,
) -> dict | None:
    """Have the judge do root-cause attribution on one trace. With mode=prefix only the first prefix_upto steps are given, without the outcome."""
    steps = R.steps(base_items)
    n = len(steps)
    if n == 0:
        return None

    if mode == "full":
        instr = JUDGE_INSTRUCTIONS.format(outcome_clause=OUTCOME_CLAUSE_FULL)
        trace = render_trace(steps)
        outcome_line = (
            f"\nFinal outcome: localization {'SUCCEEDED' if outcome_Y else 'FAILED'}.\n"
        )
    else:
        instr = JUDGE_INSTRUCTIONS.format(outcome_clause=OUTCOME_CLAUSE_PREFIX)
        trace = render_trace(steps, upto=prefix_upto)
        outcome_line = "\n(You are given only the first part of the trace; outcome unknown.)\n"

    prompt = (
        f"Issue:\n{inst.problem_statement[:2000]}\n\n"
        f"Agent trace:\n{trace}\n{outcome_line}\n"
        f"Which step was most responsible? Respond with the JSON object."
    )

    # the judge needs no tools/repository, done with a one-shot thread
    started = srv.thread_start(
        cwd=inst.repo, model=model, model_provider=provider,
        sandbox="read-only", approval_policy="never",
        base_instructions=instr, timeout=120,
    )
    tid = started["thread"]["id"]
    run = srv.turn_run(tid, prompt, timeout=turn_timeout)
    text = T.final_message_text(run["items"])
    parsed = _parse_judge(text, n)
    if parsed is None:
        return {"mode": mode, "n_steps": n, "parsed": False, "raw": text[-300:]}
    return {
        "mode": mode,
        "n_steps": n,
        "parsed": True,
        "top1_step": parsed["top1_step"],
        "top1_pos": round(parsed["top1_step"] / n, 4),
        "top3_steps": parsed["top3_steps"],
        "reason": parsed["reason"],
    }
