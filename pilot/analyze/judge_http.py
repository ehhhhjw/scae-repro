"""Direct-HTTP judge -- bypasses codex, because the judge does not need an agent runtime at all.

Why it is needed
----------------
codex 0.145 only supports `wire_api = "responses"` (`chat` was removed), and on that endpoint the
qwen family disconnects mid-stream when streaming over `/responses`: measured, after 5 reconnects the
turn ends with `status=failed` and zero token usage, **and this happens after waiting out the full
60-minute rate window**, so it is not rate limiting. Whereas `/chat/completions` returns 200 for the same model.

The judge is pure text in, pure text out: no tools, no sandbox, no repository. Using codex was merely
the original convenience choice. Connecting directly to `/chat/completions` sidesteps the whole incompatibility.

**Prompts are verbatim identical to the original experiment**: reuse `pilot.attribution.judge`'s
`JUDGE_INSTRUCTIONS` / `render_trace` / `_parse_judge`, with instructions in the system slot and
prompt in the user slot -- a faithful translation of the original `base_instructions` + turn input.
If the prompts differed, cross-model comparison would be meaningless.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.attribution.judge import (                       # noqa: E402
    JUDGE_INSTRUCTIONS, OUTCOME_CLAUSE_FULL, OUTCOME_CLAUSE_PREFIX,
    render_trace, _parse_judge,
)
from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402

BASE_URL = os.environ.get(
    "IDEALAB_BASE_URL", "https://idealab.alibaba-inc.com/api/openai/v1")


def build_prompt(base_items: list[dict], inst: T.Instance, outcome_Y: int,
                 mode: str, prefix_upto: int | None) -> tuple[str, str, int]:
    """Return (system_instructions, user_prompt, n_steps), structurally identical to the original experiment."""
    steps = R.steps(base_items)
    n = len(steps)
    if mode == "full":
        instr = JUDGE_INSTRUCTIONS.format(outcome_clause=OUTCOME_CLAUSE_FULL)
        trace = render_trace(steps)
        outcome_line = (
            f"\nFinal outcome: localization {'SUCCEEDED' if outcome_Y else 'FAILED'}.\n")
    else:
        instr = JUDGE_INSTRUCTIONS.format(outcome_clause=OUTCOME_CLAUSE_PREFIX)
        trace = render_trace(steps, upto=prefix_upto)
        outcome_line = "\n(You are given only the first part of the trace; outcome unknown.)\n"
    prompt = (
        f"Issue:\n{inst.problem_statement[:2000]}\n\n"
        f"Agent trace:\n{trace}\n{outcome_line}\n"
        f"Which step was most responsible? Respond with the JSON object."
    )
    return instr, prompt, n


def chat_once(model: str, system: str, user: str, *, timeout: float = 180.0,
              max_tokens: int = 2048, temperature: float | None = None) -> dict:
    """A single /chat/completions call. Returns {ok, text, status, error, raw_head}."""
    key = os.environ.get("IDEALAB_API_KEY")
    if not key:
        return {"ok": False, "error": "IDEALAB_API_KEY not set"}
    body: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
            status = r.status
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = {"error": {"message": str(e)}}
        status = e.code
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    if isinstance(payload, dict) and payload.get("error"):
        return {"ok": False, "status": status,
                "error": json.dumps(payload["error"], ensure_ascii=False)[:300]}
    try:
        msg = payload["choices"][0]["message"]
    except Exception:
        return {"ok": False, "status": status,
                "error": "unexpected response shape",
                "raw_head": json.dumps(payload, ensure_ascii=False)[:300]}
    # qwen puts the chain of thought in reasoning_content and the answer in content; take both, content primary
    text = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    return {"ok": True, "status": status, "text": text,
            "reasoning_chars": len(reasoning),
            "finish_reason": payload["choices"][0].get("finish_reason")}


def judge_http(base_items: list[dict], inst: T.Instance, outcome_Y: int,
               model: str, mode: str = "full",
               prefix_upto: int | None = None,
               timeout: float = 180.0) -> dict:
    """Direct-connection variant of judge attribution; return structure matches `judge.judge_attribution`."""
    instr, prompt, n = build_prompt(base_items, inst, outcome_Y, mode, prefix_upto)
    if n == 0:
        return {"mode": mode, "n_steps": 0, "parsed": False, "raw": ""}
    res = chat_once(model, instr, prompt, timeout=timeout)
    if not res.get("ok"):
        return {"mode": mode, "n_steps": n, "parsed": False,
                "error": res.get("error"), "status": res.get("status")}
    text = res["text"]
    # if the answer was truncated by finish_reason=length the JSON may be incomplete; record it faithfully
    parsed = _parse_judge(text, n)
    if parsed is None:
        return {"mode": mode, "n_steps": n, "parsed": False,
                "raw": text[-400:], "finish_reason": res.get("finish_reason"),
                "reasoning_chars": res.get("reasoning_chars")}
    return {
        "mode": mode, "n_steps": n, "parsed": True,
        "top1_step": parsed["top1_step"],
        "top1_pos": round(parsed["top1_step"] / n, 4),
        "top3_steps": parsed["top3_steps"],
        "reason": parsed["reason"],
        "finish_reason": res.get("finish_reason"),
        "reasoning_chars": res.get("reasoning_chars"),
    }
