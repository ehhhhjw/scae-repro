"""直连 HTTP 的 judge —— 绕过 codex，因为 judge 本不需要 agent 运行时。

为什么需要它
------------
codex 0.145 只支持 `wire_api = "responses"`（`chat` 已被移除），而该端点上的
qwen 系列在 `/responses` 上流式返回会中途断连：实测 5 次重连后 turn 以
`status=failed` 结束、token 用量为 0，**且这发生在等满 60 分钟速率窗口之后**，
所以不是速率限制。而 `/chat/completions` 对同一模型返回 200。

judge 是纯文本进、纯文本出：不用工具、不用沙箱、不碰仓库。用 codex 只是当初的
便利选择。直连 `/chat/completions` 就绕开了整个不兼容。

**提示词与原实验逐字一致**：复用 `pilot.attribution.judge` 的
`JUDGE_INSTRUCTIONS` / `render_trace` / `_parse_judge`，system 位放 instructions、
user 位放 prompt —— 这是原来 `base_instructions` + turn 输入的忠实翻译。
若提示词不一致，跨模型对比就无意义。
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
    """返回 (system_instructions, user_prompt, n_steps)，与原实验同构。"""
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
    """一次 /chat/completions 调用。返回 {ok, text, status, error, raw_head}。"""
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
    # qwen 把思维链放在 reasoning_content，答案在 content；两者都取，以 content 为主
    text = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    return {"ok": True, "status": status, "text": text,
            "reasoning_chars": len(reasoning),
            "finish_reason": payload["choices"][0].get("finish_reason")}


def judge_http(base_items: list[dict], inst: T.Instance, outcome_Y: int,
               model: str, mode: str = "full",
               prefix_upto: int | None = None,
               timeout: float = 180.0) -> dict:
    """直连版 judge 归因，返回结构与 `judge.judge_attribution` 一致。"""
    instr, prompt, n = build_prompt(base_items, inst, outcome_Y, mode, prefix_upto)
    if n == 0:
        return {"mode": mode, "n_steps": 0, "parsed": False, "raw": ""}
    res = chat_once(model, instr, prompt, timeout=timeout)
    if not res.get("ok"):
        return {"mode": mode, "n_steps": n, "parsed": False,
                "error": res.get("error"), "status": res.get("status")}
    text = res["text"]
    # 若答案被 finish_reason=length 截断，JSON 可能不完整；如实记录
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
