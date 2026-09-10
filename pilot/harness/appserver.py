"""Minimal JSON-RPC over stdio client for `codex app-server`.

依据实读源码写成（codex-main @ 本地快照）：
  - 启动方式：`codex app-server`（默认 --listen stdio://）
  - 握手：request "initialize" {clientInfo, capabilities:{experimentalApi}} → notify "initialized"
  - 方法名见 codex-rs/app-server-protocol/src/protocol/common.rs
  - 参数字段为 camelCase（各 struct 上有 #[serde(rename_all = "camelCase")]）
  - `thread/fork` 支持 path=<rollout jsonl 路径>（experimental，需 experimentalApi=true）
  - `Thread.path` 为 [UNSTABLE] 字段，给出该 thread 的 rollout 文件路径

设计要点
--------
1. 读取必须在独立线程里做，并把三类消息分流，否则 notification 洪流会把 response 阻塞死：
     - 有 id + result/error        → 我们发出的 request 的响应
     - 有 id + method              → **服务端反向请求**（如审批），必须回复否则 turn 永久挂住
     - 无 id + method              → 通知
2. 审批：我们全程用 approvalPolicy="never" + sandbox="read-only"，理论上不会收到反向请求。
   但仍必须实现兜底回复，并把"收到过反向请求"作为一个 smoke 失败信号上报。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 环境钉死：与 config/env_pin.sh 保持一致
# ---------------------------------------------------------------------------
PINNED_ENV: dict[str, str] = {
    "LC_ALL": "C",
    "LANG": "C",
    "TZ": "UTC",
    "NO_COLOR": "1",
    "CLICOLOR": "0",
    "CLICOLOR_FORCE": "0",
    "TERM": "dumb",
    "COLUMNS": "200",
    "LINES": "50",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "RIPGREP_CONFIG_PATH": "/dev/null",
    "DO_NOT_TRACK": "1",
    "CI": "1",
}


def pinned_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(PINNED_ENV)
    if extra:
        env.update(extra)
    return env


class AppServerError(RuntimeError):
    """JSON-RPC error 或协议层失败。"""


class TurnTimeout(RuntimeError):
    """等待 turn/completed 超时。"""


class AppServer:
    """一个 codex app-server 子进程的生命周期与 RPC 封装。"""

    def __init__(
        self,
        codex_bin: str = "codex",
        codex_home: str | Path | None = None,
        log_path: str | Path | None = None,
        experimental_api: bool = True,
        approval_reply: str = "denied",
        startup_timeout: float = 30.0,
    ) -> None:
        self.codex_bin = codex_bin
        self.codex_home = Path(codex_home).expanduser().resolve() if codex_home else None
        self.experimental_api = experimental_api
        self.approval_reply = approval_reply

        # 分流后的三个通道
        self._responses: dict[int, dict] = {}
        self._resp_cv = threading.Condition()
        self.notifications: list[dict] = []          # 全量留档
        self._notif_q: "queue.Queue[dict]" = queue.Queue()
        self.server_requests: list[dict] = []        # 反向请求留档（应为空）

        self._id = 0
        self._id_lock = threading.Lock()
        self._closed = False

        env_extra = {}
        if self.codex_home:
            self.codex_home.mkdir(parents=True, exist_ok=True)
            env_extra["CODEX_HOME"] = str(self.codex_home)

        self._log = open(log_path, "a", encoding="utf-8") if log_path else None

        self.proc = subprocess.Popen(
            [self.codex_bin, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=pinned_env(env_extra),
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_tail: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        self.server_info = self.initialize(timeout=startup_timeout)

    # ------------------------------------------------------------------ 底层

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr_tail.append(line.rstrip("\n"))
            del self._stderr_tail[:-200]        # 只留最后 200 行
            if self._log:
                self._log.write(f"[stderr] {line}")

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            if self._log:
                self._log.write(f"<< {line}\n")
                self._log.flush()
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # app-server 不应该往 stdout 写非 JSON；留档以便排查
                self.notifications.append({"method": "__nonjson__", "params": {"raw": line}})
                continue

            has_id = "id" in msg and msg["id"] is not None
            if has_id and ("result" in msg or "error" in msg):
                with self._resp_cv:
                    self._responses[msg["id"]] = msg
                    self._resp_cv.notify_all()
            elif has_id and "method" in msg:
                self.server_requests.append(msg)
                self._answer_server_request(msg)
            else:
                self.notifications.append(msg)
                self._notif_q.put(msg)
        self._closed = True
        with self._resp_cv:
            self._resp_cv.notify_all()

    def _answer_server_request(self, msg: dict) -> None:
        """兜底回复服务端反向请求。

        我们的配置（approvalPolicy=never + sandbox=read-only）下不应触发。
        一旦触发说明配置没生效 —— smoke 必须把它当失败信号。
        """
        method = msg.get("method", "")
        if "requestApproval" in method or "requestUserInput" in method:
            result: dict[str, Any] = {"decision": self.approval_reply}
        else:
            result = {}
        self._send({"jsonrpc": "2.0", "id": msg["id"], "result": result})

    def _send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        raw = json.dumps(obj, ensure_ascii=False)
        if self._log:
            self._log.write(f">> {raw}\n")
            self._log.flush()
        self.proc.stdin.write(raw + "\n")
        self.proc.stdin.flush()

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def request(self, method: str, params: dict | None = None, timeout: float = 600.0) -> dict:
        rid = self._next_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        with self._resp_cv:
            while rid not in self._responses:
                if self._closed:
                    raise AppServerError(
                        f"app-server exited while waiting for {method}; "
                        f"stderr tail:\n" + "\n".join(self._stderr_tail[-20:])
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AppServerError(f"timeout ({timeout}s) waiting for response to {method}")
                self._resp_cv.wait(min(remaining, 1.0))
            msg = self._responses.pop(rid)
        if "error" in msg:
            raise AppServerError(f"{method} failed: {json.dumps(msg['error'], ensure_ascii=False)}")
        return msg.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ------------------------------------------------------------------ 握手

    def initialize(self, timeout: float = 30.0) -> dict:
        params: dict[str, Any] = {
            "clientInfo": {"name": "scae-pilot", "title": "SCAE Pilot", "version": "0.1.0"},
        }
        if self.experimental_api:
            params["capabilities"] = {"experimentalApi": True}
        res = self.request("initialize", params, timeout=timeout)
        self.notify("initialized")
        return res

    # -------------------------------------------------------------- 高层封装

    def thread_start(
        self,
        cwd: str | Path,
        *,
        model: str | None = None,
        model_provider: str | None = None,
        sandbox: str = "read-only",
        approval_policy: str = "never",
        base_instructions: str | None = None,
        config: dict | None = None,
        ephemeral: bool = False,
        timeout: float = 120.0,
    ) -> dict:
        params: dict[str, Any] = {
            "cwd": str(Path(cwd).resolve()),
            "sandbox": sandbox,
            "approvalPolicy": approval_policy,
        }
        if model:
            params["model"] = model
        if model_provider:
            params["modelProvider"] = model_provider
        if base_instructions:
            params["baseInstructions"] = base_instructions
        if config:
            params["config"] = config
        if ephemeral:
            params["ephemeral"] = True
        return self.request("thread/start", params, timeout=timeout)

    def thread_fork(
        self,
        *,
        thread_id: str | None = None,
        path: str | Path | None = None,
        before_turn_id: str | None = None,
        cwd: str | Path | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        sandbox: str = "read-only",
        approval_policy: str = "never",
        base_instructions: str | None = None,
        exclude_turns: bool = False,
        timeout: float = 120.0,
    ) -> dict:
        """按 rollout 文件路径 fork —— 这是 SCAE 前缀重放的核心原语。

        注意：ThreadForkParams.thread_id 是必填的 String（非 Option），
        因此按 path fork 时仍需传一个占位 thread_id；服务端在 path 非空时忽略它。
        """
        if thread_id is None and path is None:
            raise ValueError("thread_id 与 path 至少给一个")
        params: dict[str, Any] = {
            "threadId": thread_id or "",
            "sandbox": sandbox,
            "approvalPolicy": approval_policy,
        }
        if path is not None:
            params["path"] = str(Path(path).resolve())
        if before_turn_id:
            params["beforeTurnId"] = before_turn_id
        if cwd:
            params["cwd"] = str(Path(cwd).resolve())
        if model:
            params["model"] = model
        if model_provider:
            params["modelProvider"] = model_provider
        if base_instructions:
            params["baseInstructions"] = base_instructions
        if exclude_turns:
            params["excludeTurns"] = True
        return self.request("thread/fork", params, timeout=timeout)

    def turn_run(
        self,
        thread_id: str,
        text: str,
        *,
        timeout: float = 900.0,
        on_item: Callable[[dict], bool] | None = None,
    ) -> dict:
        """跑一个 turn 并阻塞到 turn/completed。

        on_item: 每收到一个 `item/completed` 通知就调用一次。
                 返回 True 表示"请中断本 turn"（用于 V5 只跑一步的场景）。

        返回 {"items": [...], "notifications": [...], "interrupted": bool, "turnId": str|None}
        """
        mark = len(self.notifications)
        params = {
            "threadId": thread_id,
            "clientUserMessageId": str(uuid.uuid4()),
            "input": [{"type": "text", "text": text, "textElements": []}],
        }
        # turn/start 可能立即返回 turnId，也可能阻塞到结束；两种都要能处理，
        # 因此在后台线程里发请求，主线程只盯通知流。
        result_box: dict[str, Any] = {}

        def _fire() -> None:
            try:
                result_box["result"] = self.request("turn/start", params, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc

        t = threading.Thread(target=_fire, daemon=True)
        t.start()

        items: list[dict] = []
        interrupted = False
        turn_id: str | None = None
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TurnTimeout(f"turn on {thread_id} exceeded {timeout}s")
            try:
                msg = self._notif_q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if "error" in result_box:
                    raise result_box["error"]
                continue

            method = msg.get("method", "")
            p = msg.get("params", {}) or {}
            if p.get("threadId") not in (None, thread_id):
                continue          # 别的 thread 的通知
            if method == "turn/started":
                turn_id = p.get("turnId") or p.get("turn", {}).get("id")
            elif method == "item/completed":
                items.append(p)
                if on_item is not None and on_item(p):
                    self.turn_interrupt(thread_id, turn_id)
                    interrupted = True
            elif method == "turn/completed":
                break

        t.join(timeout=30)
        if "error" in result_box and not interrupted:
            raise result_box["error"]
        return {
            "items": items,
            "notifications": self.notifications[mark:],
            "interrupted": interrupted,
            "turnId": turn_id,
            "result": result_box.get("result"),
        }

    def turn_interrupt(self, thread_id: str, turn_id: str | None = None) -> None:
        params: dict[str, Any] = {"threadId": thread_id}
        if turn_id:
            params["turnId"] = turn_id
        try:
            self.request("turn/interrupt", params, timeout=60)
        except AppServerError:
            pass          # 已结束的 turn 打断会报错，忽略

    def thread_read(self, thread_id: str, timeout: float = 60.0) -> dict:
        return self.request("thread/read", {"threadId": thread_id}, timeout=timeout)

    def compact_start(self, thread_id: str, timeout: float = 300.0) -> dict:
        """主动触发压缩。用于把命题 4 从敏感性分析升级为随机化实验。"""
        return self.request("thread/compact/start", {"threadId": thread_id}, timeout=timeout)

    # ------------------------------------------------------------------ 生命周期

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self._log:
            self._log.close()

    def __enter__(self) -> "AppServer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def thread_rollout_path(thread_obj: dict) -> str | None:
    """从 thread/start | thread/fork 的响应里取出 rollout 文件路径。

    `Thread.path` 是 [UNSTABLE] 字段，需 experimentalApi=true 才会出现。
    兜底：从 CODEX_HOME/sessions/YYYY/MM/DD/ 下按 thread id 扫描（见 smoke 的 V1）。
    """
    thread = thread_obj.get("thread", thread_obj)
    return thread.get("path")
