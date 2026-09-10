"""Minimal JSON-RPC over stdio client for `codex app-server`.

Written against the source as actually read (codex-main @ local snapshot):
  - Launch: `codex app-server` (defaults to --listen stdio://)
  - Handshake: request "initialize" {clientInfo, capabilities:{experimentalApi}} → notify "initialized"
  - Method names: see codex-rs/app-server-protocol/src/protocol/common.rs
  - Param fields are camelCase (each struct carries #[serde(rename_all = "camelCase")])
  - `thread/fork` supports path=<rollout jsonl path> (experimental, requires experimentalApi=true)
  - `Thread.path` is an [UNSTABLE] field giving that thread's rollout file path

Design notes
------------
1. Reading must happen on a separate thread, demultiplexing three message classes, otherwise the
   notification flood deadlocks responses:
     - has id + result/error   → response to a request we sent
     - has id + method         → **server-initiated request** (e.g. approval); must be answered, otherwise the turn hangs forever
     - no id + method          → notification
2. Approvals: we use approvalPolicy="never" + sandbox="read-only" throughout, so in theory no
   server-initiated request should arrive. A fallback reply must still be implemented, and
   "a server-initiated request was received" must be reported as a smoke failure signal.
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
# Environment pinning: kept consistent with config/env_pin.sh
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
    """JSON-RPC error or protocol-level failure."""


class TurnTimeout(RuntimeError):
    """Timed out waiting for turn/completed."""


class AppServer:
    """Lifecycle and RPC wrapper for one codex app-server subprocess."""

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

        # The three channels after demultiplexing
        self._responses: dict[int, dict] = {}
        self._resp_cv = threading.Condition()
        self.notifications: list[dict] = []          # full archive
        self._notif_q: "queue.Queue[dict]" = queue.Queue()
        self.server_requests: list[dict] = []        # archive of server-initiated requests (should be empty)

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

    # ------------------------------------------------------------------ low level

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr_tail.append(line.rstrip("\n"))
            del self._stderr_tail[:-200]        # keep only the last 200 lines
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
                # app-server should not write non-JSON to stdout; archive it for diagnosis
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
        """Fallback reply to a server-initiated request.

        Should not fire under our configuration (approvalPolicy=never + sandbox=read-only).
        If it does fire, the configuration did not take effect -- smoke must treat it as a failure signal.
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

    # ------------------------------------------------------------------ handshake

    def initialize(self, timeout: float = 30.0) -> dict:
        params: dict[str, Any] = {
            "clientInfo": {"name": "scae-pilot", "title": "SCAE Pilot", "version": "0.1.0"},
        }
        if self.experimental_api:
            params["capabilities"] = {"experimentalApi": True}
        res = self.request("initialize", params, timeout=timeout)
        self.notify("initialized")
        return res

    # -------------------------------------------------------------- high-level wrappers

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
        """Fork by rollout file path -- the core primitive of SCAE prefix replay.

        Note: ThreadForkParams.thread_id is a required String (not an Option), so forking by path
        still needs a placeholder thread_id; the server ignores it when path is non-empty.
        """
        if thread_id is None and path is None:
            raise ValueError("at least one of thread_id and path must be given")
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
        """Run one turn and block until turn/completed.

        on_item: called once for every `item/completed` notification received.
                 Returning True means "interrupt this turn" (used by the V5 single-step scenario).

        Returns {"items": [...], "notifications": [...], "interrupted": bool, "turnId": str|None}
        """
        mark = len(self.notifications)
        params = {
            "threadId": thread_id,
            "clientUserMessageId": str(uuid.uuid4()),
            "input": [{"type": "text", "text": text, "textElements": []}],
        }
        # turn/start may return turnId immediately, or block until the end; both must be handled,
        # so the request is sent on a background thread while the main thread only watches notifications.
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
                continue          # notification for a different thread
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
            pass          # interrupting an already-finished turn errors out; ignore

    def thread_read(self, thread_id: str, timeout: float = 60.0) -> dict:
        return self.request("thread/read", {"threadId": thread_id}, timeout=timeout)

    def compact_start(self, thread_id: str, timeout: float = 300.0) -> dict:
        """Actively trigger compaction. Used to upgrade Proposition 4 from a sensitivity analysis to a randomized experiment."""
        return self.request("thread/compact/start", {"threadId": thread_id}, timeout=timeout)

    # ------------------------------------------------------------------ lifecycle

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
    """Extract the rollout file path from a thread/start | thread/fork response.

    `Thread.path` is an [UNSTABLE] field and only appears with experimentalApi=true.
    Fallback: scan under CODEX_HOME/sessions/YYYY/MM/DD/ by thread id (see smoke's V1).
    """
    thread = thread_obj.get("thread", thread_obj)
    return thread.get("path")
