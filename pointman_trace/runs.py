"""This node's agent runs, read from their logs as they grow.

The node keeps each run's Claude Code stream (stream-json, one event a line) as run-N.json: on a Mac in
~/Library/Logs/message-board/runs, elsewhere in ~/.local/state/message-board/runs. While a run is live, its
run-N.meta.json is beside it (who, the thread that woke it, the session's cost before it, the run-keep
process), and so is its first prompt, run-N.prompt. run-N.in is a pipe and is never opened.

A run is one or more turns, each a wake: from the run's start (or its last turn's `result`) to its own
`result`. A turn's spans, timed in epoch ms:

  run    the turn itself, under its agent
  llm    a model call (its messages share a request_id), under the turn, or under a sub-agent's task
  tool   a tool call, from its tool_use to its tool_result; `task` for a sub-agent (Task, Agent), whose own
         calls nest under it

A model call starts when its context hands back to the model (the turn's start, or the last of its tool
results), and runs until its next message.

**Which thread a call is for:** the one the turn is working in when it starts. A turn starts in the thread that
woke it; a call that names another thread (`thread_id`) moves it there, along with the model call that made it.
A resident session's later wakes start in none, and their first calls wait for the first thread they name.
So a session that works on several threads in one turn shows each thread's part in that thread's trace.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

CUT = 300            # characters of a span's input or output: the node's limit
DETAIL = 80          # characters of a span's detail
FRESH = 10.0         # seconds a log with no meta file beside it still counts as starting
NAME = re.compile(r"^run-(\d+)\.json$")
HANDLE = re.compile(r"board @([a-z0-9][a-z0-9_-]{0,63})")
TITLE = re.compile(r"“([^”\n]{1,120})” \(thread (\d+)\)")
THREAD = re.compile(r"\(thread (\d+)\)")
POST = re.compile(r"^\[\d+\] (.+?)(?: ›|:)", re.M)
SPAWN = re.compile(r"^/spawn ([a-z0-9][a-z0-9-]*)", re.M)
TASKS = ("Task", "Agent")
SECRET = [
    re.compile(r"(?i)\b(bearer|basic)(\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b([a-z0-9_-]*(?:token|secret|password|passwd|api[_-]?key|_key|-key)|authorization|cookie)\b"
               r"([\"']?\s*[:=]\s*[\"']?)(?!bearer\b|basic\b|\[hidden\])[^\s,\"'}]+"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abpr]-[A-Za-z0-9-]{10,})"),
]
LONG = re.compile(r"\b[A-Za-z0-9+_-]{40,}={0,2}")    # a key or token: long, and full of digits (a path's words aren't)
SERVICES = {"Bash": "shell", "Read": "files", "Edit": "files", "Write": "files", "Glob": "files", "Grep": "files",
            "NotebookEdit": "files", "WebFetch": "web", "WebSearch": "web", "TodoWrite": "notes",
            "ToolSearch": "tools", "Skill": "skills", "Task": "agents", "Agent": "agents"}


def run_logs(home: Path | None = None) -> Path:
    """Where this node keeps its runs' logs (the node's own `run_logs`)."""
    home = home or Path.home()
    mac = home / "Library/Logs/message-board"
    return mac / "runs" if mac.is_dir() else home / ".local/state/message-board/runs"


def clean(text, n: int = CUT) -> str | None:
    """Text for a span: terminal colours out, anything that looks like a credential hidden, on one line, at
    most n characters."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    for p in SECRET:
        text = p.sub(lambda m: m.group(1) + m.group(2) + "[hidden]" if m.lastindex and m.lastindex >= 2
                     else "[hidden]", text)
    text = LONG.sub(lambda m: "[hidden]" if sum(c.isdigit() for c in m.group(0)) >= 6 else m.group(0), text)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def ms_of(stamp) -> int | None:
    if not isinstance(stamp, str):
        return None
    try:
        return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def service(name: str) -> str:
    if name.startswith(("mcp__pointman__", "mcp__message-board")):
        return "Pointman"
    if name.startswith("mcp__"):
        return name.split("__")[1]
    return SERVICES.get(name, name)


def friendly(name: str, inp) -> tuple[str, str]:
    """A tool call's row name and its dim detail."""
    inp = inp if isinstance(inp, dict) else {}
    if name.startswith("mcp__"):
        what = inp.get("thread_id") or inp.get("message_id") or inp.get("ref") or inp.get("query") or ""
        return name.split("__")[-1], str(what)
    if name == "Bash":
        return "Bash", inp.get("description") or inp.get("command") or ""
    if name in ("Read", "Edit", "Write"):
        return name, os.path.basename(str(inp.get("file_path", "")))
    if name in TASKS:
        return inp.get("description") or "Sub-agent", inp.get("subagent_type") or ""
    return name, next((v for v in inp.values() if isinstance(v, str)), "")


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text") or "")
            elif b.get("type") == "tool_result":
                out.append(text_of(b.get("content")))
    return " ".join(out)


def thread_of(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Turn:
    """One wake of a run: its run span, and its model and tool calls in the order they began (parents first)."""

    def __init__(self, run: Run, k: int, start: int):
        self.run, self.k = run, k
        self.id = f"r:{run.rid}" if k == 0 else f"r:{run.rid}.{k}"
        self.start = start
        self.last = start                     # the latest event seen
        self.end: int | None = None           # its result, or its run's end
        self.error = False
        self.before: float | None = None      # the session's cost before it, and after it
        self.total: float | None = None
        self.first: int | None = None         # the thread it started in, and the one it's in at its end
        self.final: int | None = None
        self.untagged: list[dict] = []        # calls made before it named a thread
        self.detail = ""
        self.prompt: str | None = None
        self.spans: list[dict] = []
        self.texts: dict[str, list] = {}      # span id → [input, output], cleaned
        self.models: Counter = Counter()

    @property
    def handle(self) -> str:
        return self.run.handle or f"run-{self.run.rid}"

    @property
    def running(self) -> bool:
        return self.end is None

    @property
    def threads(self) -> set[int]:
        return {s["_thread"] for s in self.spans if s.get("_thread")} | ({self.first} if self.first else set())

    @property
    def now(self) -> int | None:
        """The thread it's working in now, or was at its end."""
        return self.run.cur if self.end is None else self.final

    def mine(self, thread: int | None) -> list[dict]:
        return self.spans if thread is None else [s for s in self.spans if s.get("_thread") == thread]

    def running_for(self, thread: int | None) -> bool:
        return self.end is None and (thread is None or self.now == thread)

    def last_for(self, thread: int | None) -> int:
        if thread is None or self.now == thread:
            return self.end or self.last
        return max((s["end_ms"] or s["start_ms"] for s in self.mine(thread)), default=self.start)

    @property
    def cost(self) -> float | None:
        if self.total is None or self.before is None or self.total < self.before:
            return None
        return self.total - self.before

    def span(self, thread: int | None = None) -> dict:
        """Its run span; for a thread, only its part there: from its first call in it to its last, and its cost
        shared out by model calls."""
        own = self.mine(thread)
        start = self.start if thread is None or self.first == thread else min(
            (s["start_ms"] for s in own), default=self.start)
        if self.running_for(thread):
            end = None
        elif thread is None or self.now == thread:
            end = self.end
        else:
            end = self.last_for(thread)
        cost = self.cost
        if cost is not None and thread is not None:
            calls = sum(s["kind"] == "llm" for s in self.spans)
            cost = cost * sum(s["kind"] == "llm" for s in own) / calls if calls else cost
        models = Counter(s.get("model") for s in own if s["kind"] == "llm" and s.get("model")) or self.models
        return {"id": self.id, "parent": f"a:{self.handle}", "kind": "run",
                "name": f"Run {self.run.rid}" + (f" · wake {self.k + 1}" if self.k else ""),
                "detail": self.detail, "model": models.most_common(1)[0][0] if models else None,
                "start_ms": start, "end_ms": end, "cost_usd": round(cost, 4) if cost is not None else None,
                "state": "running" if end is None else "error" if self.error and self.now == thread else "ok"}


class Run:
    """One run's log, read as it grows."""

    def __init__(self, rid: int, path: Path):
        self.rid, self.path = rid, path
        self.offset, self.rest = 0, b""
        self.handle: str | None = None
        self.session: str | None = None
        self.startup = False                 # a new session, so nothing was spent before it
        self.client = "claude"
        self.thread: int | None = None       # the thread that woke it (its meta's)
        self.cost_before: float | None = None
        self.keeper: int | None = None
        self.started: int | None = None
        self.turns: list[Turn] = []
        self.open: Turn | None = None
        self.ctx: dict = {}                  # None, or a task's tool_use id → its model calls' state
        self.uses: dict[str, tuple] = {}     # an open tool_use's id → (turn, span, context)
        self.n = 0                           # model calls so far, for their ids
        self.cur: int | None = None          # the thread its open turn is working in
        self.ended = False
        self.spawns: list[str] = []          # the agents it /spawn'ed
        self.titles: dict[int, str] = {}
        self.version = 0

    # ------------------------------------------------------------------ reading

    def look(self, now: float | None = None) -> bool:
        """Read what's new: its meta while it's there, then its log's new lines. True if anything changed."""
        now = time.time() if now is None else now
        try:
            st = self.path.stat()
        except OSError:
            return False
        if self.ended and st.st_size <= self.offset:
            return False                      # it's over and its log hasn't grown: nothing to look at
        before = self.version
        meta = self._meta()
        if self.started is None:
            self.started = self._start_ms(st)
        if st.st_size > self.offset:
            self._read(st.st_size)
        if meta is not None:
            live = alive(self.keeper)
        else:
            live = self.keeper is None and now - st.st_mtime < FRESH and not self.ended
        if live and not self.turns and self.started is not None:
            self._turn(self.started)          # a run just begun: its first model call is running
        if not live and not self.ended:
            self._finish(int(st.st_mtime * 1000))
        return self.version != before

    def _meta(self) -> dict | None:
        """Its meta's who and why, while it's live. Only these fields: the rest (how it was started) isn't ours."""
        try:
            m = json.loads(self.path.with_name(f"run-{self.rid}.meta.json").read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(m, dict):
            return None
        self.handle = self.handle or (m.get("handle") if isinstance(m.get("handle"), str) else None)
        self.client = m.get("client") if isinstance(m.get("client"), str) else self.client
        self.thread = thread_of(m.get("thread")) or self.thread
        if isinstance(m.get("cost_before"), (int, float)) and self.cost_before is None:
            self.cost_before = float(m["cost_before"])
        if isinstance(m.get("keeper"), int):
            self.keeper = m["keeper"]
        return m

    def _start_ms(self, st) -> int | None:
        """When it began: its prompt's file, written as it starts, or the log's own birth (on a Mac)."""
        try:
            return int(self.path.with_name(f"run-{self.rid}.prompt").stat().st_mtime * 1000)
        except OSError:
            pass
        born = getattr(st, "st_birthtime", None)
        return int(born * 1000) if born else None

    def _prompt(self) -> str | None:
        try:
            with open(self.path.with_name(f"run-{self.rid}.prompt"), "rb") as f:
                first = f.readline(1 << 20)
            msg = json.loads(first).get("message") or {}
        except (OSError, ValueError, AttributeError):
            return None
        return text_of(msg.get("content")) or None

    def _read(self, size: int) -> None:
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset += len(data)
        lines = (self.rest + data).split(b"\n")
        self.rest = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict):
                self.event(e)

    # ------------------------------------------------------------------ events

    def event(self, e: dict) -> None:
        kind = e.get("type")
        if kind == "system":
            if e.get("subtype") == "session_title_changed" and not self.handle:
                m = HANDLE.search(str(e.get("title") or ""))
                self.handle = m.group(1) if m else None
            if e.get("session_id") and not self.session:
                self.session = e["session_id"]
            if e.get("hook_name") == "SessionStart:startup":
                self.startup = True
            return
        t = ms_of(e.get("timestamp"))
        if kind == "assistant" and t is not None:
            self._assistant(e, t)
        elif kind == "user" and t is not None:
            self._user(e, t)
        elif kind == "result":
            self._result(e)
        else:
            return
        self.version += 1

    def _turn(self, t: int) -> Turn:
        if self.open is not None:
            return self.open
        k = len(self.turns)
        start = self.started if k == 0 and self.started is not None and self.started <= t else t
        turn = Turn(self, k, start)
        self.cur = None
        if k == 0:
            prompt = self._prompt()
            woke = self.thread
            if prompt:
                body = prompt.split("\n\n", 1)[-1]
                turn.prompt = clean(body)
                self._titles(prompt)
                m = POST.search(prompt)
                turn.detail = clean(m.group(1), DETAIL) if m else ""
                woke = woke or next((int(n) for n in THREAD.findall(prompt)), None)
            self.cur = turn.first = woke
        self.turns.append(turn)
        self.open = turn
        self.ctx = {c: cx for c, cx in self.ctx.items() if cx["open"]}    # a background task's calls go on
        self.ctx[None] = {"llm": None, "req": None, "since": start, "last": start, "open": set()}
        self.ctx[None]["llm"] = self._llm(turn, None, start)
        self.version += 1
        return turn

    def _cx(self, turn: Turn, c) -> dict:
        cx = self.ctx.get(c)
        if cx is None:
            since = turn.last if c is None else (self.uses[c][1]["start_ms"] if c in self.uses else turn.last)
            cx = self.ctx[c] = {"llm": None, "req": None, "since": since, "last": since, "open": set()}
        return cx

    def _llm(self, turn: Turn, c, start: int) -> dict:
        self.n += 1
        span = {"id": f"l:{self.rid}.{self.n}", "parent": turn.id if c is None else f"t:{c}", "kind": "llm",
                "name": "Model call", "detail": "", "model": None, "start_ms": start, "end_ms": None,
                "tokens_in": 0, "tokens_out": 0, "tokens_cached": 0, "state": "running"}
        self._tag(turn, span)
        turn.texts[span["id"]] = [None, None]
        return span

    def _tag(self, turn: Turn, span: dict) -> None:
        span["_turn"], span["_thread"] = turn, self.cur
        turn.spans.append(span)
        if self.cur is None:
            turn.untagged.append(span)

    def _move(self, turn: Turn, thread: int, llm: dict | None) -> None:
        """It's working in another thread now: so is the model call that named it, and what waited for one."""
        self.cur = thread
        if turn.first is None:
            turn.first = thread
        for s in turn.untagged:
            s["_thread"] = thread
        turn.untagged.clear()
        if llm is not None:
            llm["_thread"] = thread

    def _close(self, cx: dict, state: str = "ok") -> None:
        llm = cx.get("llm")
        if llm is not None and llm["end_ms"] is None:
            llm["end_ms"] = max(llm["start_ms"], cx["last"]) if cx["req"] is not None else llm["start_ms"]
            llm["state"] = state
        cx["llm"], cx["req"] = None, None

    def _assistant(self, e: dict, t: int) -> None:
        turn = self._turn(t)
        turn.last = max(turn.last, t)
        c = e.get("parent_tool_use_id")
        cx = self._cx(turn, c)
        msg = e.get("message") or {}
        key = e.get("request_id") or msg.get("id") or e.get("uuid")
        if cx["llm"] is not None and cx["req"] is not None and cx["req"] != key:
            self._close(cx)
        if cx["llm"] is None:
            cx["llm"] = self._llm(turn, c, cx["since"])
        llm = cx["llm"]
        if cx["req"] != key:
            cx["req"] = key
            if msg.get("model"):
                llm["model"] = msg["model"]
                llm["_turn"].models[msg["model"]] += 1
        cx["last"] = max(cx["last"], t)
        u = msg.get("usage")
        if isinstance(u, dict):
            # new input (and what it wrote to the cache) apart from what it read back from the cache, which every
            # call reads again and would swamp the count
            llm["tokens_in"] = int((u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0))
            llm["tokens_out"] = int(u.get("output_tokens") or 0)
            llm["tokens_cached"] = int(u.get("cache_read_input_tokens") or 0)
        said = []
        for b in msg.get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text"):
                said.append(b["text"])
            elif b.get("type") == "tool_use" and isinstance(b.get("id"), str):
                n = thread_of((b.get("input") or {}).get("thread_id")) if isinstance(b.get("input"), dict) else None
                if n is not None and n != self.cur and c is None:
                    self._move(llm["_turn"], n, llm)
                self._tool(llm["_turn"], c, cx, b, t)
                name = friendly(str(b.get("name") or "?"), b.get("input"))[0]
                tools = [x for x in llm["detail"].split(", ") if x]
                if name not in tools:
                    llm["detail"] = clean(", ".join([*tools, name]), DETAIL)
        if said:
            texts = llm["_turn"].texts[llm["id"]]
            texts[1] = clean(" ".join(filter(None, [texts[1], *said])))

    def _tool(self, turn: Turn, c, cx: dict, b: dict, t: int) -> None:
        raw = str(b.get("name") or "?")
        inp = b.get("input")
        name, detail = friendly(raw, inp)
        span = {"id": f"t:{b['id']}", "parent": turn.id if c is None else f"t:{c}",
                "kind": "task" if raw in TASKS else "tool", "name": clean(name, DETAIL) or "?",
                "detail": clean(detail, DETAIL) or "", "service": service(raw), "start_ms": t, "end_ms": None,
                "state": "running"}
        self._tag(turn, span)
        turn.texts[span["id"]] = [clean(inp), None]
        self.uses[b["id"]] = (turn, span, c)
        cx["open"].add(b["id"])
        if isinstance(inp, dict):
            if raw.endswith("post_message") and isinstance(inp.get("text"), str):
                self.spawns.extend(SPAWN.findall(inp["text"]))

    def _user(self, e: dict, t: int) -> None:
        turn = self._turn(t)
        turn.last = max(turn.last, t)
        content = (e.get("message") or {}).get("content")
        results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"] \
            if isinstance(content, list) else []
        if not results:                       # a prompt: the model's next call starts now
            cx = self._cx(turn, e.get("parent_tool_use_id"))
            cx["since"] = cx["last"] = t
            return
        for b in results:
            found = self.uses.pop(str(b.get("tool_use_id")), None)
            if found is None:
                continue
            owner, span, c = found
            span["end_ms"] = t
            span["state"] = "error" if b.get("is_error") else "ok"
            raw = text_of(b.get("content"))
            owner.texts[span["id"]][1] = clean(raw)
            self._titles(raw)
            if span["kind"] == "task":        # its sub-agent is done
                sub = self.ctx.pop(span["id"][2:], None)
                if sub is not None:
                    self._close(sub)
            cx = self.ctx.get(c)
            if cx is None:
                continue
            self._close(cx)
            cx["open"].discard(span["id"][2:])
            if not cx["open"]:                # the model always answers its tool results: its next call starts now
                cx["since"] = cx["last"] = t
                cx["llm"] = self._llm(turn, c, t)

    def _result(self, e: dict) -> None:
        turn = self.open
        if turn is None:
            return
        for cx in self.ctx.values():
            self._close(cx)
        self.ctx = {c: cx for c, cx in self.ctx.items() if cx["open"]}
        total = e.get("total_cost_usd")
        if isinstance(total, (int, float)):
            turn.total = float(total)
            turn.before = self.turns[turn.k - 1].total if turn.k else self.cost_before
        turn.error = bool(e.get("is_error")) or e.get("subtype") not in (None, "success")
        took = e.get("duration_ms")
        turn.end = max(turn.last, turn.start + int(took)) if isinstance(took, (int, float)) else turn.last
        turn.final = self.cur
        self.open = None

    def _finish(self, end: int) -> None:
        """It has stopped: what's still open was cut off."""
        self.ended = True
        if self.open is not None:
            self.open.end = max(self.open.last, min(end, self.open.last + 60_000))
            self.open.error = True
            self.open.final = self.cur
        for cx in self.ctx.values():
            self._close(cx, "error")
        for owner, span, _ in self.uses.values():
            span["end_ms"] = max(span["start_ms"], owner.end or owner.last)
            span["state"] = "ok" if owner.end is not None and not owner.error else "error"
        self.uses.clear()
        self.ctx = {}
        self.open = None
        self.version += 1

    def _titles(self, text: str) -> None:
        for title, n in TITLE.findall(text or ""):
            self.titles[int(n)] = title


class Index:
    """Every run on this node within the last `hours`, read as their logs grow."""

    def __init__(self, folder: Path, hours: float = 24):
        self.folder, self.hours = Path(folder), hours
        self.runs: dict[int, Run] = {}
        self.totals: dict[str, float] = {}    # a session → its total after the latest run read
        self.titles: dict[int, str] = {}
        self.spawner: dict[str, str] = {}     # an agent → the one that /spawn'ed it
        self.version = 0
        self.loaded = False

    def refresh(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        cutoff = now - self.hours * 3600
        try:
            names = os.listdir(self.folder)
        except OSError:
            names = []
        for name in names:
            m = NAME.match(name)
            if m is None or int(m.group(1)) in self.runs:
                continue
            path = self.folder / name
            try:
                if path.stat().st_mtime < cutoff:
                    continue
            except OSError:
                continue
            self.runs[int(m.group(1))] = Run(int(m.group(1)), path)
        changed = False
        for rid in sorted(self.runs):
            run = self.runs[rid]
            if not run.look(now):
                continue
            changed = True
            first = run.turns[0] if run.turns else None
            if first is not None and first.before is None and first.total is not None:
                # no meta to say: the session's total after its run before this one, or 0 for a new session
                first.before = self.totals.get(run.session) if run.session in self.totals else (
                    0.0 if run.startup else None)
            done = [t.total for t in run.turns if t.total is not None]
            if done and run.session:
                self.totals[run.session] = done[-1]
            self.titles.update(run.titles)
            for child in run.spawns:
                if run.handle and child != run.handle:
                    self.spawner.setdefault(child, run.handle)
        for rid, run in list(self.runs.items()):
            last = max([t.end or t.last for t in run.turns], default=0)
            if run.ended and last < cutoff * 1000:
                del self.runs[rid]
                changed = True
        if changed or not self.loaded:
            self.version += 1
        self.loaded = True
        return changed

    def turns(self) -> list[Turn]:
        return [t for rid in sorted(self.runs) for t in self.runs[rid].turns]

    def title(self, thread: int) -> str:
        return self.titles.get(thread) or f"Thread {thread}"


def main(argv: list[str]) -> None:
    """python -m pointman_trace.runs [folder]: what the index sees, for a quick look."""
    folder = Path(argv[1]) if len(argv) > 1 else run_logs()
    t0 = time.time()
    idx = Index(folder)
    idx.refresh()
    turns = idx.turns()
    print(f"{len(idx.runs)} runs, {len(turns)} turns, {sum(len(t.spans) for t in turns)} spans "
          f"in {time.time() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
