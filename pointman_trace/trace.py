"""The waterfall block (docs/dev/panes.md, "The waterfall block"; loupe's shape, Pointman thread 17979 [18057]),
made from runs' turns and kept current with edits the node takes. Blocks and edits are dicts here, in the docs'
shape; typed.py makes them the protocol's messages as they go.

- **Spans only append,** each after its parent: a new one goes on the end, and one that changed is sent again
  whole where it is, both by its id (`put_items`). The block's own fields (its state and totals) go as a `replace`
  that keeps its spans.
- **Checked here as the node checks it** (node/src/extensions/panes.rs) before anything goes: the node drops edits
  whose result doesn't check, and every later span under one it dropped would be dropped too. Content that
  wouldn't check goes as a whole `set` of what does.
- **Small enough:** a pane's content is at most 1 MiB, so a trace past FOLD_AT spans (or MAX_BYTES) goes again
  whole, with its oldest finished runs folded to their run span, until KEEP spans are left.
- **Input and output on demand:** a `set` carries none; `open` puts a span again with its own.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

from collections import Counter

from .runs import Turn

KINDS = ("agent", "run", "llm", "tool", "task")
STATES = ("running", "ok", "error")
DEPTH = {"agent": 0, "run": 1, "task": 2, "llm": 3, "tool": 3}
MAX_SPANS = 10_000           # the node's
SPAN_TEXT = 300              # the node's, for input and output
MAX_ID = 128                 # the node's, for every id
FOLD_AT = 3000               # spans before a trace goes again with old runs folded
KEEP = 2000                  # spans left after folding
MAX_BYTES = 800_000          # of a pane's content, under the node's 1 MiB
TEXT_FIELDS = ("detail", "agent", "model", "service")
AMOUNTS = ("start_ms", "end_ms", "cost_usd", "tokens_in", "tokens_out", "tokens_cached")


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def amount(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def check(block: dict) -> str | None:
    """Why the node would refuse this waterfall block, or None."""
    bid = block.get("id")
    if not isinstance(bid, str) or not 1 <= len(bid) <= MAX_ID:
        return "a block id"
    if not isinstance(block.get("title"), str):
        return "title must be text"
    if block.get("state") not in ("live", "done"):
        return "state must be live or done"
    if "compact" in block and not isinstance(block["compact"], bool):
        return "compact must be true or false"
    if block.get("thread") is not None and not (isinstance(block["thread"], int) and block["thread"] > 0):
        return "thread must be a thread's id"
    totals = block.get("totals")
    if totals is not None and not (isinstance(totals, dict) and all(v is None or amount(v) for v in totals.values())):
        return "totals must be numbers, none below 0"
    spans = block.get("spans")
    if not isinstance(spans, list) or len(spans) > MAX_SPANS:
        return f"spans must be a list of at most {MAX_SPANS}"
    before: set[str] = set()
    for n, s in enumerate(spans):
        if not isinstance(s, dict):
            return f"spans/{n} must be an object"
        sid = s.get("id")
        if not isinstance(sid, str) or not 1 <= len(sid) <= MAX_ID:
            return f"spans/{n} has no id"
        if s.get("kind") not in KINDS or s.get("state") not in STATES:
            return f"spans/{n} ({sid}): kind or state"
        if not isinstance(s.get("name"), str):
            return f"spans/{n} ({sid}): name"
        if not amount(s.get("start_ms")):
            return f"spans/{n} ({sid}): start_ms"
        for k in AMOUNTS[1:]:
            if s.get(k) is not None and not amount(s[k]):
                return f"spans/{n} ({sid}): {k}"
        for k in TEXT_FIELDS:
            if s.get(k) is not None and not isinstance(s[k], str):
                return f"spans/{n} ({sid}): {k}"
        for k in ("input", "output"):
            if s.get(k) is not None and not (isinstance(s[k], str) and len(s[k]) <= SPAN_TEXT):
                return f"spans/{n} ({sid}): {k}"
        if s.get("parent") is not None and s["parent"] not in before:
            return f"spans/{n} ({sid}): parent {s['parent']} isn't before it"
        if sid in before:
            return f"spans/{n}: a second {sid}"
        before.add(sid)
    return None


def put_spans(bid: str, spans: list[dict]) -> dict:
    """An edit: a waterfall's spans, each whole and by its id, replaced where it is or added at the end."""
    return {"put_items": {"block": bid, "items": [dict(s) for s in spans]}}


def keep_spans(block: dict) -> dict:
    """An edit: a block's own fields (its title, state and totals), its spans kept as they are."""
    return {"replace": {"block": block, "keep_items": True}}


def size(v) -> int:
    return len(json.dumps(v, ensure_ascii=False, separators=(",", ":")).encode())


def tally(turns, thread: int | None) -> dict:
    """A waterfall's totals but its time, for these turns' work in a thread (all of it, for None): the Live list's
    rows add up the same (EXT-70)."""
    llm = tools = errors = tokens = cached = 0
    cost = 0.0
    for t in turns:
        for s in t.mine(thread):
            if s["kind"] == "llm":
                llm += 1
                tokens += s["tokens_in"] + s["tokens_out"]
                cached += s["tokens_cached"]
            else:
                tools += 1
                errors += s["state"] == "error"
        cost += t.span(thread)["cost_usd"] or 0.0
    return {"cost_usd": cost, "tokens": tokens, "tokens_cached": cached, "llm_calls": llm, "tool_calls": tools,
            "errors": errors}


class Waterfall:
    """One waterfall block as the apps have it: a thread's turns, its agents above them. `compact` keeps only the
    agents and their runs (the Live pane's)."""

    def __init__(self, bid: str, title: str, *, thread: int | None = None, compact: bool = False):
        self.bid, self.title, self.thread, self.compact = bid, title, thread, compact
        self.turns: dict[str, Turn] = {}       # every turn it has shown, kept while it's open
        self.started: int | None = None        # epoch ms its times count from
        self.sent: list[dict] | None = None    # its spans as the apps have them
        self.index: dict[str, int] = {}
        self.head: dict = {}                   # its title, state and totals as sent
        self.folded: set[str] = set()          # turns sent without their calls
        self.owner: dict[str, Turn] = {}       # a span's id → its turn, for input on demand
        self.bytes = 0

    # ------------------------------------------------------------------ what it is now

    def take(self, turns) -> None:
        for t in turns:
            self.turns.setdefault(t.id, t)

    @property
    def running(self) -> bool:
        return any(t.running_for(self.thread) for t in self.turns.values())

    @property
    def last(self) -> int:
        return max((t.last_for(self.thread) for t in self.turns.values()), default=0)

    @property
    def first(self) -> int:
        return min((t.span(self.thread)["start_ms"] for t in self.turns.values()), default=0)

    def _spans(self, spawner: dict[str, str]) -> list[dict]:
        """Every span now, epoch ms, without input and output: agents, then each turn and its calls."""
        runs = sorted(((t, t.span(self.thread)) for t in self.turns.values()), key=lambda x: (x[1]["start_ms"], x[0].id))
        agents: dict[str, list[dict]] = {}
        for t, run in runs:
            agents.setdefault(t.handle, []).append(run)
        out = []
        for handle, rs in agents.items():
            models = Counter(r["model"] for r in rs if r.get("model"))
            costs = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
            running = any(r["end_ms"] is None for r in rs)
            boss = spawner.get(handle)
            out.append({"id": f"a:{handle}", "parent": f"a:{boss}" if boss in agents and boss != handle else None,
                        "kind": "agent", "name": f"@{handle}", "agent": handle,
                        "model": models.most_common(1)[0][0] if models else None,
                        "start_ms": min(r["start_ms"] for r in rs),
                        "end_ms": None if running else max(r["end_ms"] for r in rs),
                        "cost_usd": round(sum(costs), 4) if costs else None,
                        "state": "running" if running else "ok"})
        for t, run in runs:
            run["agent"] = t.handle
            if t.prompt:
                self.owner[t.id] = t
            out.append(run)
            if self.compact or t.id in self.folded:
                continue
            calls = t.mine(self.thread)
            mine = {t.id, *(s["id"] for s in calls)}
            for s in calls:
                s = {k: v for k, v in s.items() if not k.startswith("_")}
                s["agent"] = t.handle
                if s["parent"] not in mine:       # a background sub-agent's call, its task in an earlier turn
                    s["parent"] = t.id
                self.owner[s["id"]] = t
                out.append(s)
        return out

    def _totals(self) -> dict:
        got = tally(self.turns.values(), self.thread)
        return {"ms": max(0, self.last - self.first), **got, "cost_usd": round(got["cost_usd"], 2)}

    def _rel(self, s: dict) -> dict:
        """A span as the block has it: its times from `started`, the rest made to check (nothing below 0, text
        that's text), and nothing that's null but end_ms."""
        out = {}
        for k, v in s.items():
            if k in ("start_ms", "end_ms") and v is not None:
                v = max(0, int(v) - self.started)
            elif k in AMOUNTS and v is not None:
                v = v if amount(v) else 0
            elif k in TEXT_FIELDS and v is not None and not isinstance(v, str):
                v = str(v)
            if v is not None or k in ("end_ms", "parent"):
                out[k] = v
        if out.get("kind") not in KINDS:
            out["kind"] = "tool"
        if out.get("state") not in STATES:
            out["state"] = "ok"
        if not isinstance(out.get("name"), str) or not out["name"]:
            out["name"] = "?"
        return out

    def _head(self) -> dict:
        return {"title": self.title, "state": "live" if self.running else "done", "totals": self._totals()}

    def _view(self) -> dict:
        """The block as it is now, not a copy: for checking."""
        b = {"type": "waterfall", "id": self.bid, **self.head, "spans": self.sent or []}
        if self.compact:
            b["compact"] = True
        if self.thread:
            b["thread"] = self.thread
        return b

    def block(self, spans: bool = True) -> dict:
        """The block as the apps have it; without its spans, for a `replace` that keeps them."""
        b = {"type": "waterfall", "id": self.bid, **self.head, "started": iso(self.started or 0)}
        if self.compact:
            b["compact"] = True
        if self.thread:
            b["thread"] = self.thread
        b["spans"] = [dict(x) for x in self.sent or []] if spans else []   # a copy: what's sent mustn't change
        return copy.deepcopy(b)

    # ------------------------------------------------------------------ sending

    def full(self, spawner: dict[str, str]) -> dict:
        """All of it again, as a `set` has it: oldest finished runs folded until it fits."""
        self.folded.clear()
        self.started = self.first
        spans = self._spans(spawner)
        if len(spans) > FOLD_AT or self._size(spans) > MAX_BYTES:
            spans = self._fold(spans, spawner)
        self.sent = [self._rel(s) for s in order(spans, set())]
        self._fix_parents(self.sent)
        self.index = {s["id"]: n for n, s in enumerate(self.sent)}
        self.head = self._head()
        self.bytes = size(self.sent)
        return self.block()

    def _size(self, spans) -> int:
        return sum(size(s) for s in spans)

    def _fold(self, spans: list[dict], spawner) -> list[dict]:
        finished = sorted((t for t in self.turns.values() if not t.running_for(self.thread)),
                          key=lambda t: t.last_for(self.thread))
        count, used = len(spans), self._size(spans)
        for t in finished:
            if count <= KEEP and used <= MAX_BYTES * 0.8:
                break
            self.folded.add(t.id)
            count -= len(t.mine(self.thread))
            used -= 250 * len(t.mine(self.thread))
        spans = self._spans(spawner)
        for s in spans:
            if s["kind"] == "run" and s["id"] in self.folded:
                n = len(self.turns[s["id"]].mine(self.thread))
                s["detail"] = f"{n} calls folded" + (f" · {s['detail']}" if s.get("detail") else "")
        return spans

    def _fix_parents(self, spans: list[dict]) -> None:
        seen: set[str] = set()
        for s in spans:
            if s.get("parent") is not None and s["parent"] not in seen:
                s["parent"] = None
            seen.add(s["id"])

    def update(self, spawner: dict[str, str]) -> tuple[str, object] | None:
        """What to send now: ("set", block) for all of it, ("edit", edits), or None when nothing changed."""
        if self.sent is None:
            return "set", self.full(spawner)
        spans = self._spans(spawner)
        new = [s for s in spans if s["id"] not in self.index]
        if (any(s["start_ms"] < self.started for s in spans if s["id"] not in self.index)
                or len(self.sent) + len(new) > FOLD_AT * 1.3 or self.bytes + 300 * len(new) > MAX_BYTES):
            return "set", self.full(spawner)
        put: list[dict] = []
        now = {s["id"]: s for s in spans}
        for sid, n in self.index.items():
            s = now.get(sid)
            if s is None:
                continue
            was, want = self.sent[n], self._rel(s)
            changed = {k: v for k, v in want.items() if k not in ("id", "parent", "kind") and was.get(k) != v}
            if changed:
                was.update(changed)
                put.append(was)
        for s in order(new, set(self.index)):
            s = self._rel(s)
            if s.get("parent") is not None and s["parent"] not in self.index:
                s["parent"] = None
            self.index[s["id"]] = len(self.sent)
            self.sent.append(s)
            self.bytes += size(s)
            put.append(s)
        edits = [put_spans(self.bid, put)] if put else []
        head = self._head()
        if head != self.head:
            self.head = head
            edits.append(keep_spans(self.block(spans=False)))
        if not edits:
            return None
        if check(self._view()) is not None:
            block = self.full(spawner)
            if (why := check(block)) is not None:
                block = self._broken(why)
            return "set", block
        return "edit", edits

    def _broken(self, why: str) -> dict:
        self.sent, self.index = [], {}
        self.head = {"title": f"{self.title} (couldn't draw: {why})", "state": "done", "totals": None}
        return self.block()

    def open(self, sid: str) -> list[dict]:
        """A span's input and output, as an edit that puts it again with them: a `set` carries neither."""
        n = self.index.get(sid)
        t = self.owner.get(sid)
        if n is None or t is None or self.bytes > MAX_BYTES:    # full: what more it holds must stay under 1 MiB
            return []
        if sid == t.id:
            texts = [t.prompt, None]
        else:
            texts = t.texts.get(sid) or [None, None]
        changed = False
        for k, v in zip(("input", "output"), texts):
            if isinstance(v, str) and v and self.sent[n].get(k) != v[:SPAN_TEXT]:
                v = v[:SPAN_TEXT]
                self.sent[n][k] = v
                self.bytes += len(v) + 12
                changed = True
        return [put_spans(self.bid, [self.sent[n]])] if changed else []


def order(spans: list[dict], placed: set[str]) -> list[dict]:
    """By start (a parent first when they start together), each after its parent: one whose parent isn't there
    waits for it, and goes under nothing if it never comes."""
    out: list[dict] = []
    placed = set(placed)
    waiting: dict[str, list[dict]] = {}

    def place(s: dict) -> None:
        stack = [s]
        while stack:
            x = stack.pop()
            out.append(x)
            placed.add(x["id"])
            stack.extend(reversed(waiting.pop(x["id"], [])))

    for s in sorted(spans, key=lambda s: (s["start_ms"], DEPTH.get(s["kind"], 3))):
        p = s.get("parent")
        if p is None or p in placed:
            place(s)
        else:
            waiting.setdefault(p, []).append(s)
    for ws in list(waiting.values()):
        for s in ws:
            if s["id"] not in placed:
                s["parent"] = None
                place(s)
    return out
