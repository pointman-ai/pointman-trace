"""The trace extension's process: `python -m pointman_trace`, started and spoken to by the node.

- `/trace thread [n]` opens the `trace` pane: one thread's agent work on this machine as a live waterfall (the
  thread it's run in, without n). An agent opens it with open_pane(extension="trace", pane="trace",
  target={"thread": n}).
- `/trace live` opens the `live` pane: every trace running now, then those finished in the last 2 h, each folded
  to its agents' bars; ↩ on one opens its full trace. Below them, the `traces` block lists every thread worked in
  over the last 7 days, a row each, for the apps' navigator to search and filter (EXT-70); `more` pages back.
- Once a second, while a pane is open, it reads what the run logs gained and sends each pane only its changes, as
  edits by block name and span id (typed.py makes the protocol's messages of them).
- A change to its settings starts it again, with the new ones.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime
from pathlib import Path

import pointman_extension_types as types
from pointman_extension import Extension, Refused

from . import typed
from .history import DAYS, History, rows, summarize
from .runs import Index, Notes, run_logs, thread_of
from .trace import Waterfall

LIVE_FOR = 2 * 3600 * 1000           # ms a finished trace stays in the Live pane
LIVE_MAX = 12                        # traces in the Live pane at most
IDLE = 15 * 60                       # seconds with no pane open before the index is let go
RECENT = 200                         # rows of the week's list at first, and added by each `more`
RECENT_MAX = 1000                    # rows at most: well within a pane's 1 MiB
RECENT_EVERY = 10                    # seconds between sends of the list when only its numbers moved
TITLE_MAX = 200                      # characters of a pane's title, the protocol's limit

ext = Extension()
lock = threading.RLock()
state: dict = {"index": None, "history": None, "notes": None, "idle_since": time.time()}
views: dict[str, tuple] = {}         # pane id → (pane, view)


def folder() -> Path:
    named = str(ext.settings.get("folder") or "").strip()
    return Path(named).expanduser() if named else run_logs()


def kept(name: str, suffix: str) -> Path | None:
    """A file of this extension's own, in its state folder, for the run-log folder it reads."""
    d = ext.node.get("state_dir")
    return Path(d) / f"{name}-{hashlib.sha1(str(folder()).encode()).hexdigest()[:12]}{suffix}" if d else None


def notes() -> Notes:
    if state["notes"] is None:
        state["notes"] = Notes(folder(), kept("notes", ".json"))
    return state["notes"]


def index() -> Index:
    if state["index"] is None:
        state["index"] = Index(folder(), float(ext.settings.get("hours") or 24), notes())
    return state["index"]


def history() -> History:
    if state["history"] is None:
        state["history"] = History(folder(), kept("history", ".jsonl"), notes=notes())
    return state["history"]


def machine() -> str:
    return str(ext.node.get("machine") or "this machine")


def empty(title: str, note: str) -> list[dict]:
    return [{"type": "empty", "id": "none", "title": title, "note": note}]


LIST = "recent"                      # the Live pane's `traces` block


def when(v) -> int | None:
    """An ISO time (or epoch ms) as epoch ms."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, str) and v:
        try:
            return int(datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            return None
    return None


class ThreadView:
    """The `trace` pane: one thread's waterfall."""

    seen: int | None = None              # the index's version it last showed, and when
    at = 0.0

    def __init__(self, thread: int | None):
        self.thread = thread
        self.title = "Trace"
        self.reset()

    def reset(self) -> None:
        self.wf: Waterfall | None = None
        self.shown: list | None = None       # an empty block's content, as sent

    def update(self, idx: Index):
        """("set", blocks), ("edit", edits) or None."""
        if self.thread is None:
            return self._empty("Which thread?", "Run /trace thread <number>, or /trace thread in the thread.")
        self.title = f"Trace · {idx.title(self.thread)}"[:TITLE_MAX]
        if not idx.loaded:
            return self._empty("Reading the run logs…", f"Agents' runs on {machine()} in the last "
                                                      f"{idx.hours:g} h.")
        turns = [t for t in idx.turns() if self.thread in t.threads]
        if self.wf is None:
            if not turns:
                return self._empty("No agent work here yet", f"Nothing in the last {idx.hours:g} h on {machine()}: "
                                                             "it shows here as it starts.")
            self.wf = Waterfall("tr", idx.title(self.thread), machine=ext.node.get("machine") or None)
            self.wf.take(turns)
            return "set", [self.wf.full(idx.spawner)]
        self.wf.title = idx.title(self.thread)
        self.wf.take(turns)
        got = self.wf.update(idx.spawner)
        if got is None:
            return None
        return ("set", [got[1]]) if got[0] == "set" else got

    def _empty(self, title: str, note: str):
        blocks = empty(title, note)
        if blocks == self.shown:
            return None
        self.shown = blocks
        return "set", blocks

    def open(self, block, span) -> list[dict]:
        return self.wf.open(span) if self.wf is not None and block == self.wf.bid else []


class LiveView:
    """The `live` pane: a compact waterfall for each trace running now, newest first, then for each that finished
    in the last 2 h. One is added when a trace starts, and taken off 2 h after it ends."""

    title = "Live traces"
    seen: int | None = None
    at = 0.0

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.wfs: dict[int, Waterfall] = {}
        self.order: list[int] | None = None
        self.shown: list | None = None
        self.memo: dict[int, tuple[int, dict]] = {}    # a run the Index has → (its version, its summary)
        self.all: list[dict] = []                      # the week's rows, as last worked out
        self.oldest: int | None = None                 # the last activity of the oldest finished row shown
        self.floor: int | None = None                  # epoch ms: rows back to this, after `more`
        self.listed: dict | None = None                # the list's items, more and loading, as the apps have them
        self.listed_at = 0.0

    def update(self, idx: Index):
        if not idx.loaded:
            return self._set(empty("Reading the run logs…", f"Agents' runs on {machine()}."))
        heads, edits = self._heads(idx)
        listed = self._list(idx)
        if heads is not None:
            return self._set(heads + [self._block(listed)])
        edits.extend(self._list_edits(listed))
        return ("edit", edits) if edits else None

    def _heads(self, idx: Index) -> tuple[list | None, list]:
        """The compact waterfalls: (all of them, to set) or (None, their edits)."""
        now = int(time.time() * 1000)
        recent: dict[int, list] = {}
        for t in idx.turns():
            for n in t.threads:
                if t.running_for(n) or t.last_for(n) >= now - LIVE_FOR:
                    recent.setdefault(n, []).append(t)
        for n, turns in recent.items():
            wf = self.wfs.get(n)
            if wf is None:
                wf = self.wfs[n] = Waterfall(f"t{n}", idx.title(n), thread=n, compact=True,
                                            machine=ext.node.get("machine") or None)
            wf.title = idx.title(n)
            wf.take(turns)
        for n, wf in list(self.wfs.items()):
            if not wf.running and wf.last < now - LIVE_FOR:
                del self.wfs[n]
        running = sorted((w for w in self.wfs.values() if w.running), key=lambda w: -w.first)
        done = sorted((w for w in self.wfs.values() if not w.running), key=lambda w: -w.last)
        shown = (running + done)[:max(LIVE_MAX, len(running))]
        order = [w.thread for w in shown]
        if not shown:
            first = self.order != [] or self.listed is None
            self.order = []
            return (empty("Nothing running", f"Agents' work on {machine()} shows here as it starts, and for 2 h "
                                             "after it ends.") if first else None), []
        if order != self.order or self.listed is None:
            self.order = order
            return [w.full(idx.spawner) for w in shown], []
        edits: list[dict] = []
        for w in shown:
            got = w.update(idx.spawner)
            if got is None:
                continue
            if got[0] == "set":
                return [x.full(idx.spawner) if x is not w else got[1] for x in shown], []
            edits.extend(got[1])
        return None, edits

    # ------------------------------------------------------------------ the week's list (EXT-70)

    def _list(self, idx: Index) -> dict:
        """The rows shown now: the newest RECENT (every running one, however many), or back to `floor` after
        `more`; at most RECENT_MAX."""
        summaries = {s["rid"]: s for s in history().snapshot()}
        for rid, run in idx.runs.items():
            memo = self.memo.get(rid)
            if memo is None or memo[0] != run.version:
                memo = self.memo[rid] = (run.version, summarize(run))
            summaries[rid] = memo[1]
        for rid in [r for r in self.memo if r not in idx.runs]:
            del self.memo[rid]
        self.all = rows([summaries[r] for r in sorted(summaries)], idx.titles, ext.node.get("machine") or None,
                        int(time.time() * 1000))
        live = sum(r["state"] == "live" for r in self.all)
        shown = self.all[:max(RECENT, live)]
        if self.floor is not None:
            shown = [r for r in self.all if r["state"] == "live" or r["_end"] >= self.floor] or shown
        shown = shown[:max(RECENT_MAX, live)]
        self.oldest = min((r["_end"] for r in shown if r["state"] == "done"), default=None)
        return {"more": len(shown) < len(self.all), "loading": history().loading,
                "items": [{k: v for k, v in r.items() if k != "_end"} for r in shown]}

    def _block(self, listed: dict) -> dict:
        self.listed, self.listed_at = listed, time.time()
        return {"type": "traces", "id": LIST, "days": DAYS, **listed}

    def _list_edits(self, listed: dict, now: bool = False) -> list[dict]:
        """The list, as a `replace` of its block: at once when a row comes, goes, moves or changes state, or `more`
        or `loading` does; numbers alone at most every RECENT_EVERY seconds."""
        was = self.listed or {}
        if listed == was:
            return []
        same = [(r["thread"], r["state"]) for r in listed["items"]] == [(r["thread"], r["state"])
                                                                         for r in was.get("items", [])]
        if (not now and same and listed["more"] == was.get("more") and listed["loading"] == was.get("loading")
                and time.time() - self.listed_at < RECENT_EVERY):
            return []
        return [{"replace": {"block": self._block(listed)}}]

    def more(self, before) -> list[dict]:
        """`more`: every row back to `before` (an ISO time, or the oldest row shown), and up to RECENT finished before
        it. The apps send the merged list's oldest `end` to every machine, which can be older than this one's oldest
        row: the rows between them come too, or they'd never show (waterfall's navigator, 17979 [19321])."""
        if self.listed is None:
            return []
        at = when(before) or self.oldest
        if at is None:
            return []
        older = [r["_end"] for r in self.all if r["state"] == "done" and r["_end"] < at][:RECENT]
        floor = min(older) if older else at
        self.floor = floor if self.floor is None else min(self.floor, floor)
        return self._list_edits(self._list(index()), now=True)

    def _set(self, blocks: list):
        if blocks == self.shown:
            return None
        self.shown = blocks
        return "set", blocks

    def open(self, block, span) -> list[dict]:
        for w in self.wfs.values():
            if w.bid == block:
                return w.open(span)
        return []


def send(pane, got) -> None:
    if got is None:
        return
    kind, value = got
    if kind == "set":
        pane.set(typed.blocks(value))
    elif value:
        pane.edit(typed.edits(value))


def opened(pane, view) -> types.PaneOpenResponse:
    """A pane's first content, and how it answers inputs and closes."""
    @pane.on_input
    def tapped(event: types.PaneInput):
        kind, fields = event.kind, event.fields or {}
        listed = isinstance(view, LiveView) and fields.get("block") in (None, LIST)
        if kind != "open" and not (kind == "more" and listed):
            raise Refused("a trace only opens its spans, and the Live pane's list pages back with `more`")
        with lock:
            edits = view.open(fields.get("block"), fields.get("span")) if kind == "open" else view.more(fields.get("before"))
            if edits:
                pane.edit(typed.edits(edits))

    @pane.on_close
    def closed(_reason):
        with lock:
            views.pop(pane.id, None)
            if not views:
                state["idle_since"] = time.time()

    got = None
    if lock.acquire(timeout=0.5):        # not while the first read of the logs holds it: the next tick sends it
        try:
            got = view.update(index())
        finally:
            lock.release()
    with lock:
        views[pane.id] = (pane, view)
    if got is None or got[0] != "set":
        blocks = view.shown if getattr(view, "shown", None) else empty("Reading the run logs…", "")
        view.shown = blocks
    else:
        blocks = got[1]
    return types.PaneOpenResponse(title=view.title, content=types.PaneContent(blocks=typed.blocks(blocks)))


@ext.pane("trace")
def open_trace(pane):
    thread = thread_of(pane.target.get("thread")) or thread_of(typed.thread_number(pane.context.thread))
    return opened(pane, ThreadView(thread))


@ext.pane("live")
def open_live(pane):
    with lock:
        history().start()                # the week's list: from what's kept, and what's new read behind it
    return opened(pane, LiveView())


@ext.command("trace.thread")
def trace_thread(job):
    n = thread_of(job.params.get("thread"))
    # the pane opens on every machine running Trace, as one view (EXT-65), whichever machine answers this
    return {"text": f"Opening the trace of thread {n}, from every machine running Trace." if n else
            "Opening this thread's trace, from every machine running Trace.",
            "pane": {"open": "trace", "target": {"thread": n} if n else {}}}


@ext.command("trace.live")
def trace_live(job):
    return {"text": f"Opening the traces running now on every machine running Trace, and the last {DAYS} days' "
                    "list.",
            "pane": {"open": "live", "target": {}}}


def tick() -> None:
    with lock:
        if not views:
            if state["index"] is not None and time.time() - state["idle_since"] > IDLE:
                state["index"] = None        # nobody's watching: let its memory go
                state["history"] = None
            return
        idx = index()
        idx.refresh()
        listing = any(isinstance(v, LiveView) for _, v in views.values())
        if listing:
            history().start()                # newly finished runs, every AGAIN seconds
        now = time.time()
        ready = {p.id for p in ext.panes()}      # their open is answered: the apps have their first content
        for pane, view in list(views.values()):
            if pane.id not in ready:
                continue
            seen = (idx.version, history().version if isinstance(view, LiveView) else None)
            if view.seen == seen and now - view.at < 30:   # nothing new: only the Live pane's 2 h moves
                continue
            view.seen, view.at = seen, now
            title = view.title
            got = view.update(idx)
            if view.title != title:
                pane.set_title(view.title)
            try:
                send(pane, got)
            except ValueError as why:        # its edits wouldn't apply to what the apps have: all of it, next tick
                ext.log(f"trace: {pane.id}: {why}")
                view.reset()
                view.seen = None
        ext.event("trace.status", {"folder": str(idx.folder), "runs": len(idx.runs), "panes": len(views)})


def note() -> None:
    """Each run's thread while it's going, for reading it after it ends (EXT-70): panes or none."""
    with lock:
        took = notes()
    took.look()


ext.every(1, tick, lane="trace")
ext.every(5, note, lane="notes")

if __name__ == "__main__":
    ext.run()
