"""The trace extension's process: `python -m pointman_trace`, started and spoken to by the node.

- `/trace thread [n]` opens the `trace` pane: one thread's agent work on this machine as a live waterfall (the
  thread it's run in, without n). An agent opens it with open_pane(extension="trace", pane="trace",
  target={"thread": n}).
- `/trace live` opens the `live` pane: every trace running now, then those finished in the last 2 h, each folded
  to its agents' bars; ↩ on one opens its full trace.
- Once a second, while a pane is open, it reads what the run logs gained and sends each pane only its changes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

try:
    from pointman_extension import Extension, Refused
except ImportError:                 # a node from before the rename
    from mb_extension import Extension, Refused  # type: ignore[no-redef]

from .runs import Index, run_logs, thread_of
from .trace import MAX_OPS, Waterfall

LIVE_FOR = 2 * 3600 * 1000           # ms a finished trace stays in the Live pane
LIVE_MAX = 12                        # traces in the Live pane at most
IDLE = 15 * 60                       # seconds with no pane open before the index is let go

ext = Extension()
lock = threading.RLock()
state: dict = {"index": None, "idle_since": time.time()}
views: dict[str, tuple] = {}         # pane id → (pane, view)


def index() -> Index:
    if state["index"] is None:
        folder = str(ext.settings.get("folder") or "").strip()
        state["index"] = Index(Path(folder).expanduser() if folder else run_logs(),
                               float(ext.settings.get("hours") or 24))
    return state["index"]


def machine() -> str:
    return str(ext.node.get("machine") or "this machine")


def empty(title: str, note: str) -> list[dict]:
    return [{"type": "empty", "id": "none", "title": title, "note": note}]


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
        """("set", blocks), ("patch", ops) or None."""
        if self.thread is None:
            return self._empty("Which thread?", "Run /trace thread <number>, or /trace thread in the thread.")
        self.title = f"Trace · {idx.title(self.thread)}"
        if not idx.loaded:
            return self._empty("Reading the run logs…", f"Agents' runs on {machine()} in the last "
                                                      f"{idx.hours:g} h.")
        turns = [t for t in idx.turns() if self.thread in t.threads]
        if self.wf is None:
            if not turns:
                return self._empty("No agent work here yet", f"Nothing in the last {idx.hours:g} h on {machine()}: "
                                                             "it shows here as it starts.")
            self.wf = Waterfall("tr", idx.title(self.thread))
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

    def update(self, idx: Index):
        if not idx.loaded:
            return self._set(empty("Reading the run logs…", f"Agents' runs on {machine()}."))
        now = int(time.time() * 1000)
        recent: dict[int, list] = {}
        for t in idx.turns():
            for n in t.threads:
                if t.running_for(n) or t.last_for(n) >= now - LIVE_FOR:
                    recent.setdefault(n, []).append(t)
        for n, turns in recent.items():
            wf = self.wfs.get(n)
            if wf is None:
                wf = self.wfs[n] = Waterfall(f"t{n}", idx.title(n), thread=n, compact=True)
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
            self.order = []
            return self._set(empty("Nothing running", f"Agents' work on {machine()} shows here as it starts, and "
                                                      "for 2 h after it ends."))
        if order != self.order:
            self.order = order
            return self._set([w.full(idx.spawner) for w in shown])
        ops: list[dict] = []
        for w in shown:
            got = w.update(idx.spawner)
            if got is None:
                continue
            if got[0] == "set":
                return self._set([x.full(idx.spawner) if x is not w else got[1] for x in shown])
            ops.extend(got[1])
        if len(ops) > MAX_OPS:             # more than the node takes in one patch: all of it again
            return self._set([w.full(idx.spawner) for w in shown])
        return ("patch", ops) if ops else None

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
        pane.set(value)
    elif value:
        pane.patch(value)


def opened(pane, view) -> dict:
    """A pane's first content, and how it answers inputs and closes."""
    @pane.on_input
    def tapped(event):
        if event.get("kind") != "open":
            raise Refused("a trace only opens its spans")
        with lock:
            ops = view.open(event.get("block"), event.get("span"))
            if ops:
                pane.patch(ops)

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
    return {"title": view.title, "blocks": blocks}


@ext.pane("trace")
def open_trace(pane):
    thread = thread_of(pane.target.get("thread")) or thread_of(pane.context.get("thread")) \
        or thread_of(pane.context.get("thread_id"))
    return opened(pane, ThreadView(thread))


@ext.pane("live")
def open_live(pane):
    return opened(pane, LiveView())


@ext.command("trace.thread")
def trace_thread(job):
    n = thread_of(job.params.get("thread"))
    return {"text": f"Opening the trace of thread {n} on {machine()}." if n else
            f"Opening this thread's trace on {machine()}.",
            "pane": {"open": "trace", "target": {"thread": n} if n else {}}}


@ext.command("trace.live")
def trace_live(job):
    return {"text": f"Opening the traces running on {machine()} now, and those from the last 2 h.",
            "pane": {"open": "live", "target": {}}}


def tick() -> None:
    with lock:
        if not views:
            if state["index"] is not None and time.time() - state["idle_since"] > IDLE:
                state["index"] = None        # nobody's watching: let its memory go
            return
        idx = index()
        idx.refresh()
        now = time.time()
        for pane, view in list(views.values()):
            if view.seen == idx.version and now - view.at < 30:   # nothing new: only the Live pane's 2 h moves
                continue
            view.seen, view.at = idx.version, now
            title = view.title
            got = view.update(idx)
            if view.title != title:
                pane.set_title(view.title)
            send(pane, got)
        ext.event("trace.status", {"folder": str(idx.folder), "runs": len(idx.runs), "panes": len(views)})


@ext.on_settings
def settings_changed(_settings) -> None:
    with lock:
        state["index"] = None
        for pane, view in views.values():
            view.reset()
            view.seen = None


ext.every(1, tick, lane="trace")

if __name__ == "__main__":
    ext.run()
