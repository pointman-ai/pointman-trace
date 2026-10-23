"""The week of traces the Live pane lists (EXT-70; James, Pointman thread 17979 [19168]: "most recent + live traces
by default", searchable and filterable): one row per thread an agent worked in on this machine in the last 7 days.

- **Each run summed up once:** a finished run's log doesn't change, so its summary (per thread: when, which agent,
  and the waterfall's totals) is kept in the extension's state folder and only new runs are read after that. A
  week on mac-m4 is ~2,000 runs and 1.3 GB of logs: ~20 s the first time, in the background, with the list
  `loading` meanwhile. Reading it all into the Index instead would hold ~400 MB.
- **Runs the Index has** (its look back, and every run still going) are summed up from it as they grow; the kept
  summaries give the rest.
- **A row** is a thread: its title, its agents, its first and last activity, and its totals, summed over every run
  that worked in it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .runs import NAME, Notes, Run, account
from .trace import iso, tally

DAYS = 7
AGAIN = 600                 # seconds before it looks for newly finished runs again, while a Live pane is open
FORMAT = 1                  # of a kept summary: one of another format is read again


def summarize(run: Run) -> dict:
    """A run's summary: for each thread it worked in, when, and the waterfall's totals there."""
    threads: dict[int, dict] = {}
    for t in run.turns:
        for n in t.threads:
            start = t.span(n)["start_ms"]
            r = threads.get(n)
            if r is None:
                r = threads[n] = {"start": start, "end": 0, "running": False, **tally([], n)}
            r["start"] = min(r["start"], start)
            r["end"] = max(r["end"], t.last_for(n))
            r["running"] = r["running"] or t.running_for(n)
            for k, v in tally([t], n).items():
                r[k] += v
    done = [t.total for t in run.turns if t.total is not None]
    return {"v": FORMAT, "rid": run.rid, "handle": run.handle or f"run-{run.rid}", "session": run.session,
            "total": done[-1] if done else None, "threads": threads, "titles": dict(run.titles)}


def rows(summaries, titles: dict[int, str], machine: str | None, now_ms: int, days: float = DAYS) -> list[dict]:
    """The list: a row per thread worked in within `days`, from runs' summaries (oldest run first, so a newer
    title wins). Running first, newest started first; then by last activity, newest first."""
    merged: dict[int, dict] = {}
    named: dict[int, str] = {}
    for s in summaries:
        named.update(s["titles"])
        for n, r in s["threads"].items():
            m = merged.get(n)
            if m is None:
                m = merged[n] = {"thread": n, "start": r["start"], "end": r["end"], "running": False, "agents": {},
                                 **tally([], n)}
            m["start"] = min(m["start"], r["start"])
            m["end"] = max(m["end"], r["end"])
            m["running"] = m["running"] or r["running"]
            m["agents"][s["handle"]] = min(m["agents"].get(s["handle"], r["start"]), r["start"])
            for k in tally([], n):
                m[k] += r[k]
    named.update(titles)
    cutoff = now_ms - days * 86_400_000
    live = sorted((m for m in merged.values() if m["running"]), key=lambda m: -m["start"])
    done = sorted((m for m in merged.values() if not m["running"] and m["end"] >= cutoff), key=lambda m: -m["end"])
    out = []
    for m in live + done:
        n = m["thread"]
        out.append({"id": str(n), "thread": n, "title": named.get(n) or f"Thread {n}",
                    "state": "live" if m["running"] else "done", "start": iso(m["start"]),
                    "end": None if m["running"] else iso(m["end"]),
                    "agents": sorted(m["agents"], key=lambda h: (m["agents"][h], h)),
                    "machines": [machine] if machine else [],
                    "totals": {"ms": max(0, m["end"] - m["start"]), "cost_usd": round(m["cost_usd"], 2),
                               **{k: m[k] for k in ("tokens", "tokens_cached", "llm_calls", "tool_calls",
                                                    "errors")}},
                    "_end": m["end"]})
    return out


class History:
    """Every finished run's summary within DAYS, kept in `file` (in the extension's state folder) when there is
    one, and filled in by a background read."""

    def __init__(self, folder: Path, file: Path | None, days: float = DAYS, notes: Notes | None = None):
        self.folder, self.file, self.days, self.notes = Path(folder), file, days, notes
        self.summaries: dict[int, dict] = {}
        self.version = 0
        self.loading = False                 # its first read of the week is going
        self.read_at = 0.0                   # when its last read began
        self.lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self._load()

    def _load(self) -> None:
        if self.file is None:
            return
        try:
            lines = self.file.read_text().splitlines()
        except OSError:
            return
        for line in lines:
            try:
                s = json.loads(line)
                s["threads"] = {int(n): r for n, r in s["threads"].items()}
                s["titles"] = {int(n): t for n, t in s["titles"].items()}
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            if s.get("v") == FORMAT and isinstance(s.get("rid"), int):
                self.summaries[s["rid"]] = s
        self.version += 1

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [self.summaries[rid] for rid in sorted(self.summaries)]

    def start(self, now: float | None = None) -> None:
        """Read what's new in the background: at once the first time, then every AGAIN seconds while asked."""
        now = time.time() if now is None else now
        if (self.reader is not None and self.reader.is_alive()) or (self.read_at and now - self.read_at < AGAIN):
            return
        self.read_at = now
        self.loading = self.reader is None
        self.reader = threading.Thread(target=self._read_safely, name="trace-history", daemon=True)
        self.reader.start()

    def _read_safely(self) -> None:
        try:
            self.read()
        finally:
            if self.loading:
                self.loading = False
                with self.lock:
                    self.version += 1

    def read(self, now: float | None = None) -> int:
        """Sum up every finished run of the week not summed up yet, oldest first: a run's cost needs its session's
        total after the run before it. Returns how many it read."""
        now = time.time() if now is None else now
        cutoff = now - self.days * 86_400
        try:
            names = os.listdir(self.folder)
        except OSError:
            names = []
        logs = []
        for name in names:
            m = NAME.match(name)
            if m is None:
                continue
            try:
                st = (self.folder / name).stat()
            except OSError:
                continue
            if st.st_mtime >= cutoff:
                logs.append((int(m.group(1)), self.folder / name, st))
        totals: dict[str, float] = {}
        read = 0
        for rid, path, st in sorted(logs, key=lambda x: x[0]):
            have = self.summaries.get(rid)
            if have is not None and have.get("size") == st.st_size and have.get("mtime") == int(st.st_mtime):
                if have["session"] and have["total"] is not None:
                    totals[have["session"]] = have["total"]
                continue
            run = Run(rid, path, texts=False)
            if self.notes is not None:
                self.notes.seed(run)        # the thread that woke it, noted while it went: its meta's gone
            run.look(now)
            if not run.ended:
                continue                     # still going: the Index has it
            account(run, totals)
            s = {**summarize(run), "size": st.st_size, "mtime": int(st.st_mtime)}
            with self.lock:
                self.summaries[rid] = s
                self.version += 1
            read += 1
            time.sleep(0)                    # let the panes' ticks in
        with self.lock:
            # by its log's age, as above: a run that worked in no thread is kept too, or it'd be read every time
            old = [rid for rid, s in self.summaries.items() if s.get("mtime", 0) < cutoff]
            for rid in old:
                del self.summaries[rid]
            if old:
                self.version += 1
        if read or old:
            self._save()
        return read

    def _save(self) -> None:
        if self.file is None:
            return
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.file.with_name(self.file.name + ".tmp")
            tmp.write_text("".join(json.dumps(s, separators=(",", ":")) + "\n" for s in self.snapshot()))
            os.replace(tmp, self.file)
        except OSError:
            pass                             # kept in memory: the next start reads the week again
