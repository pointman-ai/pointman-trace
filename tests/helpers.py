"""Run logs made up for the tests: Claude Code's stream-json events, as the node keeps them."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

T0 = 1_792_000_000_000           # epoch ms the made-up runs start at


def stamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Log:
    """One run's log, written a few events at a time, as a run writes it."""

    def __init__(self, folder: Path, rid: int, handle: str = "scene", *, startup: bool = True,
                 thread: int | None = None, cost_before: float | None = None, live: bool = True,
                 prompt: str | None = None, start: int = T0):
        self.folder, self.rid, self.t = Path(folder), rid, start
        self.path = self.folder / f"run-{rid}.json"
        self.session = f"sess-{handle}"
        if prompt is None:
            prompt = f"You are @{handle} on Pointman:\n\n[1] mention from @james in #x › “A thread”" + (
                f" (thread {thread}):\n    hi" if thread else ":\n    hi")
        (self.folder / f"run-{rid}.prompt").write_text(json.dumps(
            {"type": "user", "message": {"role": "user", "content": prompt}}) + "\n")
        os.utime(self.folder / f"run-{rid}.prompt", (start / 1000, start / 1000))
        if live:
            self.meta(handle=handle, thread=thread, cost_before=cost_before)
        self.write([{"type": "system", "subtype": "hook_started",
                     "hook_name": "SessionStart:startup" if startup else "SessionStart:resume"},
                    {"type": "system", "subtype": "session_title_changed", "title": f"board @{handle}",
                     "session_id": self.session},
                    {"type": "system", "subtype": "init", "session_id": self.session, "model": "claude-opus-5-5"}])

    def meta(self, **fields) -> None:
        m = {"client": "claude", "run_id": self.rid, "keeper": os.getpid(), "moves": 0,
             "launch": {"env": [["SECRET_KEY", "never-read"]]}, **fields}
        (self.folder / f"run-{self.rid}.meta.json").write_text(json.dumps(m))

    def end(self) -> None:
        """The run stops: the node takes its meta away."""
        (self.folder / f"run-{self.rid}.meta.json").unlink(missing_ok=True)

    def write(self, events: list[dict]) -> None:
        with open(self.path, "a") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def at(self, ms: int) -> str:
        self.t += ms
        return stamp(self.t)

    def say(self, req: str, *blocks, after: int = 1000, parent: str | None = None,
            usage: dict | None = None) -> None:
        msg = {"model": "claude-opus-5-5", "content": list(blocks),
               "usage": usage or {"input_tokens": 10, "cache_creation_input_tokens": 5,
                                  "cache_read_input_tokens": 1000, "output_tokens": 20}}
        self.write([{"type": "assistant", "timestamp": self.at(after), "request_id": req, "message": msg,
                     "parent_tool_use_id": parent, "session_id": self.session}])

    def results(self, *pairs, after: int = 500, parent: str | None = None) -> None:
        content = [{"type": "tool_result", "tool_use_id": tid, "content": text, "is_error": err}
                   for tid, text, err in pairs]
        self.write([{"type": "user", "timestamp": self.at(after), "message": {"role": "user", "content": content},
                     "parent_tool_use_id": parent}])

    def result(self, total: float, took: int | None = None, error: bool = False) -> None:
        self.write([{"type": "result", "subtype": "error_during_execution" if error else "success",
                     "is_error": error, "total_cost_usd": total, "session_id": self.session,
                     **({"duration_ms": took} if took is not None else {})}])


def tool(tid: str, name: str, **inp) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def text(s: str) -> dict:
    return {"type": "text", "text": s}
