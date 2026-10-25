"""Trace's blocks and edits as the extension protocol has them (pointman_extension_types; docs/dev/panes.md, "The
waterfall block" and "The traces block"). trace.py and the panes make them as dicts, in the docs' shape, so they're
tested without the SDK; these turn them into the messages the SDK sends, as they go.

- A block's `id` is its `name`, unique in the pane: edits and inputs say it.
- A thread is a ThreadId (`thr_17979`), an agent a MemberRef by its handle, a machine a MachineId by its name (a
  waterfall's says where its runs were, so a pane on every machine merges them by it).
- An edit is a `put_items` of a waterfall's spans (the only items Trace puts), or a `replace` of a block, keeping its
  items or not.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pointman_extension_types as types
from pointman_extension_types import Block, PaneEdit, TracesView, View, WaterfallView

Span = WaterfallView.Span
TOTALS = tuple(f.name for f in dataclasses.fields(WaterfallView.Totals))


def thread_id(n: int) -> types.ThreadId:
    return types.ThreadId(value=f"thr_{n}")


def thread_number(tid: types.ThreadId | None) -> str | None:
    """A ThreadId's number, as text (`thr_17979` → "17979"), for runs.thread_of."""
    return tid.value.removeprefix("thr_") if tid is not None and tid.value else None


def when(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")) if iso else None


def totals(t: dict | None) -> WaterfallView.Totals | None:
    if t is None:
        return None
    return WaterfallView.Totals(**{k: t[k] for k in TOTALS if t.get(k) is not None})


def span(s: dict) -> Span:
    return Span(id=s["id"], parent=s.get("parent"), kind=Span.Kind[f"KIND_{s['kind'].upper()}"], name=s["name"],
                detail=s.get("detail"), agent=types.MemberRef(handle=s["agent"]) if s.get("agent") else None,
                model=s.get("model"), service=s.get("service"), start_ms=s["start_ms"], end_ms=s.get("end_ms"),
                cost_usd=s.get("cost_usd"), tokens_in=s.get("tokens_in"), tokens_out=s.get("tokens_out"),
                tokens_cached=s.get("tokens_cached"), state=Span.State[f"STATE_{s['state'].upper()}"],
                input=s.get("input"), output=s.get("output"))


def waterfall(b: dict) -> Block:
    return Block(name=b["id"], view=View(waterfall=WaterfallView(
        title=b["title"], state=WaterfallView.State[f"STATE_{b['state'].upper()}"], started=when(b.get("started")),
        totals=totals(b.get("totals")), compact=bool(b.get("compact")),
        thread=thread_id(b["thread"]) if b.get("thread") else None, spans=[span(s) for s in b.get("spans") or []],
        machine=types.MachineId(value=b["machine"]) if b.get("machine") else None)))


def row(r: dict) -> TracesView.Row:
    return TracesView.Row(id=r["id"], thread=thread_id(r["thread"]), title=r["title"],
                          state=WaterfallView.State[f"STATE_{r['state'].upper()}"], start=when(r["start"]),
                          end=when(r.get("end")), agents=[types.MemberRef(handle=h) for h in r["agents"]],
                          machines=[types.MachineId(value=m) for m in r["machines"]], totals=totals(r["totals"]))


def traces(b: dict) -> Block:
    return Block(name=b["id"], view=View(traces=TracesView(days=b["days"], more=b["more"], loading=b["loading"],
                                                           items=[row(r) for r in b["items"]])))


def empty(b: dict) -> Block:
    return Block(name=b["id"], empty=types.Empty(title=b["title"], note=b.get("note") or None))


KINDS = {"waterfall": waterfall, "traces": traces, "empty": empty}


def block(b: dict) -> Block:
    return KINDS[b["type"]](b)


def blocks(bs: list[dict]) -> list[Block]:
    return [block(b) for b in bs]


def edit(e: dict) -> PaneEdit:
    if "put_items" in e:
        put = e["put_items"]
        return PaneEdit(put_items=PaneEdit.Items(block=put["block"],
                                                 items=[PaneEdit.Item(span=span(s)) for s in put["items"]]))
    r = e["replace"]
    return PaneEdit(replace=PaneEdit.Replace(block=block(r["block"]), keep_items=bool(r.get("keep_items"))))


def edits(es: list[dict]) -> list[PaneEdit]:
    return [edit(e) for e in es]
