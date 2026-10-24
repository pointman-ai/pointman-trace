"""typed.py: the blocks and edits Trace makes as dicts, as the protocol's messages, applied by the SDK's own rules
(pointman_extension.apply_edits, as the node applies them). Skipped without the node's typed SDK."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from pointman_trace.runs import Index
from pointman_trace.trace import Waterfall

from helpers import Log, text, tool

SDK = os.environ.get("POINTMAN_EXTENSIONS_SDK")
TYPED = bool(SDK and Path(SDK, "pointman_extension_types.py").is_file())

if TYPED:
    sys.path.insert(0, SDK)
    import pointman_extension_types as types
    from pointman_extension import apply_edits

    from pointman_trace import typed


@unittest.skipUnless(TYPED, "needs the node's typed Python SDK")
class TypedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.idx = Index(self.dir, hours=10**6)

    def turns(self):
        self.idx.refresh()
        return [t for t in self.idx.turns() if 500 in t.threads]

    def content(self, *blocks):
        return types.PaneContent(blocks=typed.blocks(list(blocks)))

    def test_a_waterfalls_edits_apply_as_its_own(self):
        log = Log(self.dir, 1, "scene", thread=500, cost_before=0.0)
        log.say("q1", tool("t1", "Bash", command="ls"))
        w = Waterfall("tr", "A thread", thread=500)
        w.take(self.turns())
        shown = self.content(w.full({}))
        log.results(("t1", "a b", False))
        log.say("q2", text("done"))
        log.result(1.5)
        log.end()
        w.take(self.turns())
        kind, edits = w.update({})
        self.assertEqual(kind, "edit")
        shown = apply_edits(shown, typed.edits(edits))
        self.assertEqual(shown, self.content(w.block()))
        shown = apply_edits(shown, typed.edits(w.open("t:t1")))
        self.assertEqual(shown, self.content(w.block()))
        wf = shown.blocks[0].view.waterfall
        self.assertEqual((wf.state, wf.totals.cost_usd, wf.spans[3].output),
                         (types.WaterfallView.State.STATE_DONE, 1.5, "a b"))

    def test_blocks_go_as_the_protocols_json(self):
        Log(self.dir, 1, "scene", thread=500).say("q1", tool("t1", "Bash", command="ls"))
        w = Waterfall("t500", "A thread", thread=500, compact=True)
        w.take(self.turns())
        row = {"id": "500", "thread": 500, "title": "A thread", "state": "live", "start": "2026-10-23T01:12:00Z",
               "end": None, "agents": ["scene"], "machines": ["mac-m4"],
               "totals": {"ms": 5, "cost_usd": 0.5, "tokens": 30, "tokens_cached": 1000, "llm_calls": 1,
                          "tool_calls": 1, "errors": 0}}
        blocks = [w.full({}), {"type": "traces", "id": "recent", "days": 7, "more": True, "loading": False,
                               "items": [row]},
                  {"type": "empty", "id": "none", "title": "Nothing running", "note": ""}]
        for b in typed.blocks(blocks):
            self.assertEqual(types.Block.from_json(b.to_json()), b)
        wf, listed, empty = (b.to_json() for b in typed.blocks(blocks))
        self.assertEqual((wf["name"], wf["view"]["waterfall"]["thread"], wf["view"]["waterfall"]["compact"],
                          [s["kind"] for s in wf["view"]["waterfall"]["spans"]]),
                         ("t500", {"value": "thr_500"}, True, ["KIND_AGENT", "KIND_RUN"]))
        item, = listed["view"]["traces"]["items"]
        self.assertEqual((item["thread"], item["state"], item["agents"], item["machines"], "end" in item),
                         ({"value": "thr_500"}, "STATE_LIVE", [{"handle": "scene"}], [{"value": "mac-m4"}], False))
        self.assertEqual(empty, {"name": "none", "empty": {"title": "Nothing running"}})


if __name__ == "__main__":
    unittest.main()
