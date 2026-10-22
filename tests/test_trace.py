import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pointman_trace import trace
from pointman_trace.runs import Index
from pointman_trace.trace import Waterfall, check, order

from helpers import T0, Log, text, tool


def apply(block: dict, ops: list[dict]) -> dict:
    """The patches as the node applies them to what it keeps (`/b/<id>/…` paths, add only)."""
    b = copy.deepcopy(block)
    for op in ops:
        assert op["op"] == "add", op
        parts = op["path"].split("/")[1:]
        assert parts[:2] == ["b", b["id"]], op
        target = b
        for p in parts[2:-1]:
            target = target[int(p)] if isinstance(target, list) else target[p]
        last = parts[-1]
        if isinstance(target, list):
            assert last == "-", op
            target.append(op["value"])
        else:
            target[last] = op["value"]
    return b


class WaterfallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.idx = Index(self.dir, hours=10**6)

    def tearDown(self):
        self.tmp.cleanup()

    def wf(self, thread: int = 500) -> Waterfall:
        w = Waterfall("tr", "A thread", thread=thread)
        return w

    def turns(self, thread: int = 500):
        self.idx.refresh()
        return [t for t in self.idx.turns() if thread in t.threads]

    def test_it_grows_by_appends_and_field_sets(self):
        a = Log(self.dir, 1, "scene", thread=500, cost_before=0.0)
        w = self.wf()
        w.take(self.turns())
        block = w.full({})
        self.assertIsNone(check(block))
        self.assertEqual([s["kind"] for s in block["spans"]], ["agent", "run", "llm"])
        self.assertEqual(block["state"], "live")
        self.assertEqual(block["thread"], 500)
        self.assertNotIn("compact", block)

        a.say("q1", tool("t1", "mcp__pointman__post_message", thread_id=500, text="hi"))
        a.results(("t1", "posted", False))
        w.take(self.turns())
        kind, ops = w.update({})
        self.assertEqual(kind, "patch")
        block = apply(block, ops)
        self.assertIsNone(check(block))
        self.assertEqual(block, w.block())
        self.assertEqual([s["kind"] for s in block["spans"]], ["agent", "run", "llm", "tool", "llm"])
        appended = [op for op in ops if op["path"].endswith("/spans/-")]
        self.assertEqual(len(appended), 2)

        a.say("q2", text("done"))
        a.result(1.5, took=5000)
        a.end()
        w.take(self.turns())
        kind, ops = w.update({})
        block = apply(block, ops)
        self.assertEqual(block, w.block())
        self.assertEqual(block["state"], "done")
        self.assertEqual(block["totals"]["cost_usd"], 1.5)
        self.assertTrue(all(s["state"] != "running" for s in block["spans"]))
        self.assertIsNone(w.update({}))                   # nothing new: nothing sent

    def test_a_new_agent_comes_before_its_runs(self):
        Log(self.dir, 1, "scene", thread=500)
        w = self.wf()
        w.take(self.turns())
        block = w.full({})
        b = Log(self.dir, 2, "helper", thread=500, start=T0 + 5000)
        b.say("x", text("hello"))
        w.take(self.turns())
        kind, ops = w.update({"helper": "scene"})
        block = apply(block, ops)
        self.assertIsNone(check(block))
        ids = [s["id"] for s in block["spans"]]
        self.assertLess(ids.index("a:helper"), ids.index("r:2"))
        self.assertEqual(block["spans"][ids.index("a:helper")]["parent"], "a:scene")

    def test_an_earlier_start_sends_it_whole_again(self):
        Log(self.dir, 2, "scene", thread=500, start=T0 + 60_000)
        w = self.wf()
        w.take(self.turns())
        w.full({})
        Log(self.dir, 1, "other", thread=500, start=T0)
        w.take(self.turns())
        kind, block = w.update({})
        self.assertEqual(kind, "set")
        self.assertIsNone(check(block))
        self.assertEqual(min(s["start_ms"] for s in block["spans"]), 0)

    def test_input_and_output_come_when_asked(self):
        a = Log(self.dir, 1, "scene", thread=500, prompt="You are @scene:\n\n[9] mention from @james in #x › “A” "
                                                          "(thread 500):\n  hi")
        a.say("q1", tool("t1", "Bash", command="ls -la", description="List"))
        a.results(("t1", "file1 file2", False))
        w = self.wf()
        w.take(self.turns())
        block = w.full({})
        self.assertFalse(any("input" in s or "output" in s for s in block["spans"]))
        ops = w.open("t:t1")
        self.assertEqual([op["path"].rsplit("/", 1)[1] for op in ops], ["input", "output"])
        block = apply(block, ops)
        span = next(s for s in block["spans"] if s["id"] == "t:t1")
        self.assertEqual(span["output"], "file1 file2")
        self.assertIn("ls -la", span["input"])
        run = apply(block, w.open("r:1"))
        self.assertIn("mention from @james", next(s for s in run["spans"] if s["id"] == "r:1")["input"])
        self.assertEqual(w.open("t:t1"), [])            # already there
        self.assertEqual(w.open("nope"), [])

    def test_compact_keeps_agents_and_runs(self):
        a = Log(self.dir, 1, "scene", thread=500)
        a.say("q1", tool("t1", "Bash", command="ls"))
        w = Waterfall("t500", "A thread", thread=500, compact=True)
        w.take(self.turns())
        block = w.full({})
        self.assertTrue(block["compact"])
        self.assertEqual([s["kind"] for s in block["spans"]], ["agent", "run"])
        self.assertEqual(block["totals"]["tool_calls"], 1)

    def test_a_big_trace_folds_its_oldest_runs(self):
        for rid in range(1, 6):
            log = Log(self.dir, rid, "scene", thread=500, live=rid == 5, start=T0 + rid * 600_000)
            for i in range(10):
                log.say(f"q{i}", tool(f"r{rid}t{i}", "Bash", command="true"))
                log.results((f"r{rid}t{i}", "", False))
            if rid < 5:
                log.result(float(rid))
        with mock.patch.object(trace, "FOLD_AT", 40), mock.patch.object(trace, "KEEP", 30):
            w = self.wf()
            w.take(self.turns())
            block = w.full({})
        self.assertIsNone(check(block))
        self.assertLessEqual(len(block["spans"]), 30)
        self.assertIn("r:1", w.folded)
        self.assertNotIn("r:5", w.folded)                # still running: never folded
        run1 = next(s for s in block["spans"] if s["id"] == "r:1")
        self.assertTrue(run1["detail"].startswith("21 calls folded"))
        self.assertEqual(block["totals"]["tool_calls"], 50)  # the totals still count everything

    def test_a_turn_across_threads_shows_each_its_part(self):
        a = Log(self.dir, 1, "scene", thread=500, cost_before=0.0)
        a.say("q1", tool("t1", "mcp__pointman__read_thread", thread_id=500))
        a.results(("t1", "the thread", False))
        a.say("q2", tool("t2", "mcp__pointman__post_message", thread_id=600, text="over here"))
        a.results(("t2", "posted", False))
        a.say("q3", tool("t3", "Bash", command="make"))
        self.idx.refresh()
        turn, = self.idx.turns()
        self.assertEqual(turn.threads, {500, 600})
        here, there = Waterfall("tr", "A", thread=500), Waterfall("tr", "B", thread=600)
        here.take([turn]), there.take([turn])
        a_block, b_block = here.full({}), there.full({})
        self.assertIsNone(check(a_block))
        self.assertIsNone(check(b_block))
        self.assertEqual([s["id"] for s in a_block["spans"]], ["a:scene", "r:1", "l:1.1", "t:t1"])
        self.assertEqual(a_block["state"], "done")       # it moved on: its part here is over
        run_here = a_block["spans"][1]
        self.assertEqual(run_here["end_ms"], a_block["spans"][3]["end_ms"])
        self.assertEqual([s["id"] for s in b_block["spans"]], ["a:scene", "r:1", "l:1.2", "t:t2", "l:1.3", "t:t3"])
        self.assertEqual(b_block["state"], "live")
        a.say("q4", text("done"))
        a.result(2.0)
        self.idx.refresh()
        here.update({}), there.update({})
        costs = (here.block()["totals"]["cost_usd"], there.block()["totals"]["cost_usd"])
        self.assertAlmostEqual(sum(costs), 2.0, places=2)   # shared by model calls: 1 here, 3 there
        self.assertAlmostEqual(costs[0], 0.5, places=2)

    def test_check_matches_the_nodes_rules(self):
        good = {"type": "waterfall", "id": "tr", "title": "t", "state": "live", "totals": {"ms": 1, "cost_usd": None},
                "spans": [{"id": "a", "parent": None, "kind": "agent", "name": "@a", "start_ms": 0, "end_ms": None,
                           "state": "running"},
                          {"id": "b", "parent": "a", "kind": "tool", "name": "x", "start_ms": 5, "end_ms": 3,
                           "state": "ok", "input": "y" * 300}]}
        self.assertIsNone(check(good))
        for path, value in [(("spans", 1, "parent"), "zzz"), (("spans", 1, "kind"), "thing"),
                            (("spans", 1, "state"), "done"), (("spans", 1, "start_ms"), -1),
                            (("spans", 1, "input"), "y" * 301), (("spans", 1, "id"), "a"),
                            (("state",), "running"), (("totals", "ms"), -5), (("spans", 1, "detail"), 3)]:
            bad = copy.deepcopy(good)
            target = bad
            for p in path[:-1]:
                target = target[p]
            target[path[-1]] = value
            self.assertIsNotNone(check(bad), path)

    def test_order_puts_parents_first(self):
        spans = [{"id": "c", "parent": "b", "kind": "tool", "start_ms": 1},
                 {"id": "b", "parent": "a", "kind": "run", "start_ms": 2},
                 {"id": "a", "parent": None, "kind": "agent", "start_ms": 3},
                 {"id": "d", "parent": "gone", "kind": "tool", "start_ms": 0}]
        out = order(spans, set())
        ids = [s["id"] for s in out]
        self.assertEqual(ids, ["a", "b", "c", "d"])     # d's parent never comes: it goes last, under nothing
        self.assertIsNone(out[3]["parent"])


if __name__ == "__main__":
    unittest.main()
