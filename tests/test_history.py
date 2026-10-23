"""The Live pane's week of traces (EXT-70): a row per thread from every run's summary, each finished run read once
and kept, and `more` paging back."""

import importlib
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

from pointman_trace.history import History, rows, summarize
from pointman_trace.runs import Index, Notes
from pointman_trace.trace import Waterfall

from helpers import Log, text, tool

ROOT = Path(__file__).resolve().parents[1]
SDK = os.environ.get("POINTMAN_EXTENSIONS_SDK")
HOUR = 3_600_000


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.start = int(time.time() * 1000) - 5 * HOUR      # its logs' files are new, so their work is too
        self.now = time.time() + 60                          # past FRESH: a log without meta has ended

    def tearDown(self):
        self.tmp.cleanup()

    def run_log(self, rid, handle, thread, at, total, *, calls=1, startup=True, live=False):
        log = Log(self.dir, rid, handle, thread=thread, start=self.start + at, startup=startup, live=live)
        for i in range(calls):
            log.say(f"q{rid}.{i}", tool(f"t{rid}.{i}", "Bash", command="ls"))
            log.results((f"t{rid}.{i}", "ok", i == 1))
        if not live:
            log.say(f"q{rid}.z", text("done"))
            log.result(total)
            log.end()
            gone = time.time() - 30                          # its log stopped growing a while ago: it ended
            os.utime(log.path, (gone, gone))
        return log


class Rows(Base):
    def test_a_row_per_thread_with_its_agents_and_its_waterfalls_totals(self):
        self.run_log(1, "scene", 500, 0, 0.5, calls=2)
        self.run_log(2, "loupe", 500, 60_000, 0.25)
        self.run_log(3, "scene", 600, 120_000, 0.9, startup=False)     # its session's 0.5 before: 0.4
        idx = Index(self.dir, 24)
        idx.refresh(self.now)
        got = rows([summarize(idx.runs[r]) for r in sorted(idx.runs)], idx.titles, "mac-m4",
                   int(self.now * 1000))
        self.assertEqual([r["thread"] for r in got], [600, 500])        # last activity, newest first
        late, both = got
        self.assertEqual((both["id"], both["title"], both["state"], both["agents"], both["machines"]),
                         ("500", "A thread", "done", ["scene", "loupe"], ["mac-m4"]))
        self.assertEqual(late["totals"]["cost_usd"], 0.4)
        self.assertTrue(both["start"].endswith("Z") and both["end"] > both["start"])
        # the same numbers as the thread's own waterfall
        wf = Waterfall("tr", "A thread", thread=500)
        wf.take([t for t in idx.turns() if 500 in t.threads])
        wf.full(idx.spawner)
        self.assertEqual(both["totals"], wf.head["totals"])
        self.assertEqual((both["totals"]["tool_calls"], both["totals"]["errors"], both["totals"]["cost_usd"]),
                         (3, 1, 0.75))
        # a week back, and no further
        self.assertEqual(rows([summarize(idx.runs[1])], {}, None, self.start + 8 * 24 * HOUR), [])

    def test_running_rows_come_first_with_no_end(self):
        self.run_log(1, "scene", 500, 0, 0.5)
        self.run_log(2, "loupe", 700, 60_000, 0.1, live=True)
        idx = Index(self.dir, 24)
        idx.refresh(time.time())
        got = rows([summarize(idx.runs[r]) for r in sorted(idx.runs)], idx.titles, None, int(time.time() * 1000))
        self.assertEqual([(r["thread"], r["state"], r["end"], r["machines"]) for r in got],
                         [(700, "live", None, []), (500, "done", got[1]["end"], [])])

    def test_each_finished_run_is_read_once_and_kept(self):
        self.run_log(1, "scene", 500, 0, 0.5)
        self.run_log(2, "loupe", 500, 60_000, 0.25)
        going = self.run_log(3, "scene", 600, 120_000, 0, startup=False, live=True)
        state = self.dir / "state" / "history.jsonl"
        h = History(self.dir, state)
        self.assertEqual(h.read(self.now), 2)                 # the one still going is the Index's
        self.assertEqual(sorted(h.summaries), [1, 2])
        kept = History(self.dir, state)                       # a restart: from the file, nothing read again
        self.assertEqual(sorted(kept.summaries), [1, 2])
        self.assertEqual(kept.read(self.now), 0)
        going.say("q3.z", text("done"))
        going.result(0.9)
        going.end()
        self.assertEqual(kept.read(self.now + 1), 1)
        # its session's total before it came from what was kept: 0.9 − 0.5
        self.assertEqual(kept.summaries[3]["threads"][600]["cost_usd"], 0.4)
        self.assertEqual([r["thread"] for r in rows(kept.snapshot(), {}, None, int(self.now * 1000))], [600, 500])
        # nowhere to keep it: read each time, the same rows
        bare = History(self.dir, None)
        self.assertEqual(bare.read(self.now + 1), 3)
        self.assertEqual(rows(bare.snapshot(), {}, None, 0), rows(kept.snapshot(), {}, None, 0))

    def test_a_run_noted_while_it_went_keeps_its_thread_after(self):
        going = self.run_log(1, "scene", 800, 0, 0, live=True)    # none of its calls names a thread
        file = self.dir / "state" / "notes.json"
        self.assertTrue(Notes(self.dir, file).look())
        self.assertFalse(Notes(self.dir, file).look())            # noted once, and kept
        going.say("q1.z", text("done"))
        going.result(0.2)
        going.end()
        (self.dir / "run-1.prompt").unlink()                      # the node takes its meta and prompt away
        gone = time.time() - 30
        os.utime(going.path, (gone, gone))
        bare = History(self.dir, None)
        bare.read(self.now)
        self.assertEqual(bare.summaries[1]["threads"], {})        # without notes: in no thread
        noted = History(self.dir, None, notes=Notes(self.dir, file))
        noted.read(self.now)
        self.assertEqual((list(noted.summaries[1]["threads"]), noted.summaries[1]["titles"]),
                         ([800], {800: "A thread"}))
        self.assertEqual([r["thread"] for r in rows(noted.snapshot(), {}, None, int(self.now * 1000))], [800])

    def test_its_first_read_says_loading_until_done(self):
        self.run_log(1, "scene", 500, 0, 0.5)
        h = History(self.dir, None)
        h.start()
        self.assertTrue(h.loading or h.summaries)
        h.reader.join(10)
        self.assertFalse(h.loading)
        self.assertEqual(list(h.summaries), [1])
        h.start()                                              # again within AGAIN: nothing
        self.assertFalse(h.loading)


@unittest.skipUnless(SDK and Path(SDK, "pointman_extension.py").is_file(), "needs the node's Python SDK")
class Paging(Base):
    def live(self):
        """The extension's module, as the node starts it (its manifest beside its __main__)."""
        sys.path.insert(0, SDK)
        fake = types.ModuleType("__main__")
        fake.__file__ = str(ROOT / "pointman_trace" / "__main__.py")
        real, sys.modules["__main__"] = sys.modules["__main__"], fake
        try:
            m = importlib.import_module("pointman_trace.__main__")
        finally:
            sys.modules["__main__"] = real
        self.addCleanup(setattr, m, "RECENT", m.RECENT)
        m.RECENT = 2
        idx = Index(self.dir, 24)
        idx.refresh(self.now)
        m.state.update(index=idx, history=History(self.dir, None))
        return m, idx

    def test_more_pages_back_a_page_at_a_time(self):
        for n in range(5):
            self.run_log(n + 1, "scene", 501 + n, n * 60_000, 0.1)
        m, idx = self.live()
        view = m.LiveView()
        kind, blocks = view.update(idx)
        listed = blocks[-1]
        self.assertEqual((kind, listed["type"], listed["more"]), ("set", "traces", True))
        self.assertEqual([r["thread"] for r in listed["items"]], [505, 504])
        ops = view.more(listed["items"][-1]["end"])
        self.assertEqual({op["path"] for op in ops}, {"/b/recent/items"})
        self.assertEqual([r["thread"] for r in ops[0]["value"]], [505, 504, 503, 502])
        ops = view.more(None)                                  # from the oldest shown
        got = {op["path"]: op["value"] for op in ops}
        self.assertEqual(([r["thread"] for r in got["/b/recent/items"]], got["/b/recent/more"]),
                         ([505, 504, 503, 502, 501], False))
        self.assertEqual(view.more(None), [])                  # nothing older
        self.assertIsNone(view.update(idx))                    # and nothing new

    def test_more_from_another_machines_older_end_brings_the_rows_between(self):
        # the apps send the merged list's oldest end to every machine (17979 [19321]): here it's older than all of
        # this machine's rows, so none is "before" it, yet the three past the first page must still come
        for n in range(5):
            self.run_log(n + 1, "scene", 501 + n, n * 60_000, 0.1)
        m, idx = self.live()
        view = m.LiveView()
        view.update(idx)
        ops = view.more("2026-01-01T00:00:00.000Z")
        got = {op["path"]: op["value"] for op in ops}
        self.assertEqual(([r["thread"] for r in got["/b/recent/items"]], got["/b/recent/more"]),
                         ([505, 504, 503, 502, 501], False))

    def test_numbers_alone_wait_a_few_seconds(self):
        self.run_log(1, "scene", 501, 0, 0.1)
        going = self.run_log(2, "loupe", 502, 60_000, 0, live=True)
        m, idx = self.live()
        view = m.LiveView()
        view.update(idx)
        going.say("q2.b", tool("t2.b", "Bash", command="ls"))
        idx.refresh(time.time())
        got = view.update(idx)
        self.assertFalse(got and any(op["path"].startswith("/b/recent") for op in got[1]))
        view.listed_at -= m.RECENT_EVERY
        got = view.update(idx)
        self.assertIn("/b/recent/items", [op["path"] for op in got[1]])


if __name__ == "__main__":
    unittest.main()
