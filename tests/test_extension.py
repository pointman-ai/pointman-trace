"""The extension's process as a node runs it, through the SDK's stand-in node (pointman_extension_testing): the
typed protocol both ways, and each pane's content as an app has it, with every set and edit applied.

The SDK comes with the node, so this runs where $POINTMAN_EXTENSIONS_SDK names its folder (the node's own
variable, for tests), and is skipped elsewhere.
"""

import importlib
import os
import sys
import tempfile
import types as pytypes
import unittest
from pathlib import Path
from unittest import mock

from pointman_trace.runs import Index

from helpers import Log, text, tool

ROOT = Path(__file__).resolve().parents[1]
SDK = os.environ.get("POINTMAN_EXTENSIONS_SDK")
TYPED = bool(SDK and Path(SDK, "pointman_extension_types.py").is_file())

if TYPED:
    sys.path.insert(0, SDK)
    import pointman_extension_types as types
    from pointman_extension_testing import InputRefused, StandInNode

    Span = types.WaterfallView.Span


@unittest.skipUnless(TYPED, "needs the node's typed Python SDK")
class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name, "runs")
        self.dir.mkdir()
        self.node = StandInNode(ROOT, settings={"folder": str(self.dir)}, machine="mac-test",
                                state_dir=Path(self.tmp.name, "state"))
        self.node.start()
        self.addCleanup(self.node.stop)

    def spans(self, view, name):
        b = view.block(name)
        return b.view.waterfall.spans if b is not None else []

    def test_a_thread_trace_opens_grows_and_shows_a_span(self):
        log = Log(self.dir, 1, "scene", thread=500)
        log.say("q1", tool("t1", "Bash", command="echo hi", description="Say hi"))
        view = self.node.open_pane("trace", {"thread": 500})
        self.assertEqual(view.format, types.PaneFormat.PANE_FORMAT_BLOCKS)
        view.wait(lambda v: len(self.spans(v, "tr")) == 4)    # the logs weren't read yet: the next tick sets it
        wf = view.block("tr").view.waterfall
        self.assertEqual((wf.title, wf.state), ("A thread", types.WaterfallView.State.STATE_LIVE))
        self.assertEqual(wf.machine, types.MachineId(value="mac-test"))     # the apps merge every machine's by it
        self.assertEqual([s.kind for s in wf.spans], [Span.Kind.KIND_AGENT, Span.Kind.KIND_RUN, Span.Kind.KIND_LLM,
                                                      Span.Kind.KIND_TOOL])
        self.assertEqual(wf.spans[3].agent, types.MemberRef(handle="scene"))

        sets = sum(m == "pane.set" for m, _ in view.frames)
        log.results(("t1", "hi", False))
        view.wait(lambda v: len(self.spans(v, "tr")) == 5)
        tool_span = next(s for s in self.spans(view, "tr") if s.id == "t:t1")
        self.assertEqual(tool_span.state, Span.State.STATE_OK)
        self.assertEqual(sum(m == "pane.set" for m, _ in view.frames), sets)     # it grew by edits, not again whole
        method, last = view.frames[-1]
        self.assertEqual(method, "pane.edit")
        self.assertIn("t:t1", [i.span.id for e in last.edits if e.put_items for i in e.put_items.items])

        view.input("open", block="tr", span="t:t1")
        tool_span = next(s for s in self.spans(view, "tr") if s.id == "t:t1")
        self.assertIn("echo hi", tool_span.input)
        self.assertEqual(tool_span.output, "hi")
        with self.assertRaises(InputRefused) as refused:
            view.input("tap", block="tr")
        self.assertIn("only opens", str(refused.exception))
        self.assertEqual(view.problems, [])

    def test_a_pane_opened_from_a_thread_shows_that_thread(self):
        Log(self.dir, 1, "scene", thread=500).say("q1", text("thinking"))
        view = self.node.open_pane("trace", context=types.PaneContext(thread=types.ThreadId(value="thr_500")))
        view.wait(lambda v: v.block("tr") is not None)
        self.assertEqual((view.title, view.block("tr").view.waterfall.title), ("Trace · A thread", "A thread"))

    def test_live_lists_running_traces(self):
        log = Log(self.dir, 1, "scene", thread=500)
        log.say("q1", text("thinking"))
        view = self.node.open_pane("live")
        view.wait(lambda v: v.block("t500") is not None and v.block("recent") is not None
                  and v.block("recent").view.traces.items)
        wf = view.block("t500").view.waterfall
        self.assertEqual((wf.compact, wf.thread.value, wf.state), (True, "thr_500",
                                                                   types.WaterfallView.State.STATE_LIVE))
        self.assertEqual([s.kind for s in wf.spans], [Span.Kind.KIND_AGENT, Span.Kind.KIND_RUN])
        # the week's list after it (EXT-70): a row per thread, this one running
        self.assertEqual([b.name for b in view.blocks], ["t500", "recent"])
        listed = view.block("recent").view.traces
        self.assertEqual((listed.days, listed.more), (7, False))
        row, = listed.items
        self.assertEqual((row.thread.value, row.title, row.state, row.end, row.agents, row.machines),
                         ("thr_500", "A thread", types.WaterfallView.State.STATE_LIVE, None,
                          [types.MemberRef(handle="scene")], [types.MachineId(value="mac-test")]))
        # nothing older to page back to; anything but open and more is refused
        view.input("more", block="recent", before=row.start.isoformat())
        with self.assertRaises(InputRefused) as refused:
            view.input("tap", block="recent")
        self.assertIn("pages back", str(refused.exception))
        self.assertEqual(view.problems, [])

    def test_commands_open_their_panes(self):
        done = self.node.run("trace.thread", {"thread": 17979})
        self.assertEqual(done.result["pane"], {"open": "trace", "target": {"thread": 17979}})
        done = self.node.run("trace.live", {})
        self.assertEqual(done.result["pane"], {"open": "live", "target": {}})

    def test_new_settings_start_it_again_on_them(self):
        other = Path(self.tmp.name, "other")
        other.mkdir()
        Log(other, 1, "loupe", thread=600).say("q1", text("over here"))
        self.node.restart({"folder": str(other)})
        view = self.node.open_pane("trace", {"thread": 600})
        view.wait(lambda v: v.block("tr") is not None)
        self.assertEqual(self.spans(view, "tr")[0].id, "a:loupe")


@unittest.skipUnless(TYPED, "needs the node's typed Python SDK")
class TickTests(unittest.TestCase):
    def main(self):
        """The extension's module, as the node starts it (its manifest beside its __main__)."""
        fake = pytypes.ModuleType("__main__")
        fake.__file__ = str(ROOT / "pointman_trace" / "__main__.py")
        real, sys.modules["__main__"] = sys.modules["__main__"], fake
        try:
            return importlib.import_module("pointman_trace.__main__")
        finally:
            sys.modules["__main__"] = real

    def test_edits_that_wont_apply_send_it_all_next_tick(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Log(Path(tmp.name), 1, "scene", thread=500)
        log.say("q1", text("thinking"))
        m = self.main()
        self.addCleanup(m.state.update, dict(m.state))
        m.state.update(index=Index(Path(tmp.name), hours=10**6))
        sent = []

        class Pane:
            id = "pane_1"

            def set(self, blocks):
                sent.append("set")

            def edit(self, edits):
                sent.append("edit")
                raise ValueError("no block 'tr'")

            def set_title(self, title):
                pass

        pane = Pane()
        m.views[pane.id] = (pane, m.ThreadView(500))
        self.addCleanup(m.views.clear)
        with mock.patch.object(m.ext, "panes", lambda: [pane]), mock.patch.object(m.ext, "event"), \
                mock.patch.object(m.ext, "log"):
            m.tick()
            log.say("q2", tool("t1", "Bash", command="ls"))
            m.tick()                     # refused: the view starts again
            m.tick()
        self.assertEqual(sent, ["set", "edit", "set"])


if __name__ == "__main__":
    unittest.main()
