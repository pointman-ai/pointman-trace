"""The extension's process as the node runs it: JSON-RPC lines on stdin and stdout, through the Python SDK.

The SDK comes with the node, so this runs where $POINTMAN_EXTENSIONS_SDK names its folder (the node's own
variable, for tests), and is skipped elsewhere.
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from pointman_trace.trace import check

from helpers import Log, text, tool

ROOT = Path(__file__).resolve().parents[1]
SDK = os.environ.get("POINTMAN_EXTENSIONS_SDK")


@unittest.skipUnless(SDK and Path(SDK, "pointman_extension.py").is_file(), "needs the node's Python SDK")
class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([SDK, str(ROOT)])}
        self.proc = subprocess.Popen([sys.executable, "-m", "pointman_trace"], cwd=ROOT, env=env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lines: queue.Queue = queue.Queue()
        self.seen = []
        threading.Thread(target=self._read, daemon=True).start()
        self.n = 0
        self.call("initialize", {"protocol": 1, "node": "0.1.0", "machine": "mac-test",
                                 "settings": {"folder": str(self.dir)}})

    def tearDown(self):
        try:
            self.call("shutdown", {})
        finally:
            self.proc.stdin.close()
            self.proc.wait(5)
            self.proc.stdout.close()
            self.proc.stderr.close()
            self.tmp.cleanup()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(json.loads(line))

    def send(self, method, params, rid=None):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        if rid is not None:
            msg["id"] = rid
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def call(self, method, params):
        self.n += 1
        rid = f"n{self.n}"
        self.send(method, params, rid)
        return self.wait(lambda m: m.get("id") == rid)

    def wait(self, want, timeout=10.0):
        """The next message `want` takes; anything else before it is kept in self.seen."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                m = self.lines.get(timeout=0.2)
            except queue.Empty:
                continue
            self.seen.append(m)
            if want(m):
                return m
        self.fail(f"nothing came: {self.seen[-5:]}")

    def test_a_thread_trace_opens_grows_and_shows_a_span(self):
        log = Log(self.dir, 1, "scene", thread=500)
        log.say("q1", tool("t1", "Bash", command="echo hi", description="Say hi"))
        opened = self.call("pane.open", {"pane": "pane_1", "open": "trace", "target": {"thread": 500},
                                         "context": {}, "viewer": {"person": "james"}})
        result = opened["result"]
        self.assertEqual(result["format"], "blocks")
        content = result["blocks"]
        if content[0]["type"] == "empty":             # the logs weren't read yet: the next tick sets it
            content = self.wait(lambda m: m.get("method") == "pane.set")["params"]["blocks"]
        block, = content
        self.assertEqual(block["type"], "waterfall")
        self.assertIsNone(check(block))
        self.assertEqual(block["title"], "A thread")
        self.assertEqual([s["kind"] for s in block["spans"]], ["agent", "run", "llm", "tool"])

        log.results(("t1", "hi", False))
        patch = self.wait(lambda m: m.get("method") == "pane.patch")["params"]["patch"]
        paths = [op["path"] for op in patch]
        self.assertIn("/b/tr/spans/3/state", paths)
        self.assertIn("/b/tr/spans/-", paths)          # the next model call

        answer = self.call("pane.input", {"pane": "pane_1", "seq": 1,
                                          "event": {"kind": "open", "block": "tr", "span": "t:t1"}})
        self.assertEqual(answer.get("result"), {})
        shown = next(m for m in self.seen if m.get("method") == "pane.patch" and "/input" in json.dumps(m))
        self.assertEqual({op["path"] for op in shown["params"]["patch"]},
                         {"/b/tr/spans/3/input", "/b/tr/spans/3/output"})
        refused = self.call("pane.input", {"pane": "pane_1", "seq": 2, "event": {"kind": "tap", "block": "tr"}})
        self.assertIn("only opens", refused["error"]["message"])

    def test_live_lists_running_traces(self):
        log = Log(self.dir, 1, "scene", thread=500)
        log.say("q1", text("thinking"))
        opened = self.call("pane.open", {"pane": "pane_2", "open": "live", "target": {}, "context": {},
                                         "viewer": {"person": "james"}})
        content = opened["result"]["blocks"]
        if content[0]["type"] == "empty":
            content = self.wait(lambda m: m.get("method") == "pane.set")["params"]["blocks"]
        block, = content
        self.assertEqual((block["id"], block["compact"], block["thread"], block["state"]), ("t500", True, 500, "live"))
        self.assertEqual([s["kind"] for s in block["spans"]], ["agent", "run"])

    def test_commands_open_their_panes(self):
        self.send("command.start", {"id": "c1", "kind": "trace.thread", "params": {"thread": 17979}}, "s1")
        done = self.wait(lambda m: m.get("method") == "done")["params"]["result"]
        self.assertEqual(done["pane"], {"open": "trace", "target": {"thread": 17979}})
        self.send("command.start", {"id": "c2", "kind": "trace.live", "params": {}}, "s2")
        done = self.wait(lambda m: m.get("method") == "done" and m["params"]["id"] == "c2")["params"]["result"]
        self.assertEqual(done["pane"], {"open": "live", "target": {}})


if __name__ == "__main__":
    unittest.main()
