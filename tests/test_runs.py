import json
import tempfile
import unittest
from pathlib import Path

from pointman_trace.runs import Index, Run, clean

from helpers import T0, Log, text, tool


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_turn_is_its_model_and_tool_calls(self):
        log = Log(self.dir, 7, "scene", thread=17979, cost_before=1.5,
                  prompt="You are @scene on Pointman:\n\n[18083] mention from @loupe in #extensions › "
                         "“agent-visualizers” (thread 17979):\n    look at this")
        run = Run(7, log.path)
        run.look()
        turn, = run.turns
        self.assertEqual(turn.id, "r:7")
        self.assertEqual(turn.start, T0)
        self.assertEqual(run.handle, "scene")
        self.assertEqual(turn.detail, "mention from @loupe in #extensions")
        self.assertEqual(run.titles, {17979: "agent-visualizers"})
        first, = turn.spans                    # the run has begun: its first model call is running
        self.assertEqual((first["kind"], first["state"], first["start_ms"]), ("llm", "running", T0))

        log.say("req1", text("Reading it"), tool("tu1", "mcp__pointman__read_thread", thread_id=17979),
                tool("tu2", "Bash", command="ls", description="List files"))
        run.look()
        llm, read, bash = turn.spans
        self.assertIs(llm, first)
        self.assertEqual(llm["detail"], "read_thread, Bash")
        self.assertEqual((llm["tokens_in"], llm["tokens_out"], llm["tokens_cached"]), (15, 20, 1000))
        self.assertIsNone(llm["end_ms"])
        self.assertEqual((read["name"], read["detail"], read["service"]), ("read_thread", "17979", "Pointman"))
        self.assertEqual((bash["name"], bash["detail"], bash["service"]), ("Bash", "List files", "shell"))

        log.results(("tu1", "the thread", False))
        run.look()
        self.assertEqual(llm["end_ms"], T0 + 1000)        # its last message
        self.assertEqual(read["state"], "ok")
        self.assertEqual(len(turn.spans), 3)              # Bash still running: no new model call yet
        log.results(("tu2", "permission denied", True))
        run.look()
        self.assertEqual(bash["state"], "error")
        nxt = turn.spans[-1]
        self.assertEqual((nxt["kind"], nxt["state"], nxt["start_ms"]), ("llm", "running", T0 + 2000))
        log.say("req2", text("Done."), tool("tu3", "mcp__pointman__post_message", thread_id=18000,
                                            text="/spawn helper claude\nDo the thing"))
        log.results(("tu3", "posted", False))
        log.say("req3", text("All done."))
        log.result(4.0, took=9000)
        run.look()
        self.assertEqual(turn.threads, {17979, 18000})
        self.assertEqual(run.spawns, ["helper"])
        self.assertEqual(turn.end, T0 + 9000)
        self.assertAlmostEqual(turn.cost, 2.5)
        self.assertTrue(all(s["state"] != "running" for s in turn.spans))
        self.assertEqual(turn.texts[nxt["id"]][1], "Done.")
        self.assertEqual(turn.span()["state"], "ok")

    def test_the_tools_names_from_the_switch_read_as_the_old_ones_did(self):
        # create_post and get_thread name their thread as `thr_N` (or a number); a post's words are its blocks. Logs
        # from before the switch keep post_message and read_thread, which the test above reads.
        log = Log(self.dir, 9, "scene", thread=17979)
        run = Run(9, log.path)
        log.say("req1", tool("tu1", "mcp__pointman__get_thread", thread="thr_17979"))
        log.results(("tu1", "the thread", False))
        log.say("req2", tool("tu2", "mcp__pointman__create_post", thread="thr_18000",
                             blocks=["/spawn helper claude\nDo the thing", {"text": {"markdown": "/spawn other claude"}}]))
        log.results(("tu2", "posted", False))
        log.say("req3", text("All done."))
        log.result(1.0, took=5000)
        run.look()
        turn, = run.turns
        read = next(s for s in turn.spans if s["name"] == "get_thread")
        self.assertEqual((read["detail"], read["service"]), ("17979", "Pointman"))
        self.assertEqual(turn.threads, {17979, 18000})
        self.assertEqual(run.spawns, ["helper", "other"])

    def test_a_sub_agent_nests_under_its_task(self):
        log = Log(self.dir, 8)
        run = Run(8, log.path)
        log.say("r1", tool("task1", "Task", description="Find the bug", subagent_type="Explore"))
        log.say("s1", tool("sub1", "Grep", pattern="x"), parent="task1")
        log.results(("sub1", "found", False), parent="task1")
        log.say("s2", text("It's in x.py"), parent="task1")
        log.results(("task1", "It's in x.py", False))
        run.look()
        spans = {s["id"]: s for s in run.turns[0].spans}
        task = spans["t:task1"]
        self.assertEqual((task["kind"], task["name"], task["detail"]), ("task", "Find the bug", "Explore"))
        subs = [s for s in run.turns[0].spans if s["parent"] == "t:task1"]
        self.assertEqual([s["kind"] for s in subs], ["llm", "tool", "llm"])
        self.assertEqual(subs[0]["start_ms"], task["start_ms"])     # the sub-agent's first call starts with it
        self.assertTrue(all(s["end_ms"] is not None for s in subs))

    def test_a_resident_run_has_a_turn_a_wake(self):
        log = Log(self.dir, 9, cost_before=10.0)
        run = Run(9, log.path)
        log.say("a", text("one"))
        log.result(11.0, took=1000)
        log.say("b", text("two"), after=60_000)
        log.result(11.5)
        run.look()
        self.assertEqual([t.id for t in run.turns], ["r:9", "r:9.1"])
        self.assertAlmostEqual(run.turns[0].cost, 1.0)
        self.assertAlmostEqual(run.turns[1].cost, 0.5)
        self.assertEqual(run.turns[1].span()["name"], "Run 9 · wake 2")

    def test_a_run_that_stops_mid_turn_is_cut_off(self):
        log = Log(self.dir, 10)
        run = Run(10, log.path)
        log.say("a", tool("tu1", "Bash", command="sleep 100"))
        run.look()
        log.end()
        run.look()
        turn = run.turns[0]
        self.assertTrue(run.ended)
        self.assertEqual(turn.span()["state"], "error")
        self.assertTrue(all(s["state"] == "error" and s["end_ms"] is not None for s in turn.spans))

    def test_the_index_costs_a_finished_run_from_its_session(self):
        a = Log(self.dir, 11, "scene", live=False)
        a.say("a", text("hi"))
        a.result(3.0)
        b = Log(self.dir, 12, "scene", live=False, startup=False, start=T0 + 100_000)
        b.say("b", text("again"))
        b.result(4.25)
        idx = Index(self.dir, hours=10**6)
        idx.refresh()
        self.assertEqual(sorted(idx.runs), [11, 12])
        self.assertAlmostEqual(idx.runs[11].turns[0].cost, 3.0)     # a new session: nothing before it
        self.assertAlmostEqual(idx.runs[12].turns[0].cost, 1.25)

    def test_the_meta_file_is_read_for_who_only(self):
        log = Log(self.dir, 13, "scene", thread=42, cost_before=2.0)
        run = Run(13, log.path)
        run.look()
        self.assertEqual((run.handle, run.thread, run.cost_before), ("scene", 42, 2.0))
        self.assertNotIn("never-read", repr(vars(run)))

    def test_clean_hides_credentials_and_cuts(self):
        self.assertEqual(clean("Authorization: Bearer abc.def.ghi"), "Authorization: Bearer [hidden]")
        self.assertEqual(clean("a basic test of bearer tokens"), "a basic test of bearer tokens")
        self.assertEqual(clean('{"GH_TOKEN": "ghx_short", "x-pointman-key": "5f0f"}'),
                         '{"GH_TOKEN": "[hidden]", "x-pointman-key": "[hidden]"}')
        self.assertEqual(clean("password=hunter2 and the token is fine"), "password=[hidden] and the token is fine")
        self.assertEqual(clean("sk-ant-" + "a" * 20), "[hidden]")
        self.assertEqual(clean("/Users/x/Documents/a-very-long-folder-name-that-goes-on-and-on-and-on/file.py"),
                         "/Users/x/Documents/a-very-long-folder-name-that-goes-on-and-on-and-on/file.py")
        self.assertEqual(clean("\x1b[31mred\x1b[0m  text\n"), "red text")
        self.assertEqual(len(clean("y " * 400)), 300)


class CleanTests(unittest.TestCase):
    """Credential shapes clean() hides (loupe's review, Pointman thread 17979 [18127])."""

    JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ."
           "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")

    def hidden(self, raw: str, *gone: str) -> str:
        out = clean(raw)
        self.assertIn("[hidden]", out, raw)
        for g in gone:
            self.assertNotIn(g, out, raw)
        return out

    def test_aws_access_keys(self):
        self.hidden("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE")
        self.hidden("the key is ASIAIOSFODNN7EXAMPLE here", "ASIAIOSFODNN7EXAMPLE")

    def test_private_keys(self):
        pem = ("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ\n"
               "QyNTUxOQAAACDb\n-----END OPENSSH PRIVATE KEY-----")
        self.assertEqual(clean(f"cat id: {pem} done"), "cat id: [hidden] done")
        self.assertEqual(clean(pem.split("-----END")[0]), "[hidden]")          # cut before its end
        self.hidden(json.dumps({"key": pem}), "b3BlbnNz", "QyNTUxOQ")             # inside a tool's input

    def test_passwords_in_urls(self):
        self.assertEqual(clean("git clone https://user:pass@github.com/x/y"),
                         "git clone https://[hidden]@github.com/x/y")
        self.assertEqual(clean("https://ghp_abcdefghijklmnopqrstuvwxyz0123@github.com/x"),
                         "https://[hidden]@github.com/x")
        self.assertEqual(clean("see https://github.com/x/y@main"), "see https://github.com/x/y@main")

    def test_jwts(self):
        self.assertEqual(clean(f"token {self.JWT} ok"), "token [hidden] ok")

    def test_stripe_and_other_service_keys(self):
        for key in ("sk_live_51HxQabcdefghijklmnop", "rk_test_51HxQabcdefghijk", "pk_live_51HxQabcdefgh",
                    "whsec_abcdefghijklmnop", "github_pat_11ABCDEFG0123456789_abcdefghij"):
            self.assertEqual(clean(f"x {key} y"), "x [hidden] y", key)
        self.hidden("STRIPE=sk_live_51HxQabcdefghijklmnop", "51HxQ")

    def test_names_that_contain_key_token_secret_or_pass(self):
        self.assertEqual(clean("DB_PASSWORD_PROD=hunter2 AWS_SECRET_ACCESS_KEY_2=abc/def"),
                         "DB_PASSWORD_PROD=[hidden] AWS_SECRET_ACCESS_KEY_2=[hidden]")
        self.assertEqual(clean('{"apiKeyValue": "v1", "MY_TOKEN_FILE": "/x"}'),
                         '{"apiKeyValue": "[hidden]", "MY_TOKEN_FILE": "[hidden]"}')
        self.assertEqual(clean('[["SECRET_KEY", "never-shown"], ["PATH", "/bin"]]'),
                         '[["SECRET_KEY", "[hidden]"], ["PATH", "/bin"]]')
        self.assertEqual(clean("DB_PASS=x"), "DB_PASS=[hidden]")

    def test_ordinary_text_stays(self):
        for s in ("19 passed: all good", "Tests passed=19", "a basic test of bearer tokens",
                  "session 38d8ce55-f390-5969-b346-d01750c96d04 ok", "read_thread 17979",
                  "git push origin main@{u}", "https://github.com/pointman-ai/pointman-trace/pull/3"):
            self.assertEqual(clean(s), s)

    def test_a_long_text_is_cut_between_words(self):
        raw = "x " * 1990 + "password=" + "s" * 50 + " " + "y" * 5000
        self.assertNotIn("sss", clean(raw, 10_000))


if __name__ == "__main__":
    unittest.main()
