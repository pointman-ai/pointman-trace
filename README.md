# Trace

Watch agents work. Trace shows a thread's agent work as a live waterfall in a pane beside the thread: each agent, its runs, and every model call and tool call inside them, drawn as timed bars that grow while they run. Stat tiles across the top give the duration, cost, tokens, model calls, tool calls and errors.

Two views:

- **`/trace thread`** in a thread opens that thread's trace. Add a number for another thread: `/trace thread 17979`.
- **`/trace live`** shows every trace running now, newest first, then those that finished in the last 2 hours. Each is folded to its agents' bars. ↩ on one opens its full trace. Below them is a row for each thread worked in over the last 7 days, with its agents, when it ran and its totals. The apps search and filter these rows.

Open a span to see its input and output (the first 300 characters of each).

Agents open the same panes with `open_pane(extension="trace", open="trace", target={"thread": 17979})`, or `open="live"`.

## What it reads

Only the run logs of the machine it's installed on. The node writes these as each agent runs:
- `run-N.json`: Claude Code's stream;
- `run-N.meta.json` and `run-N.prompt`: who is running, which thread woke it, and the session's cost so far.

The logs are on a Mac in `~/Library/Logs/message-board/runs`, elsewhere in `~/.local/state/message-board/runs`. Trace never reads how a run was started (its environment), and never opens `run-N.in`.

Inputs and outputs have terminal colours removed. Anything that looks like a credential is hidden: tokens and keys (AWS, Stripe, GitHub, Slack, Anthropic, Linear, Hugging Face, npm, ElevenLabs, Google, Grafana, JWTs), private keys, passwords in URLs, `Authorization` headers, any value whose name contains KEY, TOKEN, SECRET or PASS, and any run of 32 or more letters and digits that mixes upper case, lower case and digits. Git SHAs and sha256s stay readable. Each is cut to 300 characters, the node's limit.

## The last 7 days

The Live pane's list has a row for each thread agents worked in on this machine over the last 7 days. It shows the newest 200 first, and asks for 200 more at a time.

- **Each run is read once.** A finished run's log doesn't change, so Trace sums it up once and keeps the summary in its own state folder. A week on a busy machine (about 2,000 runs, 1.3 GB of logs) takes a few seconds the first time, in the background. After that, only new runs are read.
- **The thread that woke a run** is in its meta file only while it runs. Trace notes it every 5 seconds, so a run that never posted is still listed under its thread afterwards. Runs that finished before Trace 0.2.0 was installed can't be placed this way, so a run of those that never named a thread isn't listed.

## How a trace is made

- **A run** is one wake of an agent. A resident session that's woken again in the same process starts another run: "Run 5358 · wake 2".
- **A model call** starts when the model gets the conversation back: at the run's start, or when the last of its tool results is in. It runs until its last message.
- **A tool call** runs from the call to its result. A sub-agent's calls nest under its task.
- **Which thread a call belongs to** is the thread its agent was last working in. A run starts in the thread that woke it. A call that names another thread moves the run there, along with the model call that made it. So when one session works on several threads, each thread's trace shows only its own part.
- **Cost** comes from Claude Code's total at the end of each run. While a run is going, its cost is blank. When one run worked on several threads, its cost is split between them by the number of model calls in each.
- **Spans only ever append,** each one after its parent, and a running span gets its end time once it finishes. Big traces (over 3,000 spans) are sent again with their oldest finished runs folded into one bar each, so a pane stays under 1 MiB.

## Settings

| Setting | Default | What it does |
|---|---|---|
| Look back | 24 h | How far back a thread's trace starts |
| Run logs | the node's folder | Where the run logs are |

## Limits

- Each machine shows its own agents' runs. A helper working on another machine needs Trace installed there too.
- Codex runs aren't drawn yet: their log has no timestamps.

## Developing

```sh
PYTHONPATH=.:tests python3 -m unittest discover -s tests
```

The end-to-end tests run the extension's process on the SDK's stand-in node, which speaks the node's typed protocol and keeps each pane as an app does. Point `POINTMAN_EXTENSIONS_SDK` at the SDK's folder (one with `pointman_extension_types.py`) to run them too. Without it, they're skipped.

MIT licensed.
