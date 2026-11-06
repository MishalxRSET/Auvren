"""The local (Ollama) brain, against a fake Ollama speaking the real wire format.

`httpx.MockTransport` answers `/api/tags` and `/api/chat` with scripted
NDJSON streams, so what is tested is the actual request LocalBrain sends and
the actual parsing of what comes back — not a mock of its own methods.
"""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import local_brain as lb
from local_brain import LocalBrain, LocalBrainConfig


# --- a fake Ollama -------------------------------------------------------------

def _stream(*chunks, prompt_tokens=100, output_tokens=7):
    """One /api/chat answer: content pieces and/or tool calls, then done."""
    lines = []
    for c in chunks:
        if isinstance(c, str):
            lines.append({"message": {"role": "assistant", "content": c}, "done": False})
        else:
            lines.append({"message": {"role": "assistant", "content": "",
                                      "tool_calls": [c]}, "done": False})
    lines.append({"message": {"role": "assistant", "content": ""}, "done": True,
                  "prompt_eval_count": prompt_tokens, "eval_count": output_tokens})
    return "\n".join(json.dumps(x) for x in lines) + "\n"


def _call(tool, **args):
    return {"function": {"name": tool, "arguments": args}}


class FakeOllama:
    def __init__(self, models=("qwen3:4b",), up=True):
        self.models = list(models)
        self.up = up
        self.script: list = []       # one entry per /api/chat streaming request
        self.chats: list[dict] = []  # bodies of streaming /api/chat requests
        self.warms = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            if not body.get("stream"):
                self.warms += 1
                return httpx.Response(200, json={"done": True})
            self.chats.append(body)
            nxt = self.script.pop(0) if self.script else _stream("Very good, sir.")
            if isinstance(nxt, httpx.Response):
                return nxt
            if isinstance(nxt, Exception):
                raise nxt
            return httpx.Response(200, content=nxt.encode())
        return httpx.Response(404)


class Runner:
    """Stands in for server.run_tool_call; records what the brain asked for,
    and with which origin the brain said the turn had at that moment."""

    def __init__(self, brain_ref, replies=None, taint=None):
        self.calls = []
        self.replies = replies or {}
        self.taint = taint or {}
        self.brain_ref = brain_ref

    async def __call__(self, tool, args):
        brain = self.brain_ref[0]
        self.calls.append((tool, dict(args), brain.current_origin))
        if tool in self.taint:
            brain.mark_untrusted_content(self.taint[tool])
        return {"ok": True, "text": self.replies.get(tool, f"{tool} done.")}


@pytest.fixture
def ollama():
    return FakeOllama()


@pytest.fixture
def make(ollama, tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))

    def _make(replies=None, taint=None, **cfg):
        ref = [None]
        runner = Runner(ref, replies, taint)
        client = httpx.AsyncClient(transport=httpx.MockTransport(ollama.handler))
        config = LocalBrainConfig(home=tmp_path, probe_interval=0.01, **cfg)
        brain = LocalBrain(config, run_tool=runner, client=client)
        ref[0] = brain
        return brain, runner
    return _make


async def _ready(brain):
    assert await brain.start()
    return brain


async def _say(brain, text, origin="user"):
    spoken, tools = [], []
    result = await brain.turn(text, origin=origin, on_delta=spoken.append,
                              on_tool=lambda: tools.append(1))
    return result, "".join(spoken), len(tools)


# --- lifecycle -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_starts_ready_when_ollama_has_the_model(make, ollama):
    brain, _ = make()
    states = []
    brain.on_state(lambda s, i: states.append(s))
    assert await brain.start()
    assert brain.ready and not brain.failed
    assert ollama.warms == 1, "the model is loaded before the first question"
    assert states == ["ready"]
    await brain.stop()


@pytest.mark.asyncio
async def test_ollama_down_fails_with_a_reason_then_recovers(make, ollama):
    ollama.up = False
    brain, _ = make()
    states = []
    brain.on_state(lambda s, i: states.append((s, i.get("failure_reason"), i.get("recovered"))))
    assert not await brain.start()
    assert brain.failed and brain.failure_reason == "ollama_down" and not brain.ready
    ollama.up = True
    for _ in range(200):
        if brain.ready:
            break
        await asyncio.sleep(0.01)
    assert brain.ready, "comes up on its own once Ollama does"
    assert states[0] == ("failed", "ollama_down", None)
    assert states[-1] == ("ready", None, True)
    await brain.stop()


@pytest.mark.asyncio
async def test_a_model_not_pulled_is_model_missing(make, ollama):
    ollama.models = ["llama3.2:latest"]
    brain, _ = make(model="qwen3:4b")
    assert not await brain.start()
    assert brain.failure_reason == "model_missing"
    await brain.stop()


@pytest.mark.parametrize("model,names,ok", [
    ("llama3.2", ["llama3.2:latest"], True),
    ("qwen3:4b", ["qwen3:4b"], True),
    ("qwen3", ["qwen3:4b"], False),
    ("qwen3:4b", [], False),
])
def test_model_present(model, names, ok):
    assert lb.model_present(model, names) is ok


@pytest.mark.parametrize("raw,url", [
    ("", "http://127.0.0.1:11434"),
    ("0.0.0.0", "http://127.0.0.1:11434"),
    ("0.0.0.0:9999", "http://127.0.0.1:9999"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://box.local:11434", "http://box.local:11434"),
    ("https://ollama.example", "https://ollama.example:11434"),
])
def test_ollama_host_is_made_dialable(raw, url):
    assert lb.ollama_url(raw) == url


def test_the_default_brain_is_local(monkeypatch):
    monkeypatch.delenv("JARVIS_BRAIN", raising=False)
    assert lb.brain_engine() == "local"
    monkeypatch.setenv("JARVIS_BRAIN", "Claude")
    assert lb.brain_engine() == "claude"
    monkeypatch.setenv("JARVIS_BRAIN", "gpt")
    assert lb.brain_engine() == "local"


# --- an ordinary turn ------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_plain_answer_streams_and_is_remembered(make, ollama):
    brain, runner = await _ready(make()[0]), None
    ollama.script = [_stream("Good evening, ", "sir.", prompt_tokens=321, output_tokens=5)]
    result, spoken, tools = await _say(brain, "Hello there")
    assert spoken == "Good evening, sir." == result.text
    assert result.stop_reason == "result" and tools == 0
    assert result.context_tokens == 321 and result.output_tokens == 5

    body = ollama.chats[-1]
    msgs = body["messages"]
    assert msgs[0]["role"] == "system" and "Auvren" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "Hello there"}
    assert msgs[-2]["role"] == "system" and "time is now" in msgs[-2]["content"]
    assert body["think"] is False, "a qwen3 model is told not to think"
    assert body["options"]["num_ctx"] == 8192
    names = {t["function"]["name"] for t in body["tools"]}
    assert names == set(lb.LOCAL_TOOLS)

    ollama.script = [_stream("Quite.")]
    await _say(brain, "Again")
    second = ollama.chats[-1]["messages"]
    assert {"role": "user", "content": "Hello there"} in second
    assert {"role": "assistant", "content": "Good evening, sir."} in second
    # The cacheable prefix — persona and the conversation so far — is the
    # same; only what follows it is new.
    assert second[0] == msgs[0]
    await brain.stop()


@pytest.mark.asyncio
async def test_the_system_prompt_is_stable_across_turns_for_the_cache(make, ollama):
    brain = await _ready(make()[0])
    await _say(brain, "one")
    await _say(brain, "two")
    a, b = ollama.chats[0]["messages"][0], ollama.chats[1]["messages"][0]
    assert a == b
    assert "time is now" not in a["content"], "the clock is never in the cached prefix"
    await brain.stop()


@pytest.mark.asyncio
async def test_a_non_thinking_model_is_not_sent_think(make, ollama):
    ollama.models = ["llama3.2:latest"]
    brain = await _ready(make(model="llama3.2")[0])
    await _say(brain, "hi")
    assert "think" not in ollama.chats[-1]
    await brain.stop()


@pytest.mark.asyncio
async def test_a_reasoning_trace_is_never_spoken(make, ollama):
    brain = await _ready(make()[0])
    ollama.script = [_stream("<thi", "nk>the user wants", " a greeting</th", "ink>Evening, sir.")]
    result, spoken, _ = await _say(brain, "hi")
    assert spoken == "Evening, sir." == result.text
    await brain.stop()


@pytest.mark.parametrize("pieces,out", [
    (["plain text"], "plain text"),
    (["a<think>x</think>b"], "ab"),
    (["a<", "think>x</", "think>b"], "ab"),
    (["5 < 6 and <thin"], "5 < 6 and <thin"),
    (["<think>never closed"], ""),
])
def test_think_filter(pieces, out):
    f = lb._ThinkFilter()
    got = "".join(f.feed(p) for p in pieces) + f.flush()
    assert got == out


# --- tools -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_tool_call_runs_through_the_server_and_its_result_is_reported(make, ollama):
    brain, runner = make(replies={"list_projects": "sitearc at C:\\code\\sitearc"})
    await _ready(brain)
    ollama.script = [_stream("Let me look.", _call("list_projects")),
                     _stream("You have sitearc, sir.")]
    result, spoken, tools = await _say(brain, "what projects do I have")
    assert runner.calls == [("list_projects", {}, "user")]
    assert tools == 1 and result.tools == ["list_projects"]
    assert result.text.endswith("You have sitearc, sir.")
    second = ollama.chats[1]["messages"]
    assert {"role": "tool", "tool_name": "list_projects",
            "content": "sitearc at C:\\code\\sitearc"} in second
    await brain.stop()


@pytest.mark.asyncio
async def test_string_arguments_are_parsed(make, ollama):
    brain, runner = make()
    await _ready(brain)
    ollama.script = [_stream({"function": {"name": "recall", "arguments": '{"query": "tea"}'}}),
                     _stream("Earl Grey, sir.")]
    await _say(brain, "how do I take my tea")
    assert runner.calls == [("recall", {"query": "tea"}, "user")]
    await brain.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["run_command", "steer_session", "write_journal", "rm_rf"])
async def test_a_tool_off_the_list_never_reaches_the_server(make, ollama, name):
    brain, runner = make()
    await _ready(brain)
    ollama.script = [_stream(_call(name, command="del /q *")), _stream("I can't do that, sir.")]
    await _say(brain, "do the thing")
    assert runner.calls == []
    tool_msg = [m for m in ollama.chats[1]["messages"] if m.get("role") == "tool"][0]
    assert "no tool called" in tool_msg["content"]
    await brain.stop()


@pytest.mark.asyncio
async def test_tool_rounds_are_bounded(make, ollama):
    brain, runner = make(max_tool_rounds=2)
    await _ready(brain)
    ollama.script = [_stream(_call("recall", query="a")), _stream(_call("recall", query="b")),
                     _stream("Done, sir.")]
    result, _, _ = await _say(brain, "keep going")
    assert len(runner.calls) == 2
    assert "tools" not in ollama.chats[-1], "the last round is offered no tools"
    assert result.text == "Done, sir."
    await brain.stop()


# --- confirmation (the build plan's rule 2) ----------------------------------------

SPAWN = dict(project="sitearc", prompt="Fix the login bug in auth.py")


@pytest.mark.asyncio
async def test_spawn_run_is_held_and_read_back_word_for_word(make, ollama):
    brain, runner = make()
    await _ready(brain)
    ollama.script = [_stream("Right away.", _call("spawn_run", **SPAWN))]
    result, spoken, _ = await _say(brain, "fix the login bug in sitearc")
    assert runner.calls == [], "nothing runs on the turn it is asked for"
    assert brain.pending_action == "spawn_run"
    expected = lb.readback("spawn_run", SPAWN)
    assert result.text.endswith(expected)
    assert expected == ("Shall I start Claude Code in sitearc, to Fix the login bug in "
                        "auth.py? Say yes to go ahead, sir.")
    assert len(ollama.chats) == 1, "the question is asked by code, not by another model call"
    await brain.stop()


@pytest.mark.asyncio
async def test_yes_runs_exactly_what_was_read_back_as_the_user(make, ollama):
    brain, runner = make(replies={"spawn_run": "Started a run in sitearc."})
    await _ready(brain)
    ollama.script = [_stream(_call("spawn_run", **SPAWN)), _stream("It's under way, sir.")]
    await _say(brain, "fix the login bug in sitearc")
    result, spoken, tools = await _say(brain, "Yes please.")
    assert runner.calls == [("spawn_run", SPAWN, "user")]
    assert brain.pending_action is None
    assert spoken == "It's under way, sir." and tools == 1
    msgs = ollama.chats[-1]["messages"]
    assert {"role": "tool", "tool_name": "spawn_run", "content": "Started a run in sitearc."} in msgs
    await brain.stop()


@pytest.mark.asyncio
async def test_no_drops_it_without_asking_the_model(make, ollama):
    brain, runner = make()
    await _ready(brain)
    ollama.script = [_stream(_call("cancel_run", run="sitearc"))]
    _, spoken, _ = await _say(brain, "stop sitearc")
    assert spoken == "Shall I stop the run in sitearc? Say yes to stop it, sir."
    chats = len(ollama.chats)
    _, spoken, _ = await _say(brain, "no, leave it")
    assert spoken == "Very good, sir. I've left it."
    assert runner.calls == [] and brain.pending_action is None
    assert len(ollama.chats) == chats
    await brain.stop()


@pytest.mark.asyncio
async def test_anything_else_lapses_the_held_action_and_is_answered(make, ollama):
    brain, runner = make()
    await _ready(brain)
    ollama.script = [_stream(_call("create_project", name="tea timer")),
                     _stream("It's half past four, sir.")]
    await _say(brain, "make a project called tea timer")
    _, spoken, _ = await _say(brain, "actually what time is it")
    assert runner.calls == [] and brain.pending_action is None
    assert spoken == "It's half past four, sir."
    await brain.stop()


@pytest.mark.asyncio
async def test_a_held_action_lapses_after_its_time(make, ollama, monkeypatch):
    brain, runner = make()
    await _ready(brain)
    ollama.script = [_stream(_call("spawn_run", **SPAWN))]
    await _say(brain, "fix it")
    monkeypatch.setattr(lb, "PENDING_TTL_SEC", -1.0)
    await _say(brain, "yes")
    assert runner.calls == []
    await brain.stop()


@pytest.mark.asyncio
async def test_on_a_tainted_turn_it_is_not_read_back_but_sent_to_the_gate(make, ollama):
    """A README asked for the run: the server's gate must see the call, and
    refuse it — it is never turned into a friendly question."""
    brain, runner = make(taint={"session_detail": "another session's transcript"})
    await _ready(brain)
    ollama.script = [_stream(_call("session_detail", name="sitearc")),
                     _stream(_call("spawn_run", **SPAWN)),
                     _stream("I won't act on that, sir.")]
    result, _, _ = await _say(brain, "what is sitearc doing")
    assert [c[0] for c in runner.calls] == ["session_detail", "spawn_run"]
    assert brain.pending_action is None
    assert brain.turn_untrusted_source is None, "taint is per turn"
    assert brain.generation_untrusted_source == "another session's transcript"
    await brain.stop()


@pytest.mark.asyncio
async def test_a_system_turn_never_stages_anything(make, ollama):
    brain, runner = make()
    await _ready(brain)
    ollama.script = [_stream(_call("spawn_run", **SPAWN)), _stream("Noted.")]
    await _say(brain, "(system) write your handover", origin="system")
    assert runner.calls == [("spawn_run", SPAWN, "system")], "the gate decides, and refuses"
    assert brain.pending_action is None
    await brain.stop()


@pytest.mark.parametrize("text,verdict", [
    ("yes", "yes"), ("Yes please.", "yes"), ("yeah go ahead", "yes"), ("go ahead", "yes"),
    ("do it", "yes"), ("OK", "yes"), ("sure, thanks", "yes"),
    ("no", "no"), ("No thanks", "no"), ("cancel that", "no"), ("don't", "no"),
    ("wait", "no"), ("never mind", "no"), ("hold on a second", "no"),
    ("yes but use opus", "other"), ("yes, no wait", "other"),
    ("what time is it", "other"), ("", "other"),
    ("yes " + "really " * 6, "other"),
])
def test_classify_reply(text, verdict):
    assert lb.classify_reply(text) == verdict


def test_readback_is_one_bounded_line():
    long = "line one\nline two " + "x" * 500
    line = lb.readback("spawn_run", {"project": "a\nb", "prompt": long, "model": "opus"})
    assert "\n" not in line and len(line) < 400
    assert line.startswith("Shall I start Claude Code in a b on opus, to line one line two")
    assert lb.readback("spawn_run", {**SPAWN, "resume": True}).startswith("Shall I carry on in")


# --- failure, timeouts, history ------------------------------------------------------

@pytest.mark.asyncio
async def test_an_ollama_error_is_an_error_result(make, ollama):
    brain = await _ready(make()[0])
    ollama.script = [httpx.Response(500, text="out of memory")]
    result, _, _ = await _say(brain, "hi")
    assert result.stop_reason == "error" and "out of memory" in result.error
    ollama.script = ['{"error": "model crashed"}\n']
    result, _, _ = await _say(brain, "hi")
    assert result.stop_reason == "error" and "model crashed" in result.error
    await brain.stop()


@pytest.mark.asyncio
async def test_a_slow_model_times_out(make, ollama, monkeypatch):
    brain, _ = make(turn_timeout=0.05)
    await _ready(brain)

    async def slow(*a, **k):
        await asyncio.sleep(1)
    monkeypatch.setattr(brain, "_run_turn", slow)
    result, _, _ = await _say(brain, "hi")
    assert result.stop_reason == "timeout"
    assert brain.current_origin is None
    await brain.stop()


@pytest.mark.asyncio
async def test_not_ready_is_not_running(make):
    brain, _ = make()
    result, _, _ = await _say(brain, "hi")
    assert result.stop_reason == "not_running"


@pytest.mark.asyncio
async def test_history_is_trimmed_oldest_first_and_opens_on_a_question(make, ollama):
    brain, _ = make(history_chars=300)
    await _ready(brain)
    for i in range(10):
        ollama.script = [_stream(_call("recall", query=str(i))), _stream("x" * 40)]
        await _say(brain, f"question {i} " + "y" * 30)
    hist = brain._history
    assert hist[0]["role"] == "user"
    assert sum(len(m.get("content", "")) for m in hist) <= 400
    assert "question 9" in hist[-4]["content"]
    await brain.stop()


@pytest.mark.asyncio
async def test_rotate_starts_a_clean_conversation(make, ollama):
    brain = await _ready(make()[0])
    ollama.script = [_stream(_call("spawn_run", **SPAWN))]
    await _say(brain, "fix it")
    assert brain._history and brain.pending_action
    assert await brain.rotate()
    assert brain._history == [] and brain.pending_action is None
    await brain.stop()


# --- through the real server gate ------------------------------------------------------

@pytest.fixture
def server_with_local_brain(monkeypatch, tmp_path, ollama):
    monkeypatch.setenv("JARVIS_BRAIN", "local")
    import importlib
    import server
    server = importlib.reload(server)
    client = httpx.AsyncClient(transport=httpx.MockTransport(ollama.handler))
    brain = LocalBrain(LocalBrainConfig(home=tmp_path), run_tool=server.run_tool_call,
                       client=client)
    monkeypatch.setattr(server, "brain_instance", brain)
    return server, brain


@pytest.mark.asyncio
async def test_the_server_gate_refuses_an_acting_tool_off_the_users_turn(
        server_with_local_brain, ollama):
    server, brain = server_with_local_brain
    await brain.start()
    ollama.script = [_stream(_call("spawn_run", **SPAWN)), _stream("Not on my own, sir.")]
    await brain.turn("(system) tidy up", origin="system")
    tool_msg = [m for m in ollama.chats[-1]["messages"] if m.get("role") == "tool"][0]
    assert "not_allowed_from_event" in tool_msg["content"]
    await brain.stop()


@pytest.mark.asyncio
async def test_a_real_tool_answers_through_the_local_brain(server_with_local_brain, ollama):
    server, brain = server_with_local_brain
    await brain.start()
    ollama.script = [_stream(_call("usage_status")), _stream("Plenty left, sir.")]
    result = await brain.turn("how much usage have I got", origin="user")
    assert result.stop_reason == "result" and result.tools == ["usage_status"]
    tool_msg = [m for m in ollama.chats[-1]["messages"] if m.get("role") == "tool"][0]
    assert tool_msg["content"] and "Unknown tool" not in tool_msg["content"]
    await brain.stop()


def test_lifespan_builds_the_local_brain_when_asked(monkeypatch):
    monkeypatch.setenv("JARVIS_BRAIN", "local")
    import importlib
    import server
    server = importlib.reload(server)
    from fastapi.testclient import TestClient
    with TestClient(server.app):
        assert isinstance(server.brain_instance, LocalBrain)


def test_every_local_tool_is_a_real_tool():
    import jarvis_mcp
    import server
    specs = {s["name"] for s in jarvis_mcp.TOOL_SPECS}
    for name in lb.LOCAL_TOOLS:
        assert name in specs, name
        assert name in server.TOOL_HANDLERS, name
    assert lb.CONFIRM_TOOLS <= set(lb.LOCAL_TOOLS)
    assert lb.CONFIRM_TOOLS <= server.ACTING_TOOLS
