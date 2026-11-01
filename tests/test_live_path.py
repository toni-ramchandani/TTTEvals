"""The live code path, exercised against a fake model server.

The tests start a small HTTP server on this machine that speaks both Ollama's
API and the OpenAI-compatible API the cloud providers use. The real network
code in `helix/llm.py` talks to it exactly as it would to a real model, so
these tests cover the plumbing a live run depends on: streaming, reasoning
blocks kept off the screen, tool calling in both formats, embeddings, the
temperature retry, citation and refusal detection on paraphrased answers, the
agent's switches, judge parsing, and the error messages. No model, no key,
no internet. Run with `python run.py test`.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agentic import AGENT_DEFAULT, AGENT_NAIVE, live_system_prompt, live_tools
from agentic import answer as agent_answer
from evals.day3_judges import Verdict
from helix import REPAIRED, answer, build_index
from helix import llm
from helix.llm import ModelError, chat_with_tools, complete, embed, is_refusal

OLLAMA = replace(REPAIRED, offline=False, generation_model="ollama/fake", embedding_model="hash")
OPENAI = replace(OLLAMA, generation_model="openai/fake")


# ---------------------------------------------------------------------------
# The fake server
# ---------------------------------------------------------------------------
def _reply(messages: list[dict]) -> str:
    user = messages[-1]["content"]
    if "### Context" in user:
        cid = user.split("### Context", 1)[1].strip().splitlines()[0].strip("[]").split()[0]
        return f"<think>check the table</think>It is 25 days (see {cid})."
    if "GROUNDED or UNGROUNDED" in user:
        return "GROUNDED\nEvery claim is in the evidence."
    if "Reply with the score only" in user:
        return "4"
    if "Write 3 to 5 short numbered steps" in user:
        return "1. Find the office.\n2. Find the figure.\n3. Compare."
    if '"verdict"' in user:
        return '{"verdict": "yes"}'
    return "<think>easy</think>ready"


def _agent_step(messages: list[dict], tools: list[dict]) -> tuple[str, dict] | None:
    """A tiny tool-using policy: profile, then search, then finish."""
    names = {t["function"]["name"] for t in tools}
    seen = [m.get("tool_name") or m.get("name") for m in messages if m["role"] == "tool"]
    if "get_user_profile" in names and "get_user_profile" not in seen:
        return "get_user_profile", {}
    if "search_handbook" not in seen:
        return "search_handbook", {"query": next(m["content"] for m in messages if m["role"] == "user")}
    hits = json.loads(next(m["content"] for m in messages if (m.get("tool_name") or m.get("name")) == "search_handbook"))
    return "finish", {"answer": "Eight weeks.", "citations": [hits[0]["ref"]]}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status: int, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, req, dict(self.headers)))
        if req.get("model") == "missing":
            return self._send(404, {"error": "model 'missing' not found, try pulling it first"})
        if self.path.startswith("/v1/") and self.headers.get("Authorization") != "Bearer test-key":
            return self._send(401, {"error": {"message": "Incorrect API key provided"}})
        if req.get("model") == "default-temperature-only" and "temperature" in req:
            return self._send(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0"}})
        if req.get("model") == "old-ollama" and isinstance(req.get("format"), dict):
            return self._send(400, {"error": "invalid format: expected \"json\""})
        if req.get("format") is not None and self.path == "/api/chat":
            return self._send(200, {"message": {"role": "assistant", "content": '{"verdict": "yes"}'},
                                    "done": True})

        if self.path == "/api/embed":
            return self._send(200, {"embeddings": [[float(len(t)), 1.0, 0.5] for t in req["input"]]})
        if self.path == "/v1/embeddings":
            return self._send(200, {"data": [{"index": i, "embedding": [float(len(t)), 1.0, 0.5]}
                                             for i, t in reversed(list(enumerate(req["input"])))]})

        if self.path == "/api/chat":  # Ollama's own format
            if req.get("tools"):
                name, args = _agent_step(req["messages"], req["tools"])
                return self._send(200, {"message": {"role": "assistant", "content": "",
                                                    "tool_calls": [{"function": {"name": name, "arguments": args}}]},
                                        "done": True})
            text = _reply(req["messages"])
            if not req.get("stream"):
                return self._send(200, {"message": {"role": "assistant", "content": text}, "done": True})
            lines = [{"message": {"content": p}, "done": False} for p in text.split(" ")]
            lines = [{**ln, "message": {"content": ln["message"]["content"] + " "}} for ln in lines[:-1]] + lines[-1:]
            lines.append({"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 7})
            return self._send(200, "\n".join(json.dumps(ln) for ln in lines).encode(), "application/x-ndjson")

        if self.path == "/v1/chat/completions":  # the OpenAI-compatible format
            if req.get("tools"):
                name, args = _agent_step(req["messages"], req["tools"])
                call = {"id": f"call_{len(req['messages'])}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}
                return self._send(200, {"choices": [{"message": {"role": "assistant", "content": None,
                                                                 "tool_calls": [call]},
                                                     "finish_reason": "tool_calls"}]})
            text = _reply(req["messages"])
            if not req.get("stream"):
                return self._send(200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
            events = [{"choices": [{"delta": {"content": p + " "}}]} for p in text.split(" ")]
            events.append({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
            body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return self._send(200, body.encode(), "text/event-stream")
        return self._send(404, {"error": "no such path"})


@pytest.fixture
def server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setenv("OLLAMA_API_BASE", base)
    monkeypatch.setenv("OPENAI_BASE_URL", base + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield srv
    srv.shutdown()


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
def test_ollama_streams_and_keeps_reasoning_off_screen(server):
    pieces = []
    text, meta = complete([{"role": "user", "content": "hi"}], OLLAMA, on_token=pieces.append)
    assert text == "ready" and "".join(pieces) == "ready"
    assert meta["completion_tokens"] == 7
    path, req, _ = server.requests[-1]
    assert path == "/api/chat" and req["options"]["num_ctx"] >= 8192 and req["options"]["temperature"] == 0


def test_openai_compatible_streams_over_sse(server):
    pieces = []
    text, meta = complete([{"role": "user", "content": "hi"}], OPENAI, on_token=pieces.append)
    assert text == "ready" and "".join(pieces).strip() == "ready"
    assert meta["prompt_tokens"] == 9


def test_retries_without_temperature_when_the_model_refuses_it(server):
    cfg = replace(OPENAI, generation_model="openai/default-temperature-only")
    text, _ = complete([{"role": "user", "content": "hi"}], cfg)
    assert text == "ready"
    assert "temperature" not in server.requests[-1][1]


def test_json_output_is_requested_from_ollama_with_the_schema(server):
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
    text, _ = complete([{"role": "user", "content": "judge"}], OLLAMA, json_schema=schema)
    assert json.loads(text) == {"verdict": "yes"}
    assert server.requests[-1][1]["format"] == schema
    old = replace(OLLAMA, generation_model="ollama/old-ollama")
    text, _ = complete([{"role": "user", "content": "judge"}], old, json_schema=schema)
    assert server.requests[-1][1]["format"] == "json", "an older Ollama gets plain JSON mode"


def test_json_mode_is_only_sent_where_it_is_known_to_work(server):
    schema = {"type": "object"}
    complete([{"role": "user", "content": "hi"}], OPENAI, json_schema=schema)
    assert "response_format" not in server.requests[-1][1], "a non-OpenAI base gets no response_format"


def test_local_answers_and_a_cloud_judge_in_the_same_run(server):
    """Answers from Ollama, every judge from an API model, one command."""
    from evals.day2_evaluators import geval_score, geval_steps
    from evals.day3_judges import judge_groundedness
    from toolkits.cases import case_from_trace
    from toolkits.judge import JudgeStats, structured

    mixed = replace(OLLAMA, judge_model="openai/fake-judge", embedding_model="ollama/fake-embed")
    index = build_index(mixed)
    start = len(server.requests)
    t = answer("How many days of annual leave do I get?", index, mixed)
    answered = [(p, r.get("model")) for p, r, _ in server.requests[start:]]
    assert ("/api/chat", "fake") in answered, "the answer comes from Ollama"

    def judged(fn):
        before = len(server.requests)
        out = fn()
        calls = [(p, r.get("model")) for p, r, _ in server.requests[before:]]
        assert calls and all(c == ("/v1/chat/completions", "fake-judge") for c in calls), calls
        return out

    assert bool(judged(lambda: judge_groundedness(t, mixed)))
    case = case_from_trace(t)
    steps = judged(lambda: geval_steps("criteria", mixed))
    assert judged(lambda: geval_score(case, "criteria", steps, mixed, samples=2)).score == 4.0

    from pydantic import BaseModel

    class Verdict(BaseModel):
        verdict: str

    stats = JudgeStats()
    assert judged(lambda: structured('Reply as {"verdict": ...}', Verdict, mixed, stats)).verdict == "yes"
    assert stats.calls == 1


def test_pipeline_accepts_a_shortened_citation(server):
    index = build_index(OLLAMA)
    t = answer("How many days of annual leave do I get?", index, OLLAMA)
    assert "<think>" not in t.answer
    assert t.citations, "a citation by chunk id alone should count"
    assert not t.refused


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
def test_switches_remove_capabilities_from_the_live_agent():
    full = {t["function"]["name"] for t in live_tools(AGENT_DEFAULT)}
    naive = {t["function"]["name"] for t in live_tools(AGENT_NAIVE)}
    assert {"get_user_profile", "check_document_status"} <= full
    assert not {"get_user_profile", "check_document_status"} & naive
    search = next(t for t in live_tools(AGENT_NAIVE) if t["function"]["name"] == "search_handbook")
    assert "office" not in search["function"]["parameters"]["properties"]
    assert "superseded" in live_system_prompt(AGENT_DEFAULT)
    assert "superseded" not in live_system_prompt(replace(AGENT_DEFAULT, act_on_observations=False))


@pytest.mark.parametrize("cfg", [OLLAMA, OPENAI])
def test_live_agent_loop_records_each_step(server, cfg):
    index = build_index(cfg)
    steps = []
    t = agent_answer("How many weeks of parental leave do I get?", index, replace(AGENT_DEFAULT, base=cfg),
                     user_office="berlin", on_step=lambda n, tool, args: steps.append(tool))
    assert steps == ["get_user_profile", "search_handbook", "finish"]
    assert t.citations and t.answer == "Eight weeks."
    assert t.span("agent.run").attributes["tool_sequence"] == steps


def test_tool_results_reach_ollama_in_its_own_format(server):
    index = build_index(OLLAMA)
    agent_answer("How many weeks of parental leave do I get?", index, replace(AGENT_DEFAULT, base=OLLAMA))
    last = [req for path, req, _ in server.requests if path == "/api/chat"][-1]
    tool_msgs = [m for m in last["messages"] if m["role"] == "tool"]
    assert tool_msgs and all("tool_name" in m and "tool_call_id" not in m for m in tool_msgs)
    calls = [m for m in last["messages"] if m.get("tool_calls")]
    assert all(isinstance(c["function"]["arguments"], dict) for m in calls for c in m["tool_calls"])


def test_live_agent_respects_the_step_budget(server):
    index = build_index(OLLAMA)
    t = agent_answer("How many weeks of parental leave do I get?", index,
                     replace(AGENT_DEFAULT, base=OLLAMA, max_steps=1))
    assert t.span("agent.run").attributes["budget_exhausted"]
    assert t.refused


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["ollama/fake-embed", "openai/fake-embed"])
def test_embeddings_in_order_and_never_written_to_disk(server, monkeypatch, model):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        monkeypatch.setattr(llm, "_MEMO", {})
        cfg = replace(OLLAMA, embedding_model=model)
        v = embed(["a", "bbb"], cfg)
        assert v.shape == (2, 3) and v[0][0] == 1.0 and v[1][0] == 3.0
        calls = len(server.requests)
        embed(["bbb", "a"], cfg)
        assert len(server.requests) == calls, "within one command a text is embedded once"
        assert not any(Path(tmp).rglob("*")), "nothing is written to disk"
        monkeypatch.setattr(llm, "_MEMO", {})  # a new command starts empty
        embed(["a"], cfg)
        assert len(server.requests) == calls + 1, "the next command asks the model again"


# ---------------------------------------------------------------------------
# Errors say what to do
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ollama_not_running(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", f"http://127.0.0.1:{_free_port()}")
    with pytest.raises(ModelError) as err:
        complete([{"role": "user", "content": "hi"}], OLLAMA)
    assert "not running" in str(err.value)


def test_ollama_model_not_pulled(server):
    with pytest.raises(ModelError) as err:
        complete([{"role": "user", "content": "hi"}], replace(OLLAMA, generation_model="ollama/missing"))
    assert "ollama pull missing" in str(err.value)


def test_missing_and_wrong_keys(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ModelError) as err:
        complete([{"role": "user", "content": "hi"}], OPENAI)
    assert "OPENAI_API_KEY" in str(err.value)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong")
    with pytest.raises(ModelError) as err:
        chat_with_tools([{"role": "user", "content": "hi"}], live_tools(AGENT_DEFAULT), OPENAI)
    assert "rejected" in str(err.value)


def test_unknown_provider_is_named():
    with pytest.raises(ModelError) as err:
        complete([{"role": "user", "content": "hi"}], replace(OLLAMA, generation_model="mystery/model"))
    assert "Unknown model provider 'mystery'" in str(err.value)


# ---------------------------------------------------------------------------
# Reading what a live model says
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,label", [
    ("GROUNDED\nall supported", "GROUNDED"),
    ("**Verdict:** UNGROUNDED - the figure is not in the evidence", "UNGROUNDED"),
    ("The answer cites the wrong office.\nUNGROUNDED", "UNGROUNDED"),
    ("I cannot tell.", "UNPARSEABLE"),
])
def test_judge_verdicts_parse_wherever_the_word_is(raw, label):
    v = Verdict.parse(raw, "GROUNDED")
    assert v.label == label
    assert bool(v) is (label == "GROUNDED")


def test_refusal_variants_are_recognised():
    assert is_refusal("I don't have enough information in the retrieved policy documents.")
    assert is_refusal("The extracts do not contain the notice period for Dublin.")
    assert not is_refusal("You get 26 weeks of parental leave.")
