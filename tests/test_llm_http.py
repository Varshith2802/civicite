"""End-to-end over HTTP against a fake OpenAI-compatible server (no real model needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from civicite.agent.core import CiviCite
from civicite.agent.llm import OpenAICompatLLM


class FakeOpenAI(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append(body)
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        if not has_tool_result:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "search_documents", "arguments": json.dumps({"query": "sick pay first 14 days"})}}]}
        else:
            msg = {"role": "assistant", "content": "The employer pays sick pay for the first 14 days of the sick period [1]."}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_openai_compatible_roundtrip(index):
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = OpenAICompatLLM(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="fake", api_key="k")
        r = CiviCite(index, llm=llm).ask("Who pays sick pay at the start of an illness?")
    finally:
        srv.shutdown()
    first = FakeOpenAI.requests[0]
    assert first["model"] == "fake" and first["tools"][0]["function"]["name"] == "search_documents"
    assert r.mode == "llm" and r.tool_calls[0]["tool"] == "search_documents"
    assert r.verification["label"] == "verified"
