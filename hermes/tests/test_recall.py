"""Recall filtering (Jev), notes over the vault socket, the per-turn log line, and the two tools."""
from __future__ import annotations

import http.server
import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from _load import load_plugin
from test_provider import FakeClient, make_provider

mod = load_plugin()
client_mod = __import__(f"{mod.__name__}.client", fromlist=["*"])

EPS = [{"id": f"owner_ep_20261107_{i:08d}", "timestamp": "2026-10-10T00:00:00Z", "subject": f"s{i}",
        "episode": f"episode {i} body", "atomic_facts": [{"content": f"fact {i}"}]} for i in range(4)]
NOTES = [{"path": f"Notes/n{i}.md", "heading": f"H{i}", "score": 0.5, "text": f"note {i} text"} for i in range(12)]


class ForgetClient(FakeClient):
    def delete_by_ids(self, req, timeout_s=None):
        self.calls.append(("delete_by_ids", req))
        return {"memory_type": "episode", "success": True}


def serve_vault(path: str, results, seen: list):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(4)

    def loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while not buf.endswith(b"\n"):
                    buf += conn.recv(4096)
                seen.append(json.loads(buf))
                conn.sendall(json.dumps(results).encode() + b"\n")

    threading.Thread(target=loop, daemon=True).start()
    return srv


class TestKeepIndices(unittest.TestCase):
    def test_cutoff_keeps_order(self):
        self.assertEqual(mod.keep_indices([0.9, 0.2, 0.5, 0.49], 0.5), [0, 2])

    def test_none_pass_keeps_best(self):
        self.assertEqual(mod.keep_indices([0.1, 0.3, 0.2], 0.5), [1])

    def test_empty(self):
        self.assertEqual(mod.keep_indices([], 0.5), [])


class RecallBase(unittest.TestCase):
    cfg: dict = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Path(self.tmp.name, "everos.json").write_text(json.dumps({"user_id": "owner", **self.cfg}))
        self.provider, self.fake = make_provider(self.tmp.name)
        self.fake.search_results = {"user": {"episodes": list(EPS), "profiles": [{"profile_data": {"a": "b"}}]}}
        self.real_jev = mod.jev_scores
        self.env = os.environ.get(mod.JEV_KEY_ENV)
        os.environ[mod.JEV_KEY_ENV] = "test-key"

    def tearDown(self):
        mod.jev_scores = self.real_jev
        if self.env is None:
            os.environ.pop(mod.JEV_KEY_ENV, None)
        else:
            os.environ[mod.JEV_KEY_ENV] = self.env
        self.tmp.cleanup()

    def recall(self):
        with self.assertLogs(mod.logger.name, "INFO") as logs:
            block = self.provider.prefetch("what car?", session_id="abcdefghijk")
        line = next(m for m in logs.output if "[everos] recall" in m)
        return block, line


class TestJevFilter(RecallBase):
    cfg = {"jev_enabled": True}

    def test_filters_episodes_keeps_profile(self):
        seen = []
        mod.jev_scores = lambda cfg, key, q, texts, t: seen.append((key, q, texts)) or [0.9, 0.1, 0.6, 0.2]
        block, line = self.recall()
        self.assertEqual(seen[0][0], "test-key")
        self.assertEqual(len(seen[0][2]), 4)
        self.assertIn("episode 0 body", block)
        self.assertIn("episode 2 body", block)
        self.assertNotIn("episode 1 body", block)
        self.assertIn('"a": "b"', block)  # the profile is never sent to Jev and always kept
        self.assertIn("session=abcdefgh eps=4/2 agent=0/0 vault=0/0 top=0.90 jev=ok", line)

    def test_failure_keeps_everything(self):
        def boom(*a):
            raise mod.JevError("timeout")
        mod.jev_scores = boom
        block, line = self.recall()
        for i in range(4):
            self.assertIn(f"episode {i} body", block)
        self.assertIn("eps=4/4", line)
        self.assertIn("jev=failopen:timeout", line)

    def test_missing_key_fails_open(self):
        os.environ.pop(mod.JEV_KEY_ENV)
        mod.jev_scores = lambda *a: self.fail("Jev called without a key")
        block, line = self.recall()
        self.assertIn("episode 3 body", block)
        self.assertIn("jev=failopen:nokey", line)


class TestJevOff(RecallBase):
    def test_disabled_never_calls(self):
        mod.jev_scores = lambda *a: self.fail("Jev called while disabled")
        block, line = self.recall()
        self.assertIn("episode 3 body", block)
        self.assertIn("jev=off", line)


class TestNotes(RecallBase):
    cfg = {"jev_enabled": True}

    def setUp(self):
        super().setUp()
        self.sockdir = tempfile.mkdtemp(prefix="v", dir="/tmp")
        self.sock = os.path.join(self.sockdir, "s.sock")
        self.seen = []
        self.srv = serve_vault(self.sock, {"results": NOTES}, self.seen)
        cfg = json.loads(Path(self.tmp.name, "everos.json").read_text())
        Path(self.tmp.name, "everos.json").write_text(json.dumps({**cfg, "vault_socket": self.sock}))
        self.provider, self.fake = make_provider(self.tmp.name)
        self.fake.search_results = {"user": {"episodes": list(EPS)}}

    def tearDown(self):
        self.srv.close()
        os.unlink(self.sock)
        os.rmdir(self.sockdir)
        super().tearDown()

    def test_notes_filtered_with_episodes_in_one_request(self):
        calls = []
        def score(cfg, key, q, texts, t):
            calls.append(texts)
            return [0.1] * 4 + [0.8 if i in (3, 7) else 0.1 for i in range(12)]
        mod.jev_scores = score
        block, line = self.recall()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 16)
        self.assertEqual(self.seen[0], {"query": "what car?", "top_k": 20})
        self.assertIn("Relevant notes:", block)
        self.assertIn("[Notes/n3.md] H3\nnote 3 text", block)
        self.assertNotIn("note 4 text", block)
        self.assertNotIn("Relevant past episodes:", block)
        self.assertIn("eps=4/0 agent=0/0 vault=12/2 top=0.80 jev=ok", line)

    def test_unfiltered_notes_capped(self):
        def boom(*a):
            raise mod.JevError("http500")
        mod.jev_scores = boom
        block, line = self.recall()
        self.assertIn("note 7 text", block)
        self.assertNotIn("note 8 text", block)
        self.assertIn("vault=12/8", line)

    def test_missing_socket_means_no_notes(self):
        self.srv.close()
        os.unlink(self.sock)
        open(self.sock, "w").close()  # a plain file: connect fails, recall goes on
        mod.jev_scores = lambda cfg, key, q, texts, t: [0.9] * len(texts)
        block, line = self.recall()
        self.assertNotIn("Relevant notes:", block)
        self.assertIn("episode 0 body", block)
        self.assertIn("vault=0/0", line)


class JevHandler(http.server.BaseHTTPRequestHandler):
    reply: bytes = b""
    status = 200
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append((dict(self.headers), body))
        self.send_response(self.status)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(self.reply)

    def log_message(self, *a):
        pass


class TestJevWire(unittest.TestCase):
    def setUp(self):
        JevHandler.seen = []
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), JevHandler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.cfg = mod.Config({"jev_url": f"http://127.0.0.1:{self.srv.server_port}/v1/systemone"})

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_request_shape_and_scores(self):
        JevHandler.status = 200
        JevHandler.reply = json.dumps({"answers": {"item_1": {"noul": 0.7}, "item_2": {"noul": 0.2}}}).encode()
        self.assertEqual(mod.jev_scores(self.cfg, "k", "q?", ["a", "b"], 2.0), [0.7, 0.2])
        headers, body = JevHandler.seen[0]
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertTrue(headers["User-Agent"].startswith("hermes-everos/"))
        self.assertEqual(body["model"], "jev-1.13")
        self.assertIn("Question: q?", body["state"])
        self.assertIn("[Item 2]\nb", body["state"])
        self.assertEqual(sorted(body["questions"]), ["item_1", "item_2"])
        self.assertEqual(body["questions"]["item_1"]["type"], "noul")

    def test_http_error_and_bad_reply(self):
        JevHandler.status = 500
        JevHandler.reply = b"{}"
        with self.assertRaisesRegex(mod.JevError, "http500"):
            mod.jev_scores(self.cfg, "k", "q", ["a"], 2.0)
        JevHandler.status = 200
        JevHandler.reply = b'{"answers": {}}'
        with self.assertRaisesRegex(mod.JevError, "badreply"):
            mod.jev_scores(self.cfg, "k", "q", ["a"], 2.0)

    def test_timeout(self):
        hang = socket.socket()
        hang.bind(("127.0.0.1", 0))
        hang.listen(1)
        cfg = mod.Config({"jev_url": f"http://127.0.0.1:{hang.getsockname()[1]}/x"})
        try:
            with self.assertRaisesRegex(mod.JevError, "timeout"):
                mod.jev_scores(cfg, "k", "q", ["a"], 0.3)
        finally:
            hang.close()


class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Path(self.tmp.name, "everos.json").write_text(json.dumps({"user_id": "owner"}))
        self.provider, _ = make_provider(self.tmp.name)
        self.fake = ForgetClient()
        self.provider._client = self.fake

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, name, args):
        return json.loads(self.provider.handle_tool_call(name, args))

    def test_search_maps_episodes(self):
        self.fake.search_results = {"user": {"episodes": EPS}}
        out = self.call("everos_search", {"query": "car", "top_k": 2})
        req = self.fake.of("search")[0]
        self.assertEqual((req["user_id"], req["top_k"], req["method"]), ("owner", 2, "hybrid"))
        self.assertEqual(req["project_id"], Path(self.tmp.name).name.replace(" ", "_") or "default")
        self.assertEqual(out["episodes"][0], {"id": EPS[0]["id"], "date": "2026-10-10", "subject": "s0",
                                              "episode": "episode 0 body", "facts": ["fact 0"]})
        self.assertEqual(len(out["episodes"]), 2)

    def test_search_needs_query(self):
        self.assertIn("error", self.call("everos_search", {"query": " "}))

    def test_forget_body(self):
        out = self.call("everos_forget", {"ids": [EPS[0]["id"]], "reason": "outdated", "note": "moved"})
        self.assertEqual(out, {"deleted": 1})
        req = self.fake.of("delete_by_ids")[0]
        self.assertEqual(req["memory_type"], "episode")
        self.assertEqual(req["app_id"], "hermes")
        self.assertEqual(req["reason"], {"code": "outdated", "note": "moved"})

    def test_forget_validates(self):
        self.assertIn("error", self.call("everos_forget", {"ids": [], "reason": "outdated"}))
        self.assertIn("error", self.call("everos_forget", {"ids": ["x"] * 51, "reason": "outdated"}))
        self.assertIn("error", self.call("everos_forget", {"ids": ["x"], "reason": "because"}))
        self.assertEqual(self.fake.of("delete_by_ids"), [])

    def test_server_error_is_json(self):
        def bad(req, timeout_s=None):
            raise client_mod.EverosError(422, "VALIDATION", "not an episode memory id")
        self.fake.delete_by_ids = bad
        out = self.call("everos_forget", {"ids": ["x"], "reason": "other"})
        self.assertEqual(out["status"], 422)

    def test_unknown_tool(self):
        self.assertIn("error", self.call("nope", {}))


if __name__ == "__main__":
    unittest.main()
