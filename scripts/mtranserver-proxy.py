#!/usr/bin/env python3
"""OpenAI-compatible proxy for MTranServer.

Translates /v1/chat/completions requests into MTranServer /translate calls.

Usage:
    python3 mtranserver-proxy.py [--port 8990] [--mtran-url http://localhost:8989] [--mtran-token TOKEN]

Scene prompt should specify target language, e.g.:
    "translate to en"
    "translate to ja"
    "translate to zh-Hans"

The user message content is the text to translate.
"""

import argparse
import json
import re
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

DEFAULT_MTRAN_URL = "http://localhost:8989"
DEFAULT_PORT = 8990

# Populated from CLI args
mtran_url = DEFAULT_MTRAN_URL
mtran_token = ""


def parse_target_lang(system_prompt: str) -> str:
    """Extract target language from system prompt like 'translate to en'."""
    m = re.search(r'translate\s+to\s+([\w-]+)', system_prompt, re.IGNORECASE)
    return m.group(1) if m else "en"


def call_mtran(text: str, to_lang: str) -> str:
    body = json.dumps({"from": "auto", "to": to_lang, "text": text, "html": False}).encode()
    headers = {"Content-Type": "application/json"}
    if mtran_token:
        headers["Authorization"] = f"Bearer {mtran_token}"
    req = Request(f"{mtran_url}/translate", data=body, headers=headers, method="POST")
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("result", "")


def make_chat_response(content: str, model: str = "mtranserver") -> dict:
    # Wrap in {"candidates": [...]} to match vinput's expected JSON schema
    wrapped = json.dumps({"candidates": [content]})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": wrapped},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        messages = body.get("messages", [])
        system_prompt = ""
        user_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg.get("content", "")
            elif msg["role"] == "user":
                user_text = msg.get("content", "")

        to_lang = parse_target_lang(system_prompt)

        try:
            result = call_mtran(user_text, to_lang)
        except (URLError, Exception) as e:
            self.send_error(502, str(e))
            return

        resp = json.dumps(make_chat_response(result, body.get("model", "mtranserver")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.encode())

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            resp = json.dumps({
                "object": "list",
                "data": [{"id": "mtranserver", "object": "model", "owned_by": "mtranserver"}],
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.encode())
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[mtranserver-proxy] {fmt % args}")


def main():
    global mtran_url, mtran_token

    parser = argparse.ArgumentParser(description="OpenAI-compatible proxy for MTranServer")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mtran-url", default=DEFAULT_MTRAN_URL)
    parser.add_argument("--mtran-token", default="")
    args = parser.parse_args()

    mtran_url = args.mtran_url.rstrip("/")
    mtran_token = args.mtran_token

    server = HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    print(f"[mtranserver-proxy] Listening on http://127.0.0.1:{args.port}")
    print(f"[mtranserver-proxy] MTranServer: {mtran_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
