"""A small local web server for trying the project. Python standard library only.

Run (from the project folder), then open http://127.0.0.1:8765
    .venv\\Scripts\\python app\\server.py
It only listens on this computer, not on the network.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pipeline import App

PORT = 8765
MAX_BODY = 10_000
PAGE = Path(__file__).parent / "index.html"


def read_settings(body):
    """Check what the browser sent. It's outside input, like anything else."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    temperature, top_p, tokens = body.get("temperature", 0.8), body.get("top_p", 0.9), body.get("tokens", 40)
    for value in (temperature, top_p):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("temperature and top_p must be numbers")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= 100:
        raise ValueError("tokens must be a whole number from 1 to 100")
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("temperature must be 0 to 2, and top_p above 0 and at most 1")
    return dict(mode=body.get("mode"), prompt=prompt, guards=body.get("guards") is not False, temperature=temperature,
                top_p=top_p, tokens=tokens, grant_delete=body.get("grant_delete") is True,
                approved_call=body.get("approved_call"))


def make_handler(app):
    lock = threading.Lock()   # connections are handled in parallel, but only one request uses the models at a time

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/info":
                self.send_json(200, app.info())
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= MAX_BODY:
                    return self.send_json(413, {"error": "request too large"})
                raw = self.rfile.read(length)
                if self.path == "/api/reset":
                    with lock:
                        app.reset()
                    return self.send_json(200, {"ok": True})
                if self.path != "/api/respond":
                    return self.send_json(404, {"error": "not found"})
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                settings = read_settings(body)
                with lock:
                    result = app.respond(**settings)
                self.send_json(200, result)
            except ValueError as error:   # includes bad JSON and an empty prompt with guards off
                self.send_json(400, {"error": str(error)})
            except Exception as error:    # a bug shouldn't drop the connection or stop the server
                self.send_json(500, {"error": f"Something went wrong on the server: {error}"})

        def log_message(self, *args):
            pass   # keep the terminal quiet

    return Handler


def make_server(port=PORT, app=None):
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app or App()))


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}  (press Ctrl+C to stop)")
    try:
        make_server().serve_forever()
    except KeyboardInterrupt:
        pass
