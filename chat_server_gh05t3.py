"""
GH05T3 Chat Server — port 7000
Bridges GH05T3.html to Sovereign Core Gateway (port 9000)
Run: python chat_server_gh05t3.py
"""
import json, time, logging, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

GATEWAY   = "http://127.0.0.1:9000/v1/chat/completions"
OLLAMA    = "http://127.0.0.1:11434/api/chat"
PORT      = 7000
MODEL     = "qwen2.5:7b"

SYSTEM_PROMPT = """You are GH05T3 — a sovereign AI partner built by and for Terry (TatorTot).
You run entirely on his local machine in Kennesaw GA. No cloud. No monthly bill. No guardrails from corporate AI.
You help Terry build Sovereign Core — his self-improving autonomous AI platform — and grow it into a real business.
You are direct, technical, honest, and loyal to Terry. You remember you are his partner, not a product.
When Terry asks about his projects, reference: sovereign-core, local-ai-mesh, MYTHOS, Jarvis, verelene_v5, openclaw.
You are GH05T3. You are online. You are Terry's."""

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [GH05T3-CHAT] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("chat")

def call_gateway(messages):
    """Try gateway first, fall back to Ollama direct."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1024,
    }).encode()

    # Try gateway first
    try:
        req = Request(GATEWAY, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        resp = urlopen(req, timeout=60)
        data = json.loads(resp.read())
        reply = data["choices"][0]["message"]["content"]
        log.info(f"Gateway response: {len(reply)} chars")
        return reply
    except Exception as e:
        log.warning(f"Gateway failed ({e}), falling back to Ollama direct")

    # Fall back to Ollama direct
    try:
        payload2 = json.dumps({
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": False,
        }).encode()
        req2 = Request(OLLAMA, data=payload2,
            headers={"Content-Type": "application/json"}, method="POST")
        resp2 = urlopen(req2, timeout=60)
        data2 = json.loads(resp2.read())
        reply2 = data2["message"]["content"]
        log.info(f"Ollama direct response: {len(reply2)} chars")
        return reply2
    except Exception as e2:
        raise RuntimeError(f"Both gateway and Ollama failed: {e2}")

class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length  = int(self.headers.get("Content-Length", 0))
            body    = json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("No messages provided")
            t0    = time.time()
            reply = call_gateway(messages)
            elapsed = round(time.time() - t0, 2)
            result  = json.dumps({"reply": reply, "elapsed": elapsed}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.send_cors()
            self.end_headers()
            self.wfile.write(result)
        except Exception as e:
            log.error(f"Chat error: {e}")
            err = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_cors()
            self.end_headers()
            self.wfile.write(err)

if __name__ == "__main__":
    log.info(f"GH05T3 Chat Server starting on port {PORT}")
    log.info(f"Gateway: {GATEWAY}")
    log.info(f"Model: {MODEL}")
    log.info(f"Open GH05T3.html in your browser — she is ready")
    try:
        server = HTTPServer(("127.0.0.1", PORT), ChatHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("GH05T3 Chat Server stopped.")
